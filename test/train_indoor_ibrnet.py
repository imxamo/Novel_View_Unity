# -*- coding: utf-8 -*-
"""
각 강의실 폴더를 받아 IBRNet 학습/추론을 수행.
- IBRNet backbone은 googleinterns/IBRNet의 모듈을 import한다 가정
- 추가: (1) DS-NeRF식 sparse depth loss, (2) NeRF-W식 appearance embedding,
       (3) mipNeRF360식 distortion loss(약식), (4) optional normal smoothness
"""

"""
6. 그럼 “본질적으로” 뭐가 유지되고, 뭐가 바뀐 거냐?
그대로 유지되는 것 (IBRNet의 정체성 쪽)

5D 입력(3D 위치 + view 방향)에서 밀도/색 예측하는 IBRNet 네트워크 구조

여러 source view를 input으로 받아 feature aggregation 하는 방식

volume rendering으로 레이 따라 색 합성하는 방식
(공식 render_rays 대신 volume_render 직접 호출이지만 수식 자체는 동일 계열)

바뀐 것 (파이프라인 쪽)

데이터 포맷: LLFF → transforms.json + COLMAP 기반 클래스룸 씬

데이터로더: create_training_dataset → IndoorRayDataset

학습 래퍼 / 로스 / 로그 구조: IBRNetModel + Criterion → 직접 짠 루프 + DS-NeRF/mipNeRF360/Ref-NeRF 요소

7. 과제/보고서 관점에서 정리해주면

이렇게 표현하면 아주 정확하고 안전하다:

We adopt the official IBRNet backbone implementation from the public repository and
build a custom indoor training pipeline on top of it.

Instead of the original LLFF-style data loader and training wrapper (IBRNetModel, Criterion),
we design our own dataset loader that reads COLMAP-based transforms.json files for classroom scenes,
and implement a training loop that combines IBRNet with sparse depth supervision (DS-NeRF style),
distortion regularization (mip-NeRF360 style), and a simple normal-smoothness prior (Ref-NeRF inspired).

한국어로 쓰면 대략:

공개된 IBRNet 레포지토리의 백본 네트워크 구조는 그대로 사용하고,
LLFF 기반 공식 데이터로더와 학습 래퍼(IBRNetModel, Criterion) 대신
COLMAP에서 생성한 transforms.json과 실내 강의실 촬영 데이터를 읽는 커스텀 데이터로더 및
DS-NeRF, mip-NeRF360, Ref-NeRF의 아이디어를 일부 결합한 손실 함수를 사용하는
별도의 학습 파이프라인을 구성하였다.
"""
import os, json, glob, math, yaml
import torch, torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# IBRNet 레포 루트 : Novel_View/IBRNet
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
IBRNET_ROOT = os.path.join(THIS_DIR, "IBRNet")
if IBRNET_ROOT not in sys.path:
    sys.path.insert(0, IBRNET_ROOT)

# IBRNet repo 내 모듈 경로 예시 (실제 경로에 맞게 수정)
from ibrnet.model import IBRNet

class AppearanceEmbedding(nn.Module):
    # 이미지(프레임)별 임베딩 → 노출/WB 차이 흡수 (NeRF-W 아이디어)
    def __init__(self, n_images, dim=8):
        super().__init__()
        self.emb = nn.Embedding(n_images, dim)
        nn.init.normal_(self.emb.weight, mean=0, std=0.01)
    def forward(self, idx):
        return self.emb(idx)

def distortion_regularizer(weights, deltas):
    # mip-NeRF 360의 distortion loss(아이디어 차용, 약식)
    # weights: (B, N), deltas: (B, N) sample interval
    # L ~ sum_{i<j} w_i w_j |t_i - t_j|  -> 근사형
    with torch.no_grad():
        mids = torch.cumsum(deltas, dim=-1)
    loss = (weights * mids).sum(dim=-1) - (weights.sum(dim=-1) ** 2) / 2.0
    return loss.mean()

def depth_loss_from_sparse(ray_term_depth, sparse_depth, sigma=0.01):
    # DS-NeRF 스타일: 레이 종료 깊이 분포의 기대값을 sparse 깊이에 맞추도록
    # 여기선 간단히 L2(예: t* - t_sfm)
    return ((ray_term_depth - sparse_depth)**2 / (2*sigma**2 + 1e-6)).mean()

def load_scene(room_dir):
    with open(os.path.join(room_dir,'transforms.json'),'r') as f:
        meta = json.load(f)
    images = sorted(glob.glob(os.path.join(room_dir,'images','*')))
    id_map = {Path(p).name:i for i,p in enumerate(images)}
    # 필요 시 마스크 로드
    mask_dir = os.path.join(room_dir,'masks')
    use_masks = os.path.isdir(mask_dir)
    masks = {}
    if use_masks:
        for p in images:
            name = Path(p).name
            mpath = os.path.join(mask_dir, name)
            if os.path.exists(mpath):
                masks[name] = mpath
    # 희소깊이
    sparse = None
    npy = os.path.join(room_dir,'sparse_points.npy')
    if os.path.exists(npy):
        sparse = np.load(npy)  # (Ns,3) world points
    return meta, images, id_map, masks, sparse

def train_one_room(room_dir, cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    meta, images, id_map, masks, sparse_points = load_scene(room_dir)
    n_images = len(images)

    # IBRNet backbone
    net = IBRNet(num_source_views=cfg['train']['num_source_views'],
                 appearance_dim=cfg['train']['appearance_embedding_dim']).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg['train']['lr'])
    app_emb = AppearanceEmbedding(n_images, cfg['train']['appearance_embedding_dim']).to(device)

    # (예시) 사용자 데이터로부터 배치 생성기는 사용 환경에 맞게 구현/교체
    from dataloader_json import IndoorRayDataset, collate_fn
    train_loader = torch.utils.data.DataLoader(
        IndoorRayDataset(room_dir, meta, images, id_map, masks,
                         rays_per_step=cfg['train']['rays_per_step'],
                         num_source=cfg['train']['num_source_views']),
        batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)

    depth_weight = cfg['regularization']['depth_loss_weight'] if cfg['regularization']['use_sparse_depth'] else 0.0
    distort_w    = cfg['regularization']['distortion_weight'] if cfg['regularization']['use_distortion_loss'] else 0.0
    normal_w     = cfg['regularization'].get('normal_smooth_weight', 0.0)

    for epoch in range(cfg['train']['num_epochs']):
        net.train()
        for batch in train_loader:
            # batch: dict with 'rays', 'gt_rgb', 'src_imgs', 'src_feats', 'img_idx', 't_near', 't_far', ...
            for k in batch:
                if torch.is_tensor(batch[k]): batch[k]=batch[k].to(device)
            rays_o, rays_d = batch['rays']['o'], batch['rays']['d']
            img_idx = batch['img_idx']  # (B,)
            app = app_emb(img_idx)      # (B, app_dim)

            # IBRNet forward -> sample t, sigma, rgb, weights
            out = net.render(rays_o, rays_d, batch, appearance=app,
                             n_coarse=cfg['render']['n_coarse'], n_fine=cfg['render']['n_fine'])
            pred_rgb, weights, deltas, t_term = out['rgb'], out['weights'], out['deltas'], out['t_term']

            # photometric
            loss_rgb = (pred_rgb - batch['gt_rgb']).abs().mean()

            # sparse depth (DS-NeRF)
            loss_depth = torch.tensor(0.0, device=device)
            if depth_weight > 0 and batch.get('sparse_depth', None) is not None:
                loss_depth = depth_loss_from_sparse(t_term, batch['sparse_depth'],
                                                    sigma=cfg['regularization']['sparse_depth_sigma'])

            # distortion (mip-NeRF360)
            loss_distort = torch.tensor(0.0, device=device)
            if distort_w > 0:
                loss_distort = distortion_regularizer(weights, deltas)

            # (옵션) normal smoothness (Ref-NeRF hint) — 구현 단순화
            loss_normal = torch.tensor(0.0, device=device)
            if normal_w > 0 and out.get('normals_coarse', None) is not None:
                n = out['normals_coarse']
                loss_normal = (n[:,1:] - n[:,:-1]).abs().mean()

            loss = loss_rgb + depth_weight*loss_depth + distort_w*loss_distort + normal_w*loss_normal
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[{Path(room_dir).name}] epoch {epoch+1}/{cfg['train']['num_epochs']}")

    # 체크포인트 저장
    os.makedirs(os.path.join(room_dir,'ckpts'), exist_ok=True)
    torch.save({'net': net.state_dict(), 'app': app_emb.state_dict()},
               os.path.join(room_dir,'ckpts','ibrnet_indoor.pth'))

def render_room(room_dir, traj_json, out_dir, cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    meta, images, id_map, masks, _ = load_scene(room_dir)
    ckpt = torch.load(os.path.join(room_dir,'ckpts','ibrnet_indoor.pth'), map_location=device)
    net = IBRNet(num_source_views=cfg['train']['num_source_views'],
                 appearance_dim=cfg['train']['appearance_embedding_dim']).to(device)
    net.load_state_dict(ckpt['net']); net.eval()
    app_emb = AppearanceEmbedding(len(images), cfg['train']['appearance_embedding_dim']).to(device)
    app_emb.load_state_dict(ckpt['app'])

    with open(traj_json,'r') as f:
        traj = json.load(f)  # frames: [{c2w:[...]}]

    os.makedirs(out_dir, exist_ok=True)
    for i,fr in enumerate(traj['frames']):
        # rays 생성 → IBRNet 추론 → 이미지를 저장 (간단 버전)
        # 실제 구현은 repo의 render 스크립트를 활용 추천
        pass

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--classrooms_root", type=str,
                    default="C:/NovelView_IBRNet/data/classrooms")
    ap.add_argument("--config", default="configs/indoor_ibrnet.yaml")
    ap.add_argument("--mode", choices=["train","render"], default="train")
    ap.add_argument("--room", default=None)  # 특정 방만
    ap.add_argument("--traj", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r'))
    rooms = [os.path.join(args.classrooms_root, d) for d in os.listdir(args.classrooms_root)
             if os.path.isdir(os.path.join(args.classrooms_root, d))]
    if args.room: rooms = [os.path.join(args.classrooms_root, args.room)]

    if args.mode == "train":
        for r in rooms:
            train_one_room(r, cfg)
    else:
        assert args.room and args.traj and args.out
        render_room(os.path.join(args.classrooms_root, args.room), args.traj, args.out, cfg)
