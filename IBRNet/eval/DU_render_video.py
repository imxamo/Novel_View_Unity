# DU_render_video.py
# DU 데이터셋 + IBRNet으로 LLFF-style novel view trajectory 생성 및 렌더링

import os
import sys
import time
import json
import numpy as np
import imageio

import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from config import config_parser
from ibrnet.model import IBRNetModel
from ibrnet.render_image import render_single_image
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.projection import Projector
from ibrnet.data_loaders import dataset_dict


# ------------------------------------------------------------
# 1) transforms.json 로드해서 포즈 정보 가져오기
# ------------------------------------------------------------
def load_du_poses(scene_root):
    # DU 데이터셋에서 학습에 사용하던 transforms 위치와 맞추기
    #   ex) /home/ubuntu/datasets/classrooms/606/transforms.json
    tf_path = os.path.join(scene_root, "transforms.json")

    if not os.path.exists(tf_path):
        raise FileNotFoundError(f"transforms.json not found: {tf_path}")

    with open(tf_path, "r") as f:
        meta = json.load(f)

    frames = meta["frames"]
    poses = []
    for fr in frames:
        poses.append(np.array(fr["transform_matrix"], dtype=np.float32))
    poses = np.stack(poses, axis=0)  # (N,4,4)

    H, W = meta["h"], meta["w"]
    return poses, H, W


# ------------------------------------------------------------
# 2) 장면 중심/반지름 계산
# ------------------------------------------------------------
def get_scene_center_and_radius(poses):
    centers = poses[:, :3, 3]
    center = np.mean(centers, axis=0)

    # 반지름은 평균 거리
    radius = np.mean(np.linalg.norm(centers - center, axis=1))
    return center, radius


# ------------------------------------------------------------
# 3) LLFF-style 원형/나선 카메라 경로 생성
# ------------------------------------------------------------
def render_path_spiral(center, radius, N_frames=120, height=0.0, depth=0.0):
    render_poses = []

    for theta in np.linspace(0, 2 * np.pi, N_frames, endpoint=False):
        c2w = np.eye(4, dtype=np.float32)

        # 카메라 위치 (원 궤적)
        x = radius * np.cos(theta)
        y = height                     # 상하 이동은 고정 또는 수정 가능
        z = radius * np.sin(theta)

        pos = np.array([x, y, z]) + center

        # 카메라 바라보는 방향: center - pos
        forward = center - pos
        forward = forward / np.linalg.norm(forward)

        # world up
        up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # world to cam 구성
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        c2w[:3, 3] = pos

        render_poses.append(c2w)

    return np.stack(render_poses, axis=0)


# ------------------------------------------------------------
# 4) 프레임 렌더링
# ------------------------------------------------------------
def render_single_pose(model, projector, device, base_data, pose, args):
    # base_data는 test_loader에서 나온 원본 src info
    # → 레퍼런스로 쓰고, 내부 수정은 shallow copy로 수행
    data = dict(base_data)

    # 1) 기존 camera 벡터 (1,34) 중 batch 0을 꺼냄
    #    camera 구조:
    #      [0]   = H
    #      [1]   = W
    #      [2:18]  = intrinsics(4x4)
    #      [18:34] = c2w(4x4)
    cam = data["camera"][0].clone()       # shape: (34,)

    # 2) 새 pose(4x4)를 camera extrinsics 영역(18:34)에 반영
    new_c2w_flat = torch.from_numpy(pose.astype(np.float32).reshape(-1))  # (16,)
    cam[18:34] = new_c2w_flat

    # 3) 다시 batch dimension 붙여서 data["camera"]로 넣기
    data["camera"] = cam.unsqueeze(0)     # shape: (1,34)

    # 4) 이제 RaySamplerSingleImage가 이 새로운 카메라 포즈를 사용하여
    #    ray_o, ray_d를 새로 생성하고, projector 및 IBRNet이 이 포즈 기준으로 렌더링함
    with torch.no_grad():
        ray_sampler = RaySamplerSingleImage(data, device=device)
        ray_batch = ray_sampler.get_all()

        # src view feature
        featmaps = model.feature_net(
            ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2)
        )

        ret = render_single_image(
            ray_sampler=ray_sampler,
            ray_batch=ray_batch,
            model=model,
            projector=projector,
            chunk_size=args.chunk_size,
            det=True,
            N_samples=args.N_samples,
            inv_uniform=args.inv_uniform,
            N_importance=args.N_importance,
            white_bkgd=args.white_bkgd,
            featmaps=featmaps
        )

    # coarse/fine 중 fine이 있으면 fine 사용
    rgb = ret["outputs_fine"]["rgb"] if ret["outputs_fine"] is not None else ret["outputs_coarse"]["rgb"]
    rgb = rgb.detach().cpu().numpy()
    rgb = np.clip(rgb, 0.0, 1.0)

    return (rgb * 255).astype(np.uint8)


# ------------------------------------------------------------
# 5) main
# ------------------------------------------------------------
def main():
    parser = config_parser()
    args = parser.parse_args()
    args.distributed = False

    device = "cuda:0"
    projector = Projector(device=device)

    # 모델 로드
    model = IBRNetModel(args, load_scheduler=False, load_opt=False)
    model.switch_to_eval()

    # 어떤 씬?
    scene_name = args.eval_scenes[0]
    scene_root = os.path.join(args.rootdir, scene_name)

    # transforms 읽기
    poses, H, W = load_du_poses(scene_root)
    center, radius = get_scene_center_and_radius(poses)

    print("Scene center:", center)
    print("Scene radius:", radius)

    # 카메라 궤적 생성
    N_frames = 120 # ← 여기서 영상 길이 결정 (fps=30이면 4초)
    render_poses = render_path_spiral(center, radius, N_frames=N_frames)

    # test 데이터 하나 로드 (src view용)
    test_dataset = dataset_dict[args.eval_dataset](args, 'test', scenes=args.eval_scenes)
    test_loader = DataLoader(test_dataset, batch_size=1)
    data = next(iter(test_loader))

    # 출력 경로
    out_dir = f"eval/{args.eval_dataset}_{args.expname}/spiral_video"
    os.makedirs(out_dir, exist_ok=True)

    frames = []

    for i, pose in enumerate(render_poses):
        start = time.time()
        frame = render_single_pose(model, projector, device, data, pose, args)
        frames.append(frame)

        print(f"[{i+1}/{len(render_poses)}] frame done, {time.time() - start:.2f}s")

    # mp4 저장
    out_video = os.path.join(out_dir, f"{scene_name}_spiral.mp4")
    imageio.mimwrite(out_video, frames, fps=30, quality=8)

    print("Saved video:", out_video)


if __name__ == "__main__":
    main()
