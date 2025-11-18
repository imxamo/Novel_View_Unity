# ibrnet/data_loaders/dataset.py

import os
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class TransformsDataset(Dataset):
    """
    IBRNet train.py에서 사용하는 형태에 맞춘 실내(classroom) 데이터셋.

    __init__(self, args, mode, scenes=None)
    - args.rootdir : C:/NovelView_IBRNet/data/classrooms  같은 루트 폴더
    - scenes       : '609' 또는 ['609'] 같은 씬 이름(들)

    지금은 한 개 씬만 쓴다고 가정하고, 첫 번째 씬만 사용.
    """

    def __init__(self, args, mode, scenes=None):
        super().__init__()
        self.args = args
        self.mode = mode  # 'train' 또는 'val' 등이 들어올 수 있음

        # ----- rootdir / scene 이름 정리 -----
        rootdir = Path(args.rootdir)  # config 에서 rootdir로 설정할 예정

        if scenes is None:
            # rootdir 안의 모든 폴더를 씬으로 사용 (여러 방일 때)
            scene_names = [d.name for d in rootdir.iterdir() if d.is_dir()]
        elif isinstance(scenes, str):
            # '609' 처럼 문자열 하나 들어온 경우
            scene_names = [scenes]
        else:
            # 리스트 등
            scene_names = list(scenes)

        # 첫 번째 씬만 사용
        self.scene_name = scene_names[0]
        self.scene_root = rootdir / self.scene_name

        # ----- transforms.json 로딩 -----
        json_path = self.scene_root / "transforms.json"
        if not json_path.exists():
            raise FileNotFoundError(f"transforms.json을 찾을 수 없습니다: {json_path}")

        with open(json_path, "r") as f:
            meta = json.load(f)

        self.W = int(meta["w"])
        self.H = int(meta["h"])

        frames = meta["frames"]

        self.image_paths = []
        self.c2w_list = []

        for fr in frames:
            rel = fr["file_path"]  # 예: "images_2x/img0001.jpg" 또는 "images/img0001.jpg"
            img_path = self.scene_root / rel
            if not img_path.exists():
                raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")
            self.image_paths.append(img_path)
            self.c2w_list.append(np.array(fr["transform_matrix"], dtype=np.float32))

        self.c2w = np.stack(self.c2w_list, axis=0)  # (N, 4, 4)
        self.n_images = len(self.image_paths)

        # ----- 이미지 미리 메모리에 올리기 -----
        imgs = []
        for p in self.image_paths:
            im = Image.open(p).convert("RGB")
            im = np.array(im, dtype=np.float32) / 255.0  # (H, W, 3)
            imgs.append(im)
        self.images = np.stack(imgs, axis=0)  # (N, H, W, 3)

        # intrinsic
        fx = fy = 0.5 * self.W
        cx = self.W * 0.5
        cy = self.H * 0.5
        self.intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        """
        IBRNetdml train
            - rgb: (H, W, 3)  타겟 이미지
            - rgb_path: str   타겟 이미지 경로
            - camera: (34,)   타겟 카메라 벡터
            - src_rgbs: (N_src, H, W, 3)
            - src_cameras: (N_src, 34)
            - depth_range: (2,)
        """

        H, W = self.H, self.W

        # ----- 타겟 뷰 -----
        tgt_rgb = self.images[idx]            # (H, W, 3), np.float32
        tgt_c2w = self.c2w[idx]               # (4, 4), np.float32
        tgt_path = str(self.image_paths[idx])

        # ----- 소스 뷰 인덱스 선택 -----
        all_ids = list(range(self.n_images))
        src_ids = [i for i in all_ids if i != idx]

        # num_source_views 만큼만 사용 (config의 args.num_source_views)
        N_src_cfg = getattr(self.args, "num_source_views", 4)
        if len(src_ids) > N_src_cfg:
            # 앞에서 N개만 사용 (원하면 랜덤샘플링 가능)
            src_ids = src_ids[:N_src_cfg]

        src_rgbs = self.images[src_ids]       # (N_src, H, W, 3), np.float32
        src_c2w = self.c2w[src_ids]           # (N_src, 4, 4), np.float32
        N_src = src_rgbs.shape[0]

        # ----- intrinsics에서 카메라 벡터 34차원 만들기 -----
        fx, fy, cx, cy = self.intrinsics  # (4,)
        
        # 4x4 K 행렬을 직접 구성
        K = np.eye(4, dtype=np.float32)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        
        intr_flat = K.reshape(-1)  # (16,)
        
        cam_vec = np.zeros((34,), dtype=np.float32)
        cam_vec[0] = H
        cam_vec[1] = W
        cam_vec[2:18] = intr_flat
        cam_vec[18:34] = tgt_c2w.reshape(-1)

        # 소스 카메라들 34차원 벡터 (N_src, 34)
        src_cam_vecs = np.zeros((N_src, 34), dtype=np.float32)
        for i in range(N_src):
            src_cam_vecs[i, 0] = H
            src_cam_vecs[i, 1] = W
            src_cam_vecs[i, 2:18] = intr_flat
            src_cam_vecs[i, 18:34] = src_c2w[i].reshape(-1)

        depth_range = np.array([0.5, 5.0], dtype=np.float32)

        sample = {
            "rgb":        torch.from_numpy(tgt_rgb),          # (H, W, 3)
            "rgb_path":   tgt_path,
            "camera":     torch.from_numpy(cam_vec),          # (34,)
            "src_rgbs":   torch.from_numpy(src_rgbs),         # (N_src, H, W, 3)
            "src_cameras": torch.from_numpy(src_cam_vecs),    # (N_src, 34)
            "depth_range": torch.from_numpy(depth_range),     # (2,)
            "scene_name": self.scene_name,
            "img_index":  idx,
        }

        return sample
