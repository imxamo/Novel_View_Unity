import os
import json
import numpy as np
import imageio.v2 as imageio  # imageio가 없다면 pip install imageio

def load_transforms(room_dir):
    """
    room_dir: 예) C:/NovelView_IBRNet/data/classrooms/609

    반환:
      imgs   : (N, H, W, 3) float32, [0,1]로 정규화된 이미지
      poses  : (N, 4, 4) float32, world로 가는 c2w 행렬
      H, W   : 이미지 높이, 너비
      img_files: 이미지 파일 경로 리스트 (디버깅용)
    """
    # 1) transforms.json 읽기
    json_path = os.path.join(room_dir, "transforms.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"transforms.json이 {json_path} 에 없습니다.")

    with open(json_path, "r") as f:
        meta = json.load(f)

    W = meta["w"]
    H = meta["h"]

    frames = meta["frames"]

    imgs = []
    poses = []
    img_files = []

    # 2) 각 frame에 대해 이미지 + 포즈 로딩
    for fmeta in frames:
        rel_path = fmeta["file_path"]   # 예: "images_2x/img0001.jpg"
        img_path = os.path.join(room_dir, rel_path)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")

        # 이미지 읽기 (H, W, 3), 0~1 float32 로 변환
        img = imageio.imread(img_path)
        if img.ndim == 2:
            # 혹시 흑백이면 3채널로
            img = np.stack([img, img, img], axis=-1)
        img = img.astype(np.float32) / 255.0

        # 포즈 행렬 (4x4)
        c2w = np.array(fmeta["transform_matrix"], dtype=np.float32)

        imgs.append(img)
        poses.append(c2w)
        img_files.append(img_path)

    imgs = np.stack(imgs, axis=0)     # (N, H, W, 3)
    poses = np.stack(poses, axis=0)   # (N, 4, 4)

    return imgs, poses, H, W, img_files
