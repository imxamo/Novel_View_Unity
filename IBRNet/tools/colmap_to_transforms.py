import os, json, argparse
import numpy as np
from pathlib import Path
from PIL import Image  # downscale할 때만 사용

def qvec2rotmat(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz),  2 * (qx * qy - qz * qw),      2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),      1 - 2 * (qx * qx + qz * qz),  2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),      2 * (qy * qz + qx * qw),      1 - 2 * (qx * qx + qy * qy)]
    ], dtype=np.float64)

def load_cameras_txt(path):
    """
    COLMAP cameras.txt 파싱.
    CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    """
    cams = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            toks = line.strip().split()
            cam_id = int(toks[0])
            model = toks[1]
            w = int(toks[2])
            h = int(toks[3])
            params = list(map(float, toks[4:]))
            cams[cam_id] = dict(model=model, w=w, h=h, params=params)
    return cams

def load_images_txt(path):
    """
    COLMAP images.txt 파싱.
    각 이미지 헤더 라인 형식:
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID IMAGE_NAME
    그 다음 줄(2D-3D 대응)은 무시.
    """
    imgs = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            toks = line.split()
            # 헤더 라인만 선택 (첫 토큰이 int 여야 함)
            try:
                image_id = int(toks[0])
            except ValueError:
                continue

            if len(toks) < 10:
                continue

            qw, qx, qy, qz = map(float, toks[1:5])
            tx, ty, tz     = map(float, toks[5:8])
            cam_id         = int(toks[8])
            name           = toks[9]

            imgs[image_id] = dict(
                qw=qw, qx=qx, qy=qy, qz=qz,
                t=np.array([tx, ty, tz], dtype=np.float64),
                name=name, cam_id=cam_id
            )
    return imgs

def build_transforms(scene_root, downscale=1):
    """
    scene_root: 예) /home/ubuntu/datasets/classrooms/609
      - scene_root/images/       : 원본 이미지들
      - scene_root/sparse/0/*.txt: COLMAP 결과
    downscale: 집컴으로 테스트 할 때 쓰던 것
    """
    scene_root = Path(scene_root)
    sparse0 = scene_root / "sparse" / "0"
    cam_txt = sparse0 / "cameras.txt"
    img_txt = sparse0 / "images.txt"

    cams = load_cameras_txt(cam_txt)
    imgs = load_images_txt(img_txt)

    # cameras.txt 의 W,H 그대로 사용
    first_cam = cams[next(iter(cams))]
    W, H = first_cam["w"], first_cam["h"]

    img_dir = scene_root / "images"

    # downscale > 1이면 images_2x, images_4x 같은 폴더에 저장
    if downscale > 1:
        out_img_dir = scene_root / f"images_{downscale}x"
        out_img_dir.mkdir(exist_ok=True, parents=True)
    else:
        out_img_dir = img_dir

    frames = []

    for k in sorted(imgs.keys()):
        it = imgs[k]

        # world 좌표계에서의 카메라 pose (c2w)
        R_wc = qvec2rotmat(it["qw"], it["qx"], it["qy"], it["qz"]).T
        t_wc = -R_wc @ it["t"]
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R_wc
        c2w[:3, 3]  = t_wc

        fname = it["name"]
        src = img_dir / fname

        # 출력 이미지 상대 경로
        if out_img_dir != img_dir:
            # images_2x/xxx.jpg 형태
            dst_rel = f"{out_img_dir.name}/{fname}"
            out_path = out_img_dir / fname
            if not out_path.exists():
                im = Image.open(src).convert("RGB")
                newW = W // downscale
                newH = H // downscale
                im = im.resize((newW, newH), Image.LANCZOS)
                out_path.parent.mkdir(exist_ok=True, parents=True)
                im.save(out_path, quality=95)
        else:
            # images/xxx.jpg
            dst_rel = f"images/{fname}"

        frames.append({
            "file_path": dst_rel,
            "transform_matrix": c2w.tolist()
        })

    out = {
        "w": (W // downscale) if downscale > 1 else W,
        "h": (H // downscale) if downscale > 1 else H,
        "frames": frames
    }

    # transforms.json 을 scene_root 에 저장 (IBRNet dataset.py 가 찾는 위치)
    out_path = scene_root / "transforms.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✔ Saved {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_root", required=True,
                    help="예) /home/ubuntu/datasets/classrooms/609")
    ap.add_argument("--downscale", type=int, default=1,
                    help="1(원본), 2 or 4 로 리사이즈")
    args = ap.parse_args()
    build_transforms(args.scene_root, downscale=args.downscale)
