# IBRNet/tools/colmap_to_transforms.py
import os, json, argparse, shutil
import numpy as np
from pathlib import Path
from PIL import Image

def qvec2rotmat(qw, qx, qy, qz):
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]
    ], dtype=np.float64)

def load_cameras_txt(path):
    cams = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            toks = line.strip().split()
            cam_id = int(toks[0]); model = toks[1]
            w = int(toks[2]); h = int(toks[3])
            params = list(map(float, toks[4:]))
            cams[cam_id] = dict(model=model, w=w, h=h, params=params)
    return cams

def load_images_txt(path):
    """
    Robust parser for COLMAP images.txt.
    Each image has TWO lines:
      line A: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID IMAGE_NAME
      line B: 2D-3D correspondences (we ignore this line)
    We must only parse line A.
    """
    imgs = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            toks = line.split()
            # Try to parse the first token as an integer image_id.
            # If it fails (e.g., "60.1403 ..."), it's the correspondences line -> skip.
            try:
                image_id = int(toks[0])
            except ValueError:
                # not a header line (likely the correspondences line)
                continue

            # Header line must have at least 10 tokens
            if len(toks) < 10:
                continue

            # Parse header
            qw, qx, qy, qz = map(float, toks[1:5])
            tx, ty, tz     = map(float, toks[5:8])
            cam_id         = int(toks[8])
            name           = toks[9]  # image filename

            imgs[image_id] = dict(
                qw=qw, qx=qx, qy=qy, qz=qz,
                t=np.array([tx, ty, tz], dtype=np.float64),
                name=name, cam_id=cam_id
            )
    return imgs


def build_transforms(scene_root, downscale=1):
    sparse0 = Path(scene_root) / "sparse" / "0"
    cam_txt = sparse0 / "cameras.txt"
    img_txt = sparse0 / "images.txt"

    cams = load_cameras_txt(cam_txt)
    imgs = load_images_txt(img_txt)

    frames = []
    # assume single camera intrinsics for this scene (usual for phone/camera)
    first_cam = cams[next(iter(cams))]
    W, H = first_cam["w"], first_cam["h"]

    # prepare (optional) downscale images to images_ds folder
    img_dir = Path(scene_root) / "images"
    if downscale > 1:
        out_img_dir = Path(scene_root) / f"images_{downscale}x"
        out_img_dir.mkdir(exist_ok=True, parents=True)
    else:
        out_img_dir = img_dir

    for k in sorted(imgs.keys()):
        it = imgs[k]
        R_wc = qvec2rotmat(it["qw"], it["qx"], it["qy"], it["qz"]).T
        t_wc = -R_wc @ it["t"]
        c2w = np.eye(4)
        c2w[:3,:3] = R_wc
        c2w[:3, 3] = t_wc

        fname = it["name"]
        src = img_dir / fname
        dst_rel = (out_img_dir.name + "/" + fname) if out_img_dir != img_dir else ("images/" + fname)

        # optional resize
        if out_img_dir != img_dir:
            out_path = out_img_dir / fname
            if not out_path.exists():
                im = Image.open(src).convert("RGB")
                newW = W // downscale; newH = H // downscale
                im = im.resize((newW, newH), Image.LANCZOS)
                out_path.parent.mkdir(exist_ok=True, parents=True)
                im.save(out_path, quality=95)

        frames.append({
            "file_path": dst_rel if dst_rel.startswith("images") else f"{dst_rel}",
            "transform_matrix": c2w.tolist()
        })

    # approximate intrinsics (SIMPLE_RADIAL / PINHOLE). Use fx from cameras.txt if available.
    # For SIMPLE_RADIAL (fx, cx, cy), params[0]=f, [1]=cx, [2]=cy typically (COLMAP stores normalized; depends on model).
    # To keep robust, we record only image size here; loaders often ignore top-level intrinsics.
    out = {
        "w": (W // downscale) if downscale>1 else W,
        "h": (H // downscale) if downscale>1 else H,
        "frames": frames
    }

    out_path = sparse0 / ("transforms_%dx.json" % downscale if downscale>1 else "transforms.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✔ Saved {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_root", required=True, help="e.g., C:/NovelView_IBRNet/data/classrooms/609")
    ap.add_argument("--downscale", type=int, default=1, help="2 or 4 to quickly make smaller images")
    args = ap.parse_args()
    build_transforms(args.scene_root, downscale=args.downscale)
