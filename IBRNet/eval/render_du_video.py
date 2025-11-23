# eval/render_du_video.py
# 학습된 IBRNet 모델을 사용해 뷰들을 렌더링하고
# 결과를 mp4 영상으로 저장하는 코드

import os
import sys
import time
import numpy as np
import imageio

import torch
from torch.utils.data import DataLoader

sys.path.append('../')

from config import config_parser
from ibrnet.sample_ray import RaySamplerSingleImage        # :contentReference[oaicite:0]{index=0}
from ibrnet.render_image import render_single_image        # :contentReference[oaicite:1]{index=1}
from ibrnet.model import IBRNetModel
from utils import colorize_np                              # :contentReference[oaicite:2]{index=2}
from ibrnet.projection import Projector                    # :contentReference[oaicite:3]{index=3}
from ibrnet.data_loaders import dataset_dict               # 학습과 동일한 DU 데이터셋 사용

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    parser = config_parser()
    args = parser.parse_args()
    args.distributed = False

    # 1) 모델 로드
    model = IBRNetModel(args, load_scheduler=False, load_opt=False)
    device = "cuda:0"

    eval_dataset_name = args.eval_dataset
    extra_out_dir = '{}/{}'.format(eval_dataset_name, args.expname)
    print("saving results to eval/{}...".format(extra_out_dir))
    os.makedirs(extra_out_dir, exist_ok=True)

    projector = Projector(device=device)

    assert len(args.eval_scenes) == 1, "현재는 eval_scenes에 하나의 씬만 지원"
    scene_name = args.eval_scenes[0]

    # 체크포인트 step별 폴더 (eval.py / render_llff_video.py와 동일 구조) 
    out_scene_dir = os.path.join(
        extra_out_dir,
        '{}_{:06d}'.format(scene_name, model.start_step),
        'videos'
    )
    os.makedirs(out_scene_dir, exist_ok=True)
    print("per-frame results will be saved to:", out_scene_dir)

    # 2) DU 데이터셋의 test split 사용
    test_dataset = dataset_dict[args.eval_dataset](args, 'test', scenes=args.eval_scenes)
    test_loader = DataLoader(test_dataset, batch_size=1)

    out_frames = []
    crop_ratio = 0.075  # render_llff_video.py에서 쓰던 것과 동일 비율로 바깥 테두리 잘라내기 :contentReference[oaicite:5]{index=5}

    total_num = len(test_loader)
    print("num test views:", total_num)

    for i, data in enumerate(test_loader):
        start = time.time()

        # src view 평균 이미지 저장 (디버깅용, 없어도 상관 없음)
        if 'src_rgbs' in data:
            src_rgbs = data['src_rgbs'][0].cpu().numpy()
            averaged_img = (np.mean(src_rgbs, axis=0) * 255.).astype(np.uint8)
            imageio.imwrite(
                os.path.join(out_scene_dir, f'{i:06d}_average.png'),
                averaged_img
            )
        else:
            averaged_img = None

        # 3) 한 뷰 렌더링
        model.switch_to_eval()
        with torch.no_grad():
            ray_sampler = RaySamplerSingleImage(data, device=device)
            ray_batch = ray_sampler.get_all()              # 모든 픽셀에 대한 ray, src_rgbs 등 :contentReference[oaicite:6]{index=6}
            featmaps = model.feature_net(
                ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2)
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
            )                                             # 

            torch.cuda.empty_cache()

        # coarse / fine 결과에서 RGB만 뽑기
        coarse_pred_rgb = ret['outputs_coarse']['rgb'].detach().cpu()
        coarse_pred_rgb_np = (
            255 * np.clip(coarse_pred_rgb.numpy(), a_min=0.0, a_max=1.0)
        ).astype(np.uint8)
        image_path_coarse = os.path.join(
            out_scene_dir, f'{i:06d}_pred_coarse.png'
        )
        imageio.imwrite(image_path_coarse, coarse_pred_rgb_np)

        if ret['outputs_fine'] is not None:
            fine_pred_rgb = ret['outputs_fine']['rgb'].detach().cpu()
            fine_pred_rgb_np = (
                255 * np.clip(fine_pred_rgb.numpy(), a_min=0.0, a_max=1.0)
            ).astype(np.uint8)
            image_path_fine = os.path.join(
                out_scene_dir, f'{i:06d}_pred_fine.png'
            )
            imageio.imwrite(image_path_fine, fine_pred_rgb_np)
            frame = fine_pred_rgb_np
        else:
            frame = coarse_pred_rgb_np

        # 4) 가장자리 crop 해서 프레임으로 사용 (옵션)
        if averaged_img is not None:
            h, w = averaged_img.shape[:2]
        else:
            h, w = frame.shape[:2]

        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        frame_cropped = frame[crop_h:h - crop_h, crop_w:w - crop_w, :]
        out_frames.append(frame_cropped)

        print(f'frame {i+1}/{total_num} done, time {time.time() - start:.3f}s')

    # 5) mp4로 저장
    video_path = os.path.join(extra_out_dir, f'{scene_name}.mp4')
    imageio.mimwrite(video_path, out_frames, fps=30, quality=8)
    print("video saved to:", video_path)


if __name__ == "__main__":
    main()
