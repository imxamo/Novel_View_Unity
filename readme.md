IBRNet Novel View

1) IBRNet (CVPR’21): 
  다수의 소스 뷰에서 피쳐를 뽑아 “광선 기준”으로 샘플을 집계(aggregation)해 MLP로 색·밀도를 예측하는 일반화된 IBR.
  포즈가 있는 멀티뷰 사진만으로 미분가능한 볼륨 렌더링으로 학습하며, 새 장면에도 제로샷/소량 파인튠으로 일반화함. 
  https://arxiv.org/abs/2102.13090

2) 경로 구조 : 강의실별로 구분 (임시)
  /classrooms/
    roomA/
      images/                # 원본 jpg/png
      masks/   (optional)    # 있을 경우 사람/유리 반사 제외
      colmap/                # COLMAP DB/outputs
      transforms.json        # (LLFF/NeRF 스타일) 캘리브/포즈
      sparse_points.ply      # COLMAP 희소 점군
    roomB/
    ...

3) 본 프로젝트는 공식 IBRNet 코드 : pytorch로 구현되어 있음
  원본 레포 : https://github.com/googleinterns/IBRNet

4) 코드
  (A) prep_colmap_to_transforms.py — COLMAP → transforms.json & sparse depth
  (B) configs/indoor_ibrnet.yaml — 공통 설정(실내 최적화 옵션)
  (C) train_indoor_ibrnet.py — IBRNet + 실내 강화(깊이/노출/왜곡) 학습
  (D) dataloader_indoor.py — 실내 데이터 로더(희소깊이 지원)

5) 사용 순서(강의실 여러 개)
  # (1) 각 강의실 폴더에 대해 COLMAP -> transforms
  python prep_colmap_to_transforms.py --room_dir /classrooms/roomA
  python prep_colmap_to_transforms.py --room_dir /classrooms/roomB
  # ...
  
  # (2) 일괄 학습
  python train_indoor_ibrnet.py --classrooms_root /classrooms --config configs/indoor_ibrnet.yaml --mode train
  
  # (3) 뷰 경로(traj.json) 정의 후 렌더링(영상은 ffmpeg로 합치기)
  python train_indoor_ibrnet.py --classrooms_root /classrooms --room roomA \
    --config configs/indoor_ibrnet.yaml --mode render --traj ./trajectories/roomA_path.json --out ./renders/roomA
  ffmpeg -r 30 -i ./renders/roomA/%06d.png -c:v libx264 -pix_fmt yuv420p roomA_novelview.mp4


IBRNet Github 자료 : https://github.com/googleinterns/IBRNet

IBRNet은 사전학습 가중치를 사용하면 제로샷 성능이 좋음. 공식 레포 release에서 확인 가능. 
장면당 수 분~수십 분 파인튜닝으로 디테일을 올릴 수 있다고 함.

실내별로 조명·블라인드 상태를 최대한 일관되게 촬영하면 appearance 임베딩 의존도를 낮출 수 있어 결과가 더 선명하다.


6) 학습 순서 
  - 학습 준비
    1. 환경 세팅
      conda 가상환경 세팅
      경로 설정 : IBRNet 기본 세팅에 맞춤
      데이터폴더\data\classrooms\강의실명\images, sparse\0
      images 폴더에서 경로란 클릭, powershell 입력 엔터
      $cnt=1; gci *.jpg | sort $_.LastWriteTime | % { mv $_ ("img{0:D4}.jpg" -f $cnt); $cnt++ }
      하면 이미지 파일 일괄 rename 가능 
      colmap : 환경변수 path에 colmap, colmap\bin 추가
    
    2. colmap feature 추출
      conda powershell에서 :
        conda activate 가상환경명
        colmap
        으로 colmap 실행
      colmap에서 :
        File → New Project
        DB 파일 경로 입력 : C:\데이터폴더\data\classrooms\강의실명\database.db
        Images 경로 입력 : C:\데이터폴더\data\classrooms\강의실명\images -> save
        Processing → Feature Extract -> Extract
        Processing → Feature Matching -> Run
        Reconstruction → Start Reconstruction (오래 걸림)
        File → Extract model as txt
        강의실명\sparse\0\ 의 위치에 txt파일 옮기기 (db파일 제외)

    3. json 파일 생성
      IBRNet 깃허브에서 다운받은 경로 : IBRNet 폴더에서 colmap_to_transforms.py 난 tools 폴더 만듬
      conda activate ibrnet
      cd 코드 위치에서
      python tools\colmap_to_transforms.py --scene_root C:\데이터폴더\data\classrooms\강의실명\sparse\0 --downscale 2
      -> C:\데이터폴더\data\classrooms\강의실명\sparse\0\transforms_2x.json 생성
      * 다운스케일링 필요 없다면
      python tools\colmap_to_transforms.py --scene_root C:\NovelView_IBRNet\data\classrooms\609 --downscale 1
      모델 생성 시간과 영상 렌더링 시간 이슈로 다운 스케일 필요함. --downscale 2 적용

    4. 학습
      IBRNet에서 : python train.py --config configs/finetune_DU.txt -j 0
      다른 장소를 학습할 때 finetune_DU에 경로를 수정해 줄 것

    5. 비디오 렌더링 : 무거움
      cd ~/Novel_View/IBRNet
      mkdir -p eval
      nano eval/render_du_video.py
      cd ~/Novel_View/IBRNet
      python eval/DU_render_video.py --config configs/finetune_DU.txt

    

