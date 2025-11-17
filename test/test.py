from dataloader_json import load_transforms
imgs, poses, H, W, paths = load_transforms(r"C:/NovelView_IBRNet/data/classrooms/609")
print(imgs.shape, poses.shape, H, W)
