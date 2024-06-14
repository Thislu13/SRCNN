import os

import cv2
import torchvision.transforms as transforms
from PIL import Image
CROP_SIZE = 320

def low_hight_img_write(raw_img, upscale_factor):
    crop_size = CROP_SIZE

    #CROP_SIZE 高分辨率图像中心裁剪
    hr_transform = transforms.Compose([
        transforms.CenterCrop(crop_size)
    ])

    lr_transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        transforms.Resize(crop_size, interpolation=Image.BICUBIC)
    ])

    hr_img = hr_transform(raw_img)
    lr_img = lr_transform(raw_img)

    return lr_img, hr_img

img_dir = r'/home/lu13/Documents/BSDS300/images/train/raw_img'
lr_dir = r'/home/lu13/Documents/BSDS300/images/train/lr_img'
hr_dir = r'/home/lu13/Documents/BSDS300/images/train/hr_img'
for img in os.listdir(img_dir):
    raw_path = os.path.join(img_dir, img)
    raw_img = Image.open(raw_path).convert('YCbCr')
    lr_path = os.path.join(lr_dir, img)
    hr_path = os.path.join(hr_dir, img)
    lr_img, hr_img = low_hight_img_write(raw_img, upscale_factor=3)

    lr_img.save(lr_path)
    hr_img.save(hr_path)
    # cv2.imwrite(lr_path, lr_img)
    # cv2.imwrite(hr_path, hr_img)
    # break