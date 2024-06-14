import os

import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np
import torch.nn as nn


class SRCNN(nn.Module):
    # def __init__(self, num_channels=1):
    #     super(SRCNN, self).__init__()
    #     self.relu = nn.ReLU()
    #     self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9//2)
    #     self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5//2)
    #     self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5//2)
    #
    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.relu(x)
    #     x = self.conv2(x)
    #     x = self.relu(x)
    #     x = self.conv3(x)
    #     return x
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(num_channels, 8, kernel_size=3, padding=3//2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=3//2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=3//2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=3//2)

        self.conv5 = nn.Conv2d(64, 32, kernel_size=5, padding=5//2)

        self.conv6 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(16, num_channels, kernel_size=3, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.conv7(x)
        return x

# 参数设置
model_path = "./pat/test_0.0001_2_499epochs.pth" # 模型路径
lr_path = "/home/lu13/Documents/BSDS300/images/test/lr_img/" # 自己的图像路径
sr_path = "/home/lu13/Documents/BSDS300/images/test/sr_img/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for img in os.listdir(lr_path):
    lr_img_path = os.path.join(lr_path, img)
    sr_img_path = os.path.join(sr_path, img)
    # 读取图片
    img = Image.open(lr_img_path ).convert('YCbCr') # PIL类型，转成YCbCr
    y, cb, cr = img.split() # 划分Y，cb，cr三个通道
    img_to_tensor = transforms.ToTensor() # 获得一个ToTensor对象

    # view中-1的含义是该维度不明，让其自动计算, input是由torch.Tensor([1,h,w])变成torch.Tensor([1,1,h,w])
    # 图像Tensor格式：(batch_size, channel, h, w)
    input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0]).to(device) # 将y通道变换成网络输入的tensor格式

    # 输出图片
    # device = 'cuda'
    model = SRCNN(1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    out = model(input).cpu() # 模型输出
    out_img_y = out[0].detach().numpy() # 返回新的三维张量并转成numpy
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255) # 取0-255内的值
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L') # numpy转成PIL
    out_img = Image.merge('YCbCr', [out_img_y, cb, cr]).convert('RGB') # 合并三个通道变成RGB格式

    out_img.save(sr_img_path)
