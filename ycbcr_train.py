import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from itertools import product
from collections import OrderedDict, namedtuple



m = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112.0],
                  [112, -93.786, -18.214]])

def rgb2ycbcr(rgb):
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:, 0] += 16.
    ycbcr[:, 1:] += 128.
    ycbcr = np.round(ycbcr)
    return ycbcr.reshape(shape)


class SRCNN(nn.Module):
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

    def init_weights(self):
        self.conv1.weight.data.normal_(mean=0.0, std=0.001)
        self.conv1.bias.data.zero_()
        self.conv2.weight.data.normal_(mean=0.0, std=0.001)
        self.conv2.bias.data.zero_()
        self.conv3.weight.data.normal_(mean=0.0, std=0.001)
        self.conv3.bias.data.zero_()


class CustomDataset(Dataset):
    def __init__(self, root_dir_blur, root_dir_clear, transform=None):
        self.root_dir_blur = root_dir_blur
        self.root_dir_clear = root_dir_clear
        self.transform = transform
        self.images_blur = os.listdir(root_dir_blur)
        self.images_clear = os.listdir(root_dir_clear)
    def __len__(self):
        return len(self.images_blur)

    def rgb2Y(self,image):
        image_Ycbcr = rgb2ycbcr(image)
        img_y, img_cb, img_cr = cv2.split(image_Ycbcr)
        return  img_y

    def __getitem__(self, idx):
        img_blur_name = os.path.join(self.root_dir_blur, self.images_blur[idx])
        img_clear_name = os.path.join(self.root_dir_clear, self.images_clear[idx])
        img_blur_rgb = Image.open(img_blur_name).convert("RGB")
        img_clear_rgb = Image.open(img_clear_name).convert("RGB")

        img_blur_y = self.rgb2Y(np.array(img_blur_rgb))
        img_clear_y = self.rgb2Y(np.array(img_clear_rgb))

        img_blur_y = Image.fromarray(img_blur_y.astype('uint8'))
        img_clear_y = Image.fromarray(img_clear_y.astype('uint8'))

        if self.transform:
            img_blur_y = self.transform(img_blur_y)
            img_clear_y = self.transform(img_clear_y)

        return img_blur_y, img_clear_y

class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

params = OrderedDict(
epoch=[500]
,lr=[0.0001]
,batch_size=[2]
)



for run in RunBuilder.get_runs(params):

    dataset = CustomDataset(root_dir_blur='/home/lu13/Documents/BSDS300/images/train/lr_img',root_dir_clear='/home/lu13/Documents/BSDS300/images/train/hr_img',transform=transform)
    dataloader = DataLoader(dataset, batch_size=run.batch_size, shuffle=True)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    device = 'cuda'
    srcnn = SRCNN(1).to(device)
    criterion = nn.MSELoss()
    srcnn.init_weights()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
[
        {'params': srcnn.conv1.parameters()},
        {'params': srcnn.conv2.parameters()},
        {'params': srcnn.conv3.parameters(), 'lr': run.lr * 0.1}
        ], lr=run.lr)  # 前两层学习率lr， 最后一层学习率lr*0.1


    # print(srcnn)

    sel, target_sel = next(iter(dataloader))
    grid = torchvision.utils.make_grid(sel)

    sel, target_sel = sel.to(device), target_sel.to(device)
    grid = torchvision.utils.make_grid(sel)

    comment = f'-{run}'
    tb = SummaryWriter(comment=comment)
    tb.add_image('sel',grid)
    tb.add_graph(srcnn, sel)
    for epoch in range(run.epoch):
        running_loss = 0.0
        for idx, (images_blur, images_clear) in enumerate(dataloader):
            images_blur, images_clear = images_blur.to(device), images_clear.to(device)

            optimizer.zero_grad()
            outputs = srcnn(images_blur)
            loss = criterion(outputs, images_clear)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{run.epoch}, Loss: {running_loss/len(dataloader)}")
        tb.add_scalar('Loss', running_loss, epoch)
        for name, weight in srcnn.named_parameters():
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f'{name}.grad',weight.grad, epoch)
        if (epoch+1)%250 == 0:
            torch.save(srcnn.state_dict(), f'./pat/test_{run.lr}_{run.batch_size}_{epoch}epochs.pth')

tb.close()