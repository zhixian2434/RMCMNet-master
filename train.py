# -*- coding: utf-8 -*-
from numpy import outer
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch
from torchvision import transforms
from torchvision import utils as vutils
import torchvision
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math
from torch import Tensor
from torch.autograd import Variable
import network
from network import net
from pytorch_msssim import SSIM, MS_SSIM
from torchvision import models
import Metric
import cv2
import torch.optim as optim
import kornia


class MyDataSet(Dataset):
    def __init__(self, low, normal, mode):
        super(MyDataSet, self).__init__()
        self.low = low
        self.normal = normal
        self.mode = mode

    def __getitem__(self, item):
        name = self.low[item].split("/")[-1]

        low_image = np.float32(cv2.cvtColor(cv2.imread(self.low[item]), cv2.COLOR_BGR2RGB)) / 255.
        normal_image = np.float32(cv2.cvtColor(cv2.imread(self.normal[item]), cv2.COLOR_BGR2RGB)) / 255.

        if self.mode == "train":
            h, w, _ = low_image.shape
            x = np.random.randint(0, h - 256 + 1)
            y = np.random.randint(0, w - 256 + 1)

            low_image = low_image[x:x+256, y:y+256, :]
            normal_image = normal_image[x:x+256, y:y+256, :]
        
        low_image = torch.from_numpy(low_image).permute(2, 0, 1)
        normal_image = torch.from_numpy(normal_image).permute(2, 0, 1)

        return low_image, normal_image, name

    def __len__(self):
        return len(self.low)
    

class SSIML1Loss(nn.Module):
    def __init__(self, channels):
        super(SSIML1Loss, self).__init__()
        self.l1_loss_func = nn.L1Loss()
        self.ssim = SSIM(data_range=1., size_average=True, channel=channels)
        self.alpha = 1.4

    def forward(self, output, target):
        l1_loss = self.l1_loss_func(output, target)
        ssim_loss = 1 - self.ssim(output, target)
        total_loss = ssim_loss
        return l1_loss + self.alpha * total_loss
    

class LightLoss(nn.Module):
    def __init__(self):
        super(LightLoss, self).__init__()
        self.l1_loss_func = nn.L1Loss()

    def forward(self, output, target):
        _, _, output_v = torch.split(kornia.color.rgb_to_hsv(output), 1, dim=1)
        _, _, target_v = torch.split(kornia.color.rgb_to_hsv(target), 1, dim=1)
        l1_loss = self.l1_loss_func(output_v, target_v)
        total_loss = l1_loss
        return total_loss
    

def train():

    train_dark_path = glob("./Dataset/Data/LOL/train/low/*") + glob("./Dataset/Data/LOL-v2/train/low/*") + glob("./Dataset/Data/LOLv2-SYS/Train/low/*")
    val_dark_path = glob("./Dataset/Data/LOL/val/low/*")
    train_gth_path = glob("./Dataset/Data/LOL/train/high/*") + glob("./Dataset/Data/LOL-v2/train/high/*") + glob("./Dataset/Data/LOLv2-SYS/Train/high/*")
    val_gth_path = glob("./Dataset/Data/LOL/val/high/*")

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    net_Pre = net.RMCMNet().cuda()

    train_datasets = MyDataSet(train_dark_path, train_gth_path, mode="train")
    val_datasets = MyDataSet(val_dark_path, val_gth_path, mode="val")
    train_data = DataLoader(train_datasets, batch_size=12, shuffle=True, num_workers=8, drop_last=True)
    val_data = DataLoader(val_datasets, batch_size=1, shuffle=True, num_workers=8, drop_last=True)

    print('dataset loaded!')
    print('%d images for training and %d images for evaluating.' % (len(train_data), len(val_data)))
    
    optimizer_Pre = optim.Adam(net_Pre.parameters(), lr=2e-4, betas=[0.9, 0.999], eps=0.00000001)
    scheduler = optim.lr_scheduler.StepLR(optimizer_Pre, step_size=500,
                                          gamma=0.8, last_epoch=-1)

    net_Pre.train()

    L_sl1 = SSIML1Loss(channels=3)
    L_light = LightLoss()

    min_loss = 99999
    for epoch in range(1000):
        index = 0
        # train
        for dark_image, target_image, name in train_data:
            index += 1
            optimizer_Pre.zero_grad()

            dark_image = dark_image.cuda()
            target_image = target_image.cuda()

            restore, result = net_Pre(dark_image)

            loss1 = L_sl1(result, target_image) + L_light(restore, target_image)
            loss = loss1

            loss.backward()
            optimizer_Pre.step()

        scheduler.step()
        # eval
        index = 0
        factor = 4
        val_loss = 0
        with torch.no_grad():
            net_Pre.eval()
            for dark_image, target_image, name in val_data:
                index += 1

                dark_image = dark_image.cuda()
                target_image = target_image.cuda()

                b,c,h,w = dark_image.shape
                H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
                padh = H - h if h % factor != 0 else 0
                padw = W - w if w % factor != 0 else 0
                dark = F.pad(dark_image, (0, padw, 0, padh), 'reflect')

                restore, result = net_Pre(dark)

                restore, result = restore[:, :, :h, :w], result[:, :, :h, :w]

                loss1 = L_sl1(result, target_image) + L_light(restore, target_image)
                loss = loss1

                val_loss += loss

            val_loss = val_loss / len(val_data)
            print(val_loss)
            state = {'PreNet': net_Pre.state_dict()}
            if val_loss < min_loss:
                min_loss = val_loss
                torch.save(state,  "./pretrained_model/BestEpoch.pth", _use_new_zipfile_serialization=False)
                print('saving the best epoch %d model with the loss %.5f' % (epoch + 1, min_loss))
        net_Pre.train()


if __name__ == "__main__":
    train()