# -*- coding: utf-8 -*-
import torch
import torch
import torch.nn.functional as F
import numpy as np
import os
from glob import glob
from network import net
import cv2
import argparse
from skimage import img_as_ubyte
import time


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def test(img):

    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda
    low_image = np.float32(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)) / 255.
    low_image = torch.from_numpy(low_image).permute(2, 0, 1)
    data_lowlight = low_image.cuda().unsqueeze(0)

    factor = 8 
    b,c,h,w = data_lowlight.shape
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    dark = F.pad(data_lowlight, (0, padw, 0, padh), 'reflect')

    _, result = net_Pre(dark)

    result = result[:, :, :h, :w]
    result = torch.clamp(result, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

    result_path = img.replace(img.split("/")[-2], 'result')
    save_img(result_path, img_as_ubyte(result))


if __name__ == '__main__':
    with torch.no_grad():
        parser = argparse.ArgumentParser()
        parser.add_argument('--cuda', type=str, default="0")
        parser.add_argument('--checkpoint', type=str, default="./pretrained_model/RetinexMac.pth")

        config = parser.parse_args()
        net_Pre = net.RMCMNet().cuda()
        net_Pre.load_state_dict(torch.load(config.checkpoint)["PreNet"])
        net_Pre.eval()

        test_list = glob("./Dataset/Data/DICM/low/*")
        times = 0
        for image in test_list:
            start = time.time()
            print(image)
            test(image)
            end_time = (time.time() - start)
            times += end_time
        print("Time: %.5f " % (times / len(test_list)))

            