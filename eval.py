import numpy as np
from glob import glob
import utils
from skimage import img_as_ubyte
from natsort import natsorted

input_dir = glob("/home/liu/wzl/RetinexMac/result/*")
target_dir = glob("/home/liu/wzl/Dataset/Data/LOL/val/high/*")
test_list = natsorted(input_dir)
target_list = natsorted(target_dir)
psnrs = []
ssims = []
lpips = []

for i in range(len(test_list)):
    res_path = test_list[i]
    tar_path = target_list[i]
    name = res_path.split("/")[-1] 

    print(res_path.split("/")[-1], tar_path.split("/")[-1])

    restored = np.float32(utils.load_img(res_path)) / 255.
    target = np.float32(utils.load_img(tar_path)) / 255.

    psnrs.append(utils.PSNR(target, restored))
    ssims.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored)))
    lpips.append(utils.calculate_lpips(target, restored))

psnr = np.mean(np.array(psnrs))
ssim = np.mean(np.array(ssims))
lpip = np.mean(np.array(lpips))
print("PSNR: %.2f " % (psnr), "SSIM: %.3f " % (ssim), "LPIPS: %.3f " % (lpip))
