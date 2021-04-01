import os
import torch
from datasets import IsprsVaihingenUnlabeled
from torch.utils.data.dataset import Dataset
from tqdm.notebook import tqdm
from time import time
import rasterio
import numpy as np
N_CHANNELS = 3

before = time()
mean = torch.zeros(N_CHANNELS)
std = torch.zeros(N_CHANNELS)
print('==> Computing mean and std..')
indices = [1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37]
for idx in indices:
    filepath = '/home/pierre/Documents/ONERA/ai4geo/ISPRS_VAIHINGEN/top' \
               '/top_mosaic_09cm_area{}.tif'.format(idx)
    with rasterio.open(filepath) as image:
        im = image.read()
        for i in range(N_CHANNELS):
            mean[i] += im[i,:,:].mean()
            std[i] += im[i,:,:].std()
mean.div_(len(indices)*255)
std.div_(len(indices)*255)
print(mean, std)

print("time elapsed: ", time()-before)