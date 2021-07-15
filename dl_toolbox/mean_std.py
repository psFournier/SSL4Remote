import os
import torch
from torch.utils.data.dataset import Dataset
from tqdm.notebook import tqdm
from dl_toolbox.torch_datasets import Miniworld
from time import time
import rasterio
import numpy as np
N_CHANNELS = 3

before = time()
mean = torch.zeros(N_CHANNELS)
std = torch.zeros(N_CHANNELS)
num_im = 0
print('==> Computing mean and std..')
indices = [1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37]
for city_info in Miniworld.city_info_list:
    for i in range(city_info[2]):
        filepath = '/scratch_ai4geo/miniworld/{}/train/{}_x.png'.format(
            city_info[0], i)
        with rasterio.open(filepath) as image:
            im = image.read()
            for i in range(N_CHANNELS):
                mean[i] += im[i,:,:].mean()
                std[i] += im[i,:,:].std()
            num_im += 1
        if i>1000:
            break
mean.div_(num_im*255)
std.div_(num_im*255)
print(mean, std)