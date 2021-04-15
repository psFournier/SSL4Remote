import os
import torch
from torch.utils.data.dataset import Dataset
from tqdm.notebook import tqdm
from time import time
import rasterio
import numpy as np
N_CHANNELS = 3

before = time()

indices = [1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37] + [
    2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29,31, 33, 35, 38]
img_stack = []
for idx in indices[:2]:
    filepath = '/home/pierre/Documents/ONERA/ai4geo/ISPRS_VAIHINGEN/top' \
               '/top_mosaic_09cm_area{}.tif'.format(idx)
    with rasterio.open(filepath) as image:
        im = image.read().astype(float).transpose(1, 2, 0) / 255
        im = im.reshape(-1, 3)
        img_stack.append(im)
all_pixels = np.vstack(img_stack)
centered = all_pixels - np.mean(all_pixels, axis=0)
cov = np.cov(centered, rowvar=False)
eig_vals, eig_vecs = np.linalg.eigh(cov)
print('eigenvals', eig_vals)
print('eivenvect', eig_vecs)
sort_perm = eig_vals[::-1].argsort()
eig_vals[::-1].sort()
eig_vecs = eig_vecs[:, sort_perm]
# get [p1, p2, p3]
m1 = np.column_stack((eig_vecs))
# get 3x1 matrix of eigen values multiplied by random variable draw from normal
# distribution with mean of 0 and standard deviation of 0.1
m2 = np.zeros((3, 1))
# according to the paper alpha should only be draw once per augmentation (not once per channel)
alpha = np.random.normal(0, [0.1,0.2,0.3])
# broad cast to speed things up
m2[:, 0] = alpha * eig_vals[:]
# this is the vector that we're going to add to each pixel in a moment
add_vect = np.matrix(m1) * np.matrix(m2)
print(add_vect)
print("time elapsed: ", time()-before)