import rasterio
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2
import torch

from augmentations import *


def histogramnormalization(
        im, removecentiles=2, tile=0, stride=0, vmin=1, vmax=-1, verbose=True, pivot=None
):
    if pivot is None:
        if verbose:
            print("extracting pivot")
        if tile <= 0 or stride <= 0 or tile > stride:
            allvalues = list(im.flatten())
        else:
            allvalues = []
            for row in range(0, im.shape[0] - tile, stride):
                for col in range(0, im.shape[1] - tile, stride):
                    allvalues += list(im[row : row + tile, col : col + tile].flatten())

        ## remove "no data"
        if vmin < vmax:
            allvalues = [v for v in allvalues if vmin <= v and v <= vmax]

        if verbose:
            print("sorting pivot")
        allvalues = sorted(allvalues)
        n = len(allvalues)
        allvalues = allvalues[0 : int((100 - removecentiles) * n / 100)]
        allvalues = allvalues[int(removecentiles * n / 100) :]

        n = len(allvalues)
        k = n // 255

        pivot = [0] + [allvalues[i] for i in range(0, n, k)]

    assert len(pivot) >= 255

    if verbose:
        print("normalization")
    out = np.uint8(np.zeros(im.shape, dtype=int))
    for i in range(1, 255):
        if i % 10 == 0 and verbose:
            print("normalization in progress", i, "/255")
        out = np.maximum(out, np.uint8(im > pivot[i]) * i)

    if verbose:
        print("normalization succeed")
    return np.uint8(out)

# src = rasterio.open('/home/pierre/Documents/ONERA/ai4geo/miniworld_tif/toulouse/train/0_x.tif')
src = rasterio.open('/d/pfournie/Documents/ai4geo/data/SemcityTLS_DL/BDSD_M_3_4_7_8.tif')
img = src.read(out_dtype=np.uint16)
# print(np.unique(img.reshape(img.shape[0], -1), axis=1, return_counts=True))

# src2 = rasterio.open('/home/pierre/Documents/ONERA/ai4geo/miniworld_tif/vienna/train/0_x.tif')
# img2 = src2.read(out_dtype=np.float32)[[0,1,2], 0:1000, 0:1000].transpose(1,2,0)

# histo_match = A.HistogramMatching(
#     reference_images=[img2],
#     read_fn=lambda x: x,
#     p=1)
# img3 = histo_match(image=img)['image']

# Stretch to minmax
out1 = np.copy(img).astype(np.float32)
for i in range(3):
    m, M = np.percentile(img[:, :, i], [2, 98])
    out1[:, :, i] = np.clip(((img[:, :, i] - m) / (M - m)), 0, 1) * 255
# out1 = np.uint8(out1)
## Attempt to stretch to min max in HSV space; abandonned
# out2 = np.copy(img/255).astype(np.uint8)
# out2 = cv2.cvtColor(out2, cv2.COLOR_RGB2HSV)
# m, M = np.percentile(out2[:, :, 2], [2, 98])
# out2[:, :, 2] = np.clip(((out2[:, :, 2] - m) / (M - m)), 0, 1) * 255
# out2 = cv2.cvtColor(np.uint8(out2), cv2.COLOR_HSV2RGB)

## Histogram eq

# r = histogramnormalization(np.int16(src.read(3)))
# g = histogramnormalization(np.int16(src.read(2)))
# b = histogramnormalization(np.int16(src.read(1)))
# out3 = np.stack([r, g, b], axis=2)

# out3 = A.functional.clahe(img)

f, ax = plt.subplots(1, 3, figsize=(25, 25))
ax[0].imshow(out1)
# ax[1].imshow(np.uint8(out2))
# ax[2].imshow(np.uint8(out3))

t = Brightness(bounds=(1.15, 1.2), p=1)
out2, _ = t(img=torch.from_numpy(out1.transpose(2,0,1)))
out2 = out2.numpy().transpose(1,2,0)
print(out1 == out2)
ax[1].imshow(out2)

plt.show()
