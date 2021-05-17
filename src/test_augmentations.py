from torch_datasets import ChristchurchLabeled, AustinLabeled
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from utils import get_image_level_aug
import torch

augname='gamma'

plt.switch_backend("TkAgg")

fontsize = 18

dataset = AustinLabeled(
    data_path='/home/pierre/Documents/ONERA/ai4geo/miniworld_tif',
    idxs=list(range(5)),
    crop=256,
)

dataloader = DataLoader(
    dataset,
    batch_size=1
)

image, mask = next(iter(dataloader))

f, ax = plt.subplots(3, 3, figsize=(20, 20))

ax[1, 1].imshow(image[0].permute(1, 2, 0))
for i in range(3):
    for j in range(3):
        if i!=1 or j!=1:
            aug = get_image_level_aug(names=[augname], p=1)[1]
            aug_im, aug_mask = aug(image, mask)
            ax[i, j].imshow(aug_im[0].permute(1, 2, 0))
            print(torch.mean(image - aug_im)*255)
f.suptitle(augname)

plt.tight_layout()

plt.show()