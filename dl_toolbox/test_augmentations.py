from torch_datasets import ChristchurchLabeled, AustinLabeled
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from utils import get_image_level_aug
import torch
from torch.utils.data._utils.collate import default_collate


augname='gamma'

plt.switch_backend("TkAgg")

fontsize = 18

dataset = AustinLabeled(
    data_path='/home/pierre/Documents/ONERA/ai4geo/miniworld_tif',
    idxs=list(range(5)),
    crop=256,
    crop_step=128
)

def test_collate(batch):

    to_collate = [{k: v for k, v in elem.items() if k in ['image', 'mask']} for elem in batch]
    batch = default_collate(to_collate)

    return batch

dataloader = DataLoader(
    dataset=dataset,
    shuffle=False,
    collate_fn=test_collate,
    batch_size=1,
)

batch = next(iter(dataloader))
image = batch['image']
mask = batch['mask']

f, ax = plt.subplots(3, 3, figsize=(15, 15))

ax[1, 1].imshow(image[0].permute(1, 2, 0))
for i in range(3):
    for j in range(3):
        if i!=1 or j!=1:
            aug = get_image_level_aug(names=[augname], p=1)[0]
            aug_im, aug_mask = aug(image, mask)
            ax[i, j].imshow(aug_im[0].permute(1, 2, 0))
            print(torch.mean(image - aug_im)*255)
f.suptitle(augname)

plt.tight_layout()

plt.show()
