from torch_datasets import IsprsVLabeled
from torch.utils.data import DataLoader
import albumentations as A
import torch
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from time import time
from torch.utils.data._utils.collate import default_collate
import random
torch.manual_seed(12)
from albumentations.pytorch import ToTensorV2

albu_aug = A.Compose([
    # D4 Augmentations
    A.RandomRotate90(p=1),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Transpose(p=0.5),
    A.Normalize(),
    ToTensorV2(transpose_mask=False)
])

def D4_transforms(image, mask):

    angle = random.choice([0,90,180,270])
    image = TF.rotate(image, angle=angle)
    mask = TF.rotate(mask, angle=angle)
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)
    if random.random() > 0.5:
        image = image.transpose(2,3)
        mask = mask.transpose(2,3)
    # image = image.transpose()
    # mask.transpose()

    return image, mask

# aug = A.MaskDropout(p=1, max_objects=1)

ds = IsprsVLabeled(
    data_path='/home/pierre/Documents/ONERA/ai4geo/ISPRS_VAIHINGEN',
    idxs=list(range(1)),
    crop=128,
    augmentations=ToTensorV2(transpose_mask=False)
)

def collate(batch):

    batch = [(elem["image"], elem["mask"]) for elem in batch]

    return default_collate(batch)

dl = DataLoader(
    ds,
    batch_size=8,
    num_workers=8,
    collate_fn=collate
)
before = time()

for epoch in range(10):
    for batch in dl:
        pass

print("time elapsed: ", time()-before)