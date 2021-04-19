from torch_datasets import IsprsVaihingenLabeled
from torch.utils.data import DataLoader
import albumentations as A
from matplotlib import pyplot as plt
import numpy as np

def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)

aug = A.CLAHE(p=1)

ds = IsprsVaihingenLabeled(
    data_path='/home/pierre/Documents/ONERA/ai4geo/ISPRS_VAIHINGEN',
    idxs=list(range(2)),
    crop=1024,
    augmentations=A.NoOp()
)

dl = DataLoader(
    ds,
    batch_size=1,
)

image, mask = next(iter(dl))
image = image[0,...].numpy()
mask = mask[0,...].numpy()
augmented = aug(image=image, mask=mask)
visualize(augmented['image'], augmented['mask'], original_image=image,
          original_mask=mask)
plt.show()