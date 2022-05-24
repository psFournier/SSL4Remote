import os
from torch.utils.data import Dataset
import torch
import dl_toolbox.augmentations as aug
from dl_toolbox.torch_datasets.commons import minmax
from dl_toolbox.utils import get_tiles
import rasterio
import imagesize
import numpy as np
from rasterio.windows import Window, bounds, from_bounds
from dl_toolbox.utils import MergeLabels, OneHot
import matplotlib.pyplot as plt

from dl_toolbox.torch_datasets.utils import *


class DigitanieDs(Dataset):

    DATASET_DESC = {

        'labels': [
            (0, 'other'),
            (1, 'bare ground'),
            (2, 'low vegetation'),
            (3, 'water'),
            (4, 'building'),
            (5, 'high vegetation'),
            (6, 'parking'),
            (7, 'pedestrian'),
            (8, 'road'),
            (9, 'railways'),
            (10, 'swimmingpool')
        ],
        'percentile2': [0.0357, 0.0551, 0.0674],
        'percentile98': [0.2945, 0.2734, 0.2662],
        'min': {
            'Toulouse': np.array([0, 0.0029, 0.0028, 0]),
            'Montpellier':np.array([1,1,1,2]),
            'Biarritz':np.array([-544, -503, 473, -652]),
            'Paris':np.array([0, 0, 0, 0]),
            'Strasbourg':np.array([89, 159, 202,92])
        },
        'max': {
            'Toulouse': np.array([1.5431, 1.1549, 1.1198, 2.0693]),
            'Montpellier':np.array([4911,4736,4753,5586]),
            'Biarritz':np.array([19498, 19829, 17822, 27880]),
            'Paris':np.array([19051, 16216, 15239, 29244]),
            'Strasbourg':np.array([4670, 4311, 4198, 6188])
        }
        'label_colors' : [
            (0,0,0),
            (100,50,0),
            (0,250,50),
            (0,50,250),
            (250,50,50),
            (0,100,50),
            (200,200,200),
            (200,150,50),
            (100,100,100),
            (200,100,200),
            (50,150,250)
        ]
    }
    color_map = {k: v for k, v in enumerate(DATASET_DESC['label_colors'])}

    def __init__(
            self,
            image_path,
            tile,
            fixed_crops,
            crop_size,
            crop_step,
            img_aug,
            read_window_fn,
            norm_fn,
            label_path=None,
            merge_labels=None,
            one_hot_labels=True,
            *args,
            **kwargs
            ):

        self.image_path = image_path
        self.label_path = label_path
        self.read_window_fn = read_window_fn
        self.norm_fn = norm_fn
        self.tile = tile
        self.crop_windows = list(get_tiles(
            nols=tile.width, 
            nrows=tile.height, 
            size=crop_size, 
            step=crop_step,
            row_offset=tile.row_off, 
            col_offset=tile.col_off)) if fixed_crops else None
        self.crop_size = crop_size
        self.img_aug = aug.get_transforms(img_aug)

        self.merge_labels = merge_labels
        if merge_labels is None:
            self.labels, self.label_names = map(list, zip(*self.DATASET_DESC['labels']))
            self.label_merger = None
        else:
            labels, self.label_names = merge_labels
            self.labels = list(range(len(labels)))
            self.label_merger = MergeLabels(labels)

        self.one_hot_labels = one_hot_labels
        if self.one_hot_labels:
            self.one_hot = OneHot(self.labels)

    def __len__(self):

        return len(self.crop_windows) if self.crop_windows else 1

    def __getitem__(self, idx):
        
        if self.crop_windows:
            window = self.crop_windows[idx]
        else:
            cx = self.tile.col_off + np.random.randint(0, self.tile.width - self.crop_size + 1)
            cy = self.tile.row_off + np.random.randint(0, self.tile.height - self.crop_size + 1)
            window = Window(cx, cy, self.crop_size, self.crop_size)
       
        image = self.read_window_fn(window=window, path=self.image_path)
        image = self.norm_fn(image[:3, ...])
        image = torch.from_numpy(image).float().contiguous()
       
        label = None
        if self.label_path:
            label = read_window_basic(window=window, path=self.label_path)
            if self.label_merger:
                label = self.label_merger(label)
            if self.one_hot:
                label = self.one_hot(label)
            label = torch.from_numpy(label).float().contiguous()

        if self.img_aug is not None:
            end_image, end_mask = self.img_aug(img=image, label=label)
        else:
            end_image, end_mask = image, label

        return {'orig_image':image,
                'orig_mask':label,
                'image':end_image,
                'window':window,
                'mask':end_mask}

def main():

    dataset = DigitanieDs(
        image_path='/d/pfournie/ai4geo/data/DIGITANIE/Toulouse/toulouse_tuile_7_img_normalized.tif',
        label_path='/d/pfournie/ai4geo/data/DIGITANIE/Toulouse/toulouse_tuile_7.tif',
        crop_size=256,
        crop_step=256,
        img_aug='no',
        tile=Window(col_off=500, row_off=502, width=400, height=400),
        fixed_crops=False
    )

    for data in dataset:
        pass
    img = plt.imshow(dataset[0]['image'].numpy().transpose(1,2,0))

    plt.show()


if __name__ == '__main__':
    main()
