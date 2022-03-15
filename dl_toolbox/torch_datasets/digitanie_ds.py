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

DATASET_DESC = {
    'labels': [
        (0, 'void'),
        (1, 'bareland'),
        (2, 'low vegetation'),
        (3, 'water'),
        (4, 'building'),
        (5, 'high vegetation'),
        (6, 'parking'),
        (7, 'pedestrian'),
        (8, 'road'),
        (9, 'swimming pool'),
        (10, 'railway')
    ],
    'percentile2': [0.0357, 0.0551, 0.0674],
    'percentile98': [0.2945, 0.2734, 0.2662],
}

class DigitanieDs(Dataset):

    def __init__(
            self,
            image_path,
            fixed_crops,
            crop_size,
            img_aug,
            col_offset=0,
            row_offset=0,
            tile_size=None,
            label_path=None,
            merge_labels=None,
            one_hot_labels=True,
            *args,
            **kwargs
            ):

        self.image_path = image_path
        self.label_path = label_path
        self.tile_size = imagesize.get(image_path) if not tile_size else tile_size
        self.col_offset = col_offset
        self.row_offset = row_offset
        self.crop_size = crop_size
        self.crop_windows = None if not fixed_crops else list(get_tiles(
            nols = self.tile_size[0],
            nrows = self.tile_size[1],
            size = self.crop_size,
            col_offset = self.col_offset,
            row_offset = self.row_offset
        ))
        self.img_aug = aug.get_transforms(img_aug)

        self.merge_labels = merge_labels
        if merge_labels is None:
            self.labels, self.label_names = map(list, zip(*DATASET_DESC['labels']))
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
            cx = self.col_offset + np.random.randint(0, self.tile_size[0] - self.crop_size + 1)
            cy = self.row_offset + np.random.randint(0, self.tile_size[1] - self.crop_size + 1)
            window = Window(cx, cy, self.crop_size, self.crop_size)
        
        with rasterio.open(self.image_path) as image_file:
            image = image_file.read(window=window, out_dtype=np.float32)[:3, ...]
        image = torch.from_numpy(image).float().contiguous()
        
        label = None
        if self.label_path:

            with rasterio.open(self.label_path) as label_file:
                label = label_file.read(window=window, out_dtype=np.float32)
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
