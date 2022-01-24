import os
from torch.utils.data import Dataset
import torch
import augmentations as aug
from torch_datasets.commons import minmax
from utils import get_tiles
import rasterio
import imagesize
import numpy as np
from rasterio.windows import Window, bounds, from_bounds

class SemcityBdsdDs(Dataset):

    labels_desc = [
        (0, (255, 255, 255), 'void', 1335825),
        (1, (38, 38, 38), 'impervious surface', 13109372),
        (2, (238, 118, 33), 'building', 9101418),
        (3, (34, 139, 34), 'pervious surface', 12857668),
        (4, (0, 222, 137), 'high vegetation', 8214402),
        (5, (255, 0, 0), 'car', 1015653),
        (6, (0, 0, 238), 'water', 923176),
        (7, (160, 30, 230), 'sport venues', 1825718)
    ]
    # Min and max are 1 and 99 percentiles
    image_stats = {
        'num_channels': 8,
        'min' : np.array([245, 166, 167, 107, 42, 105, 60, 48]),
        'max' : np.array([615, 681, 1008, 1087, 732, 1065, 1126, 1046])
    }

    def __init__(
        self,
        image_path,
        label_path,
        tile,
        fixed_crops,
        crop_size,
        img_aug,
        *args,
        **kwargs):

        self.image_path = image_path
        self.label_path = label_path
        self.tile = tile
        self.crop_windows = list(get_tiles(
            tile.width, 
            tile.height, 
            crop_size, 
            row_offset=tile.row_off, 
            col_offset=tile.col_off)) if fixed_crops else None
        self.crop_size = crop_size
        self.img_aug = aug.get_transforms(img_aug)

    def __len__(self):

        return len(self.crop_windows) if self.crop_windows else 1

    def __getitem__(self, idx):
        
        if self.crop_windows:
            window = self.crop_windows[idx]
        else:
            cx = np.random.randint(self.tile.col_off, self.tile.col_off + self.tile.width - self.crop_size + 1)
            cy = np.random.randint(self.tile.row_off, self.tile.row_off + self.tile.height - self.crop_size + 1)
            window = Window(cx, cy, self.crop_size, self.crop_size)
        
        with rasterio.open(self.label_path) as label_file:
            label = label_file.read(window=window, out_dtype=np.float32)
        
        with rasterio.open(self.image_path) as image_file:
            image = image_file.read(window=window, out_dtype=np.float32)
        
        m, M = self.image_stats['min'][[3,2,1]], self.image_stats['max'][[3,2,1]]
        image = torch.from_numpy(minmax(image[[3,2,1],...], m, M)).float().contiguous()

        onehot_masks = []
        for _, color, _, _ in self.labels_desc:
            d = label[0, :, :] == color[0]
            d = np.logical_and(d, (label[1, :, :] == color[1]))
            d = np.logical_and(d, (label[2, :, :] == color[2]))
            onehot_masks.append(d.astype(float))
        onehot = np.stack(onehot_masks, axis=0)
        label = torch.from_numpy(np.stack(onehot_masks, axis=0)).float().contiguous()
        
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

    dataset = SemcityBdsdDs(
        image_path='/home/pfournie/ai4geo/data/SemcityTLS_DL/BDSD_M_3_4_7_8.tif',
        label_path='/home/pfournie/ai4geo/data/SemcityTLS_DL/GT_3_4_7_8.tif',
        crop_size=128,
        img_aug='no',
        tile=Window(col_off=876, row_off=863, width=876, height=863),
        fixed_crops=True
    )

    for data in dataset:

        print(data['window'])

if __name__ == '__main__':
    
    main()
