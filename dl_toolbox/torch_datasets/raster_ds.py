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



class RasterDs(Dataset):

    def __init__(
            self,
            image_path,
            tile,
            fixed_crops,
            crop_size,
            crop_step,
            img_aug='no',
            label_path=None,
            *args,
            **kwargs
            ):

        self.image_path = image_path
        self.label_path = label_path
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

    def __len__(self):

        return len(self.crop_windows) if self.crop_windows else 1

    def __getitem__(self, idx):
        
        if self.crop_windows:
            window = self.crop_windows[idx]
        else:
            cx = self.tile.col_off + np.random.randint(0, self.tile.width - self.crop_size + 1)
            cy = self.tile.row_off + np.random.randint(0, self.tile.height - self.crop_size + 1)
            window = Window(cx, cy, self.crop_size, self.crop_size)
        
        with rasterio.open(self.image_path) as image_file:
            image = image_file.read(window=window, out_dtype=np.float32)
        image = torch.from_numpy(image).float().contiguous()
        
        label = None
        if self.label_path:
            with rasterio.open(self.label_path) as label_file:
                label = label_file.read(window=window, out_dtype=np.float32)
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

    dataset = RasterDs(
        image_path='/work/OT/ai4geo/DATA/DATASETS/DIGITANIE/Toulouse/toulouse_tuile_7_img_normalized.tif',
        label_path='/work/OT/ai4geo/DATA/DATASETS/DIGITANIE/Toulouse/toulouse_tuile_7.tif',
        crop_size=256,
        img_aug='no',
        tile=Window(col_off=500, row_off=502, width=400, height=400),
        fixed_crops=True
    )

    for data in dataset:
        print(data['window'])

if __name__ == '__main__':
    main()
