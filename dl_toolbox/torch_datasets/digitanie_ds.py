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

class DigitanieDs(Dataset):

    labels_desc = [
            (0, (255,255,255), 'void'),
            (1, (34, 139, 34), 'bareland'),
            (2, (35, 140, 35), 'grass'),
            (3, (0, 0, 238), 'lake'),
            (4, (238, 118, 33), 'building'),
            (5, (0, 222, 137), 'high vegetation'),
            (6, (38, 38, 38), 'parking'),
            (7, (40, 40, 40), 'roadway'),
            (8, (42, 42, 42), 'traffic lanes'),
            (9, (160, 30, 230), 'sport venue'),
            (10, (0, 0, 250), 'swimming pool'),
            (11, (44, 44, 44), 'railway')]

    def __init__(
            self,
            image_path,
            label_path,
            fixed_crops,
            crop_size,
            img_aug,
            *args,
            **kwargs
            ):

        self.image_path = image_path
        self.label_path = label_path
        self.tile_size = imagesize.get(image_path)
        self.crop_windows = None if not fixed_crops else list(get_tiles(*self.tile_size, crop_size))
        self.crop_size = crop_size
        self.img_aug = aug.get_transforms(img_aug)

    def __len__(self):

        return len(self.crop_windows) if self.crop_windows else 1

    def __getitem__(self, idx):
        
        if self.crop_windows:
            window = self.crop_windows[idx]
        else:
            cx = np.random.randint(0, self.tile_size[0] - self.crop_size + 1)
            cy = np.random.randint(0, self.tile_size[1] - self.crop_size + 1)
            window = Window(cx, cy, self.crop_size, self.crop_size)
        
#        with rasterio.open(self.label_path) as label_file:
#            label = label_file.read(window=label_window, out_dtype=np.float32)
#            window_bounds = bounds(label_window, label_file.transform)
#        
#        with rasterio.open(self.image_path) as image_file:
#            image_window = from_bounds(*window_bounds, transform=image_file.transform)
#            image = image_file.read(window=image_window, out_dtype=np.float32)

        with rasterio.open(self.image_path) as image_file:
            image = image_file.read(window=window, out_dtype=np.float32)

        with rasterio.open(self.label_path) as label_file:
            label = label_file.read(window=window, out_dtype=np.float32)
        
#        m = np.array([0.0357, 0.0551, 0.0674]
#        M = np.array([0.2945, 0.2734, 0.2662])
#        image = torch.from_numpy(minmax(image[:3,...], m, M)).float().contiguous()
        image = torch.from_numpy(image[:3,...]).float().contiguous()
        
        onehot_masks = [(label==val).astype(float).squeeze() for val, _, _ in self.labels_desc]
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
