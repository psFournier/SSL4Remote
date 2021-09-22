import numpy as np
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset
from abc import ABC
from dl_toolbox.utils import get_tiles
import imagesize
import torch

class OneImage(Dataset, ABC):

    def __init__(self,
                 image_path,
                 label_path=None,
                 idxs=None,
                 tile_size=None,
                 tile_step=None,
                 crop_size=None,
                 transforms=None,
                 *args,
                 **kwargs
                 ):

        super().__init__()

        self.image_path = image_path
        self.label_path = label_path
        width, height = imagesize.get(self.image_path)
        self.tile_size = tile_size
        self.tile_step = tile_size if tile_step is None else tile_step
        self.tile_windows = [
            w for w in get_tiles(
                nols=width,
                nrows=height,
                width=self.tile_size[1],
                height=self.tile_size[0],
                col_step=self.tile_step[1],
                row_step=self.tile_step[0]
            )
        ]
        self.idxs = list(range(len(self.tile_windows))) if idxs is None else idxs
        self.crop_size = crop_size
        self.transforms = transforms

    def get_window(self, idx):

        tile_window = self.tile_windows[idx]
        col_offset = tile_window.col_off
        row_offset = tile_window.row_off
        cx = np.random.randint(col_offset, col_offset + self.tile_size[1] - self.crop_size + 1)
        cy = np.random.randint(row_offset, row_offset + self.tile_size[0] - self.crop_size + 1)
        window = Window(cx, cy, self.crop_size, self.crop_size)

        return window

    def process_image(self, image):

        return torch.from_numpy(image).contiguous() / 255

    def process_label(self, label):

        label = torch.from_numpy(label).contiguous()

        return label, None

    def __len__(self):

        return len(self.idxs)

    def __getitem__(self, idx):

        tile_idx = self.idxs[idx]
        window = self.get_window(tile_idx)

        with rasterio.open(self.image_path) as image_file:
            image = image_file.read(window=window, out_dtype=np.float32)
            image = self.process_image(image)

        label = None
        if self.label_path:
            with rasterio.open(self.label_path) as label_file:
                label = label_file.read(window=window, out_dtype=np.float32)
                label = self.process_label(label)

        if self.transforms is not None:
            # image needs to be either [0, 255] ints or [0,1] floats
            end_image, end_mask = self.transforms(img=image, label=label)
        else:
            end_image, end_mask = image, label

        # end_image needs to be float for the nn
        return {'orig_image': image, 'orig_mask': label, 'image': end_image, 'window': window, 'mask': end_mask}

#
# class OneLabeledImage(OneImage):
#
#     def __init__(self, label_path, formatter, *args, **kwargs):
#
#         super(OneLabeledImage, self).__init__(*args, **kwargs)
#         self.label_path = label_path
#         self.label_formatter = formatter
#
#     def __getitem__(self, idx):
#
#         tile_idx = self.idxs[idx]
#         window = self.get_window(tile_idx)
#
#         with rasterio.open(self.image_path) as image_file:
#             image = image_file.read(window=window, out_dtype=np.float32)
#             image = torch.from_numpy(image).contiguous() / 255
#
#         with rasterio.open(self.label_path) as label_file:
#             label = label_file.read(window=window, out_dtype=np.float32)
#             mask = self.label_formatter(label)
#             mask = torch.from_numpy(mask).contiguous()
#
#         if self.transforms is not None:
#             end_image, end_mask = self.transforms(img=image, label=mask)
#         else:
#             end_image, end_mask = image, mask
#
#         return {'orig_image': image, 'orig_mask': mask, 'image': end_image, 'window': window, 'mask': end_mask}

