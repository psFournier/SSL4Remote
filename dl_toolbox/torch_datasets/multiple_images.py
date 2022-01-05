import warnings
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset
from abc import ABC
from utils import get_tiles
import imagesize
import matplotlib.pyplot as plt
import augmentations as aug

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

class MultipleImages(Dataset):

    def __init__(self,
                 images_paths,
                 labels_paths=None,
                 crop_size=None,
                 img_aug=None,
                 *args,
                 **kwargs
                 ):

        super().__init__()

        self.images_paths = images_paths
        self.labels_paths = labels_paths
        self.crop_size = crop_size
        self.img_aug = aug.get_transforms(img_aug)

    def get_window(self, image_path):

        width, height = imagesize.get(image_path)
        cx = np.random.randint(0, width - self.crop_size + 1)
        cy = np.random.randint(0, height - self.crop_size + 1)
        window = Window(cx, cy, self.crop_size, self.crop_size)

        return window

    def process_image(self, image):

        return torch.from_numpy(image).contiguous() / 255

    def process_label(self, label):

        return torch.from_numpy(label).contiguous()

    def __len__(self):

        return len(self.images_paths)

    def __getitem__(self, idx):

        image_path = self.images_paths[idx]
        window = self.get_window(image_path)

        with rasterio.open(image_path) as image_file:
            image = image_file.read(window=window, out_dtype=np.float32)
            image = self.process_image(image)

        label = None
        if self.labels_paths:
            label_path = self.labels_paths[idx]
            with rasterio.open(label_path) as label_file:
                label = label_file.read(window=window, out_dtype=np.float32)
                label = self.process_label(label)

        if self.img_aug is not None:
            # image needs to be either [0, 255] ints or [0,1] floats
            end_image, end_mask = self.img_aug(img=image, label=label)
        else:
            end_image, end_mask = image, label

        # end_image needs to be float for the nn
        return {'orig_image': image, 'orig_mask': label, 'image': end_image, 'window': window, 'mask': end_mask}


# class MultipleImagesLabeled(MultipleImages):
#
#     def __init__(self, labels_paths, formatter, *args, **kwargs):
#
#         super(MultipleImagesLabeled, self).__init__(*args, **kwargs)
#         assert len(labels_paths) == len(self.images_paths)
#         self.labels_paths = labels_paths
#         self.label_formatter = formatter
#
#     def __getitem__(self, idx):
#
#         image_path = self.images_paths[idx]
#         label_path = self.labels_paths[idx]
#         window = self.get_window(image_path)
#
#         with rasterio.open(image_path) as image_file:
#             image = image_file.read(window=window, out_dtype=np.float32)
#             image = torch.from_numpy(image).contiguous() / 255
#
#         with rasterio.open(label_path) as label_file:
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
