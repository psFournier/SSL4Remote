import os
import warnings

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from torch.utils.data import Dataset
from abc import ABC
from utils import get_tiles

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class Base(Dataset, ABC):

    nb_labeled_images = 0
    nb_unlabeled_images = 0
    labeled_image_paths = []
    unlabeled_image_paths = []
    label_paths = []
    image_size = ()

    def __init__(self,
                 data_path,
                 idxs,
                 crop,
                 augmentations,
                 fixed_crop=False,
                 *args,
                 **kwargs
                 ):

        super().__init__()

        self.data_path = data_path
        self.idxs = idxs
        self.crop = crop
        self.augmentations = augmentations
        self.mean_labeled_pixels = []
        self.std_labeled_pixels = []
        self.fixed_crop = fixed_crop

        if self.fixed_crop:
            # Assumes all image are the same size.
            self.fixed_crops = [
                window for window in get_tiles(
                    image_size=self.image_size,
                    width=self.crop,
                    height=self.crop,
                    col_step=self.crop,
                    row_step=self.crop
                )
            ]

    @staticmethod
    def get_crop_window(crop, image_file):

        cols = image_file.width
        rows = image_file.height
        cx = np.random.randint(0, cols - crop - 1)
        cy = np.random.randint(0, rows - crop - 1)
        w = Window(cx, cy, crop, crop)

        return w

    def get_image(self, image_idx, crop_idx):

        image_filepath = (self.labeled_image_paths + self.unlabeled_image_paths)[self.idxs[image_idx]]

        with rasterio.open(image_filepath) as image_file:

            if self.fixed_crop:
                window = self.fixed_crops[crop_idx]
            else:
                window = self.get_crop_window(self.crop, image_file)

            image = image_file.read(window=window, out_dtype=np.uint8).transpose(1, 2, 0)

        return image, window

    def get_label(self, image_idx, window):

        label_filepath = self.label_paths[self.idxs[image_idx]]

        with rasterio.open(label_filepath) as label_file:

            label = label_file.read(window=window, out_dtype=np.uint8).transpose(1, 2, 0)

        return label

    def __len__(self):

        if self.fixed_crop:
            return len(self.fixed_crops) * len(self.idxs)
        else:
            return len(self.idxs)

    def __getitem__(self, idx):

        raise NotImplementedError


class BaseUnlabeled(Base, ABC):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):

        image_idx = idx % len(self.idxs)
        crop_idx = idx // len(self.idxs)
        image, window = self.get_image(image_idx, crop_idx)
        augment = self.augmentations(image=image)

        return augment['image']


class BaseLabeled(BaseUnlabeled, ABC):

    @staticmethod
    def colors_to_labels(labels_color):

        raise NotImplementedError

    def __init__(self, label_merger=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.label_merger = label_merger

    def __getitem__(self, idx):

        image_idx = idx % len(self.idxs)
        crop_idx = idx // len(self.idxs)
        image, window = self.get_image(image_idx, crop_idx)
        label = self.get_label(image_idx, window)
        mask = self.colors_to_labels(label)
        if self.label_merger is not None:
            mask = self.label_merger(mask)
        augment = self.augmentations(
            image=image,
            mask=mask
        )

        return augment['image'], augment['mask']
