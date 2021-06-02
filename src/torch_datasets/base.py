import warnings
import numpy as np
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset
from abc import ABC
from utils import get_tiles

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class Base(Dataset, ABC):

    """
    Abstract class that inherits from the standard Torch Dataset abstract class
    and define utilities for remote sensing dataset classes.
    """

    image_size = (0, 0)

    def __init__(self,
                 data_path,
                 crop,
                 idxs=None,
                 fixed_crop=False,
                 *args,
                 **kwargs
                 ):

        super().__init__()

        self.data_path = data_path
        self.path_idxs = idxs
        self.crop = crop
        self.mean_labeled_pixels = []
        self.std_labeled_pixels = []

        self.labeled_image_paths = None
        self.label_paths = None
        self.unlabeled_image_paths = None

        self.fixed_crop = fixed_crop
        # Assumes all image are the same size.
        self.precomputed_crops = [
            window for window in get_tiles(
                nols=self.image_size[1],
                nrows=self.image_size[0],
                width=self.crop,
                height=self.crop,
                col_step=self.crop,
                row_step=self.crop
            )
        ]

    def get_random_crop(self):

        cols, rows = self.image_size
        cx = np.random.randint(0, cols - self.crop - 1)
        cy = np.random.randint(0, rows - self.crop - 1)
        w = Window(cx, cy, self.crop, self.crop)

        return w

    def get_image(self, path_idx, crop_idx):

        image_filepath = (self.labeled_image_paths + self.unlabeled_image_paths)[path_idx]

        with rasterio.open(image_filepath) as image_file:

            if self.fixed_crop:
                window = self.precomputed_crops[crop_idx]
            else:
                window = self.get_random_crop()

            image = image_file.read(window=window, out_dtype=np.float32) / 255

        return image, window

    def get_label(self, path_idx, window):

        label_filepath = self.label_paths[path_idx]

        with rasterio.open(label_filepath) as label_file:

            label = label_file.read(window=window, out_dtype=np.float32)

        return label

    def __len__(self):

        return len(self.precomputed_crops) * len(self.path_idxs)

    def __getitem__(self, idx):

        raise NotImplementedError


class BaseUnlabeled(Base):

    """Generic class for datasets without labels."""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):

        path_idx = self.path_idxs[idx % len(self.path_idxs)]
        crop_idx = idx // len(self.path_idxs)
        image, window = self.get_image(path_idx, crop_idx)

        return image


class BaseLabeled(Base):

    """Generic class for datasets wtih labels. Child classes must define
    the colors_to_labels static method. """

    @staticmethod
    def colors_to_labels(labels_color):

        raise NotImplementedError

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):

        path_idx = self.path_idxs[idx % len(self.path_idxs)]
        crop_idx = idx // len(self.path_idxs)
        image, window = self.get_image(path_idx, crop_idx)
        label = self.get_label(path_idx, window)
        mask = self.colors_to_labels(label)

        return image, mask
