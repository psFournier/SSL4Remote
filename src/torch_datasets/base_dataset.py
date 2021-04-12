import os
import warnings

import numpy as np
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset
from abc import ABC

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class BaseDataset(Dataset, ABC):

    labeled_image_paths = []
    unlabeled_image_paths = []
    label_paths = []
    image_size = ()

    def __init__(self,
                 data_path,
                 idxs,
                 crop,
                 *args,
                 **kwargs
                 ):

        super().__init__()
        self.data_path = data_path
        self.idxs = idxs
        self.crop = crop
        self.mean_labeled_pixels = []
        self.std_labeled_pixels = []
        # The length of the dataset should be the number of get_item calls needed to
        # span the whole dataset. If get_item gives the full image, this is obviously
        # the total number of images in the dataset.
        # On the contrary, here get_item only gives a cropped tile from the image. Given the
        # crop parameter of the class and the average image size, provided they are all
        # close, we can say approx how many get_item calls are needed.
        self.approx_crop_per_image = int(
            self.image_size[0] * self.image_size[1] / (crop**2)
        )

    def get_crop_window(self, image_file):

        cols = image_file.width
        rows = image_file.height
        cx = np.random.randint(0, cols - self.crop - 1)
        cy = np.random.randint(0, rows - self.crop - 1)
        w = Window(cx, cy, self.crop, self.crop)

        return w

    def get_image(self, idx):

        path = (self.labeled_image_paths + self.unlabeled_image_paths)[idx]
        image_filepath = os.path.join(self.data_path, path)

        with rasterio.open(image_filepath) as image_file:

            window = self.get_crop_window(image_file)
            image = image_file.read(window=window, out_dtype=np.float32)
            image = image.transpose(1, 2, 0) / 255

        return image, window

    def get_label(self, idx, window):

        path = self.label_paths[idx]
        label_filepath = os.path.join(self.data_path, path)

        with rasterio.open(label_filepath) as label_file:

            label = label_file.read(window=window, out_dtype=np.uint8)
            label = label.transpose(1, 2, 0)

        return label

    def __len__(self):

        return self.approx_crop_per_image * len(self.idxs)

    def __getitem__(self, idx):

        raise NotImplementedError


class BaseDatasetUnlabeled(BaseDataset, ABC):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):

        idx = idx % len(self.idxs)
        idx = self.idxs[idx]
        image, window = self.get_image(idx)

        return image


class BaseDatasetLabeled(BaseDatasetUnlabeled, ABC):

    @classmethod
    def colors_to_labels(cls, labels_color):

        raise NotImplementedError

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):

        idx = idx % len(self.idxs)
        idx = self.idxs[idx]
        image, window = self.get_image(idx)
        label_colors = self.get_label(idx, window)
        label = self.colors_to_labels(label_colors)

        return image, label
