import os
import warnings

import numpy as np
import rasterio as rio
from rasterio.windows import Window
from torch.utils.data import Dataset

warnings.filterwarnings(
    "ignore", category=rio.errors.NotGeoreferencedWarning
)


class MiniworldCity(Dataset):

    labeled_image_paths = []
    label_paths = []
    unlabeled_image_paths = []
    mean_labeled_pixels = []
    std_labeled_pixels = []

    @classmethod
    def colors_to_labels(cls, labels_color):

        labels = np.uint8(labels_color != 0)

        return labels

    def __init__(self, data_path, idxs, crop):

        super().__init__()
        self.data_path = data_path
        self.idxs = idxs
        self.crop = crop
        self.approx_crop_per_image = int(
            self.__image_size__ / (crop**2)
        )

    def get_crop_window(self, image_file):

        cols = image_file.width
        rows = image_file.height
        cx = np.random.randint(0, cols - self.crop - 1)
        cy = np.random.randint(0, rows - self.crop - 1)
        w = Window(cx, cy, self.crop, self.crop)

        return w

    def get_image(self, idx):

        image_filepath = os.path.join(
            self.data_path, self.labeled_image_paths[idx]
        )

        with rio.open(image_filepath) as image_file:

            window = self.get_crop_window(image_file)
            tile = image_file.read(
                window=window, out_dtype=np.float32
            )
            tile = tile.transpose(1, 2, 0) / 255

        return tile, window

    def get_label(self, idx, window):

        label_filepath = os.path.join(
            self.data_path,
            self.label_paths[idx]
        )

        with rio.open(label_filepath) as label_file:

            tile_label = label_file.read(
                window=window, out_dtype=np.uint8
            )
            tile_label = tile_label.transpose(1, 2, 0)

        return tile_label

    @property
    def __image_size__(cls):

        raise NotImplementedError

    def __len__(self):

        return self.approx_crop_per_image * len(self.idxs)

    def __getitem__(self, idx):

        raise NotImplementedError

class MiniworldCityUnlabeled(MiniworldCity):

    def __init__(self, data_path, idxs, crop):

        super().__init__(data_path, idxs, crop)

    def __getitem__(self, idx):

        # idx is taken between 0 and self.__len__, which can be greather than len(idxs) due to tiling.
        idx = self.idxs[idx % len(self.idxs)]
        image, window = self.get_image(idx)

        return image

class MiniworldCityLabeled(MiniworldCity):

    def __init__(self, data_path, idxs, crop):

        super().__init__(data_path, idxs, crop)

    def __getitem__(self, idx):

        # idx is taken between 0 and self.__len__, which can be greather than len(idxs) due to tiling.
        idx = self.idxs[idx % len(self.idxs)]
        image, window = self.get_image(idx)
        label_colors = self.get_label(idx, window)
        label = self.colors_to_labels(label_colors)

        return image, label
