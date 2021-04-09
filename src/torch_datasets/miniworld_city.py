import os
import warnings

import numpy as np
import rasterio as rio
from abc import ABC
import PIL
from PIL import Image
from torch_datasets import BaseDataset

warnings.filterwarnings(
    "ignore", category=rio.errors.NotGeoreferencedWarning
)


class MiniworldCity(BaseDataset, ABC):

    @classmethod
    def colors_to_labels(cls, labels_color):

        labels = np.zeros(labels_color.shape[:2], dtype=int)
        labels[np.where(np.all(labels_color != [0, 0, 0], axis=2))] = 1

        return labels