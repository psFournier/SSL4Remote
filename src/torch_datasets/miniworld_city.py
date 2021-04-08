import os
import warnings

import numpy as np
import rasterio as rio
from abc import ABC

from torch_datasets import BaseDataset

warnings.filterwarnings(
    "ignore", category=rio.errors.NotGeoreferencedWarning
)


class MiniworldCity(BaseDataset, ABC):

    @classmethod
    def colors_to_labels(cls, labels_color):

        labels = np.uint8(labels_color != 0)

        return labels