import os
import warnings

import numpy as np
import rasterio as rio
from abc import ABC
import PIL
from PIL import Image
from torch_datasets import BaseDataset, BaseDatasetLabeled, BaseDatasetUnlabeled

warnings.filterwarnings(
    "ignore", category=rio.errors.NotGeoreferencedWarning
)


class MiniworldCities(BaseDataset, ABC):

    city_info_list = [
        # ('Arlington', 1, 2, (3000,3000)),
        # ('Atlanta', 1, 2, (1800,1200)),
        # ('austin', 15, 20, (3000,3000)),
        # ('Austin', 1, 2, (3063,3501)),
        # ('bruges', 2, 4, (1000,1000)),
        # ('chicago', 15, 20, (3000,3000)),
        ('christchurch', 73, 730, (1500,1500)),
        # ('DC', 1, 1, (1600,1600)),
        # ('khartoum', 345,667, (390,390)),
        # ('kitsap', 15, 20, (3000,3000)),
        # ('NewHaven', 1, 1, (3000,3000)),
        # ('NewYork', 1, 2, (1500,1500)),
        # ('Norfolk', 1, 1, (3000,3000)),
        # ('paris', 391,757,(390,390)),
        # ('potsdam', 10,14,(600,600)),
        # ('rio', 2360, 4580, (438,406)),
        # ('SanFrancisco', 1, 2, (3000,3000)),
        # ('Seekonk', 1,2, (3000,3000)),
        # ('shanghai', 1558,3024,(390,390)),
        # ('toulouse', 2,2,(3504,3452)),
        # ('tyrol-w', 15,20,(3000,3000)),
        # ('vegas', 1310,2541,(390,390)),
        # ('vienna', 15,20,(3000,3000))
    ]
    # pixels_per_class = [3985625189, 560767075]
    pixels_per_class = [1481907417, 148316583]
    mean_labeled_pixels = (0.4050, 0.4140, 0.3783)
    std_labeled_pixels = (0.2102, 0.2041, 0.1965)

    @staticmethod
    def colors_to_labels(labels_color):

        labels = np.zeros(labels_color.shape[:2], dtype=int)
        labels[np.where(np.any(labels_color != [0, 0, 0], axis=2))] = 1

        return labels

    def __init__(self,
                 labeled_image_paths,
                 label_paths,
                 unlabeled_image_paths,
                 image_size,
                 *args, **kwargs):

        self.labeled_image_paths = labeled_image_paths
        self.label_paths=label_paths
        self.unlabeled_image_paths = unlabeled_image_paths
        self.image_size = image_size
        super().__init__(
            *args,
            **kwargs
        )


class MiniworldCitiesLabeled(MiniworldCities, BaseDatasetLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class MiniworldCitiesUnlabeled(MiniworldCities, BaseDatasetUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)