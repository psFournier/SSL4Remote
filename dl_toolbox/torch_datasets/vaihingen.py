import os
import warnings
from torch_datasets import Base, BaseLabeled, BaseUnlabeled
import numpy as np
import rasterio
from abc import ABC

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class IsprsV(Base, ABC):

    nb_labeled_images = 16
    nb_unlabeled_images = 17
    image_size = (1900,2600)
    pixels_per_class = [21815349, 20417332]
    mean_labeled_pixels = (0.4727, 0.3205, 0.3159)
    std_labeled_pixels = (0.2100, 0.1496, 0.1426)

    def __init__(self, *args, **kwargs):

        super().__init__(
            *args,
            **kwargs
        )
        self.labeled_image_paths = [
            os.path.join(
                self.data_path,
                'top/top_mosaic_09cm_area{}.tif'.format(i)
            ) for i in [
                1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32,
                34, 37
            ]
        ]
        self.unlabeled_image_paths = [
            os.path.join(
                self.data_path,
                'top/top_mosaic_09cm_area{}.tif'.format(i)
            ) for i in [
                2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29,
                31, 33, 35, 38
            ]
        ]
        self.label_paths = [
            os.path.join(
                self.data_path,
                'gts_for_participants/top_mosaic_09cm_area{}.tif'.format(i)
            ) for i in [
                1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32,
                34, 37
            ]
        ]

    @staticmethod
    def colors_to_labels(labels_color):

        labels = np.zeros(labels_color.shape[:2], dtype=np.uint8)
        colors = [
            [255, 255, 255],
            [0, 0, 255],
            [0, 255, 255],
            [255, 255, 0],
            [0, 255, 0],
            [255, 0, 0],
        ]

        for id_col, col in enumerate(colors):
            d = labels_color[:, :, 0] == col[0]
            d = np.logical_and(d, (labels_color[:, :, 1] == col[1]))
            d = np.logical_and(d, (labels_color[:, :, 2] == col[2]))
            labels[d] = id_col

        return labels


class IsprsVUnlabeled(IsprsV, BaseUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class IsprsVLabeled(IsprsV, BaseLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
