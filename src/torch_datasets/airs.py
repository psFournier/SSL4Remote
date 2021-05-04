from abc import ABC
from torch_datasets import Base, BaseLabeled, BaseUnlabeled
import glob
import numpy as np


class Airs(Base, ABC):

    # default_train_val = (857, 94)
    default_train_val = (8,1)
    nb_labeled_images = default_train_val[0] + default_train_val[1]
    image_size = (10000, 10000)

    def __init__(self, *args, **kwargs):

        super().__init__(
            *args,
            **kwargs
        )
        self.labeled_image_paths = sorted(
            glob.glob(f'{self.data_path}/trainval/train/image/*.tif')
        ) + sorted(
            glob.glob(f'{self.data_path}/trainval/val/image/*.tif')
        )

        self.unlabeled_image_paths = []

        self.label_paths = sorted(
            glob.glob(f'{self.data_path}/trainval/train/label/*vis.tif')
        ) + sorted(
            glob.glob(f'{self.data_path}/trainval/val/label/*vis.tif')
        )

    @staticmethod
    def colors_to_labels(labels_color):

        labels = np.zeros(labels_color.shape[:2], dtype=int)
        labels[np.where(np.any(labels_color != [0], axis=2))] = 1

        return labels


class AirsUnlabeled(Airs, BaseUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class AirsLabeled(Airs, BaseLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)