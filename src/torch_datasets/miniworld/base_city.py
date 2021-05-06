from abc import ABC
from torch_datasets import Base
import glob
import numpy as np


class BaseCity(Base, ABC):

    @staticmethod
    def colors_to_labels(labels_color):

        labels0 = np.zeros(shape=labels_color.shape[:2], dtype=float)
        labels1 = np.zeros(shape=labels_color.shape[:2], dtype=float)
        mask = np.any(labels_color != [0], axis=2)
        np.putmask(labels0, ~mask, 1.)
        np.putmask(labels1, mask, 1.)
        labels = np.stack([labels0, labels1], axis=2)

        return labels

    def __init__(self, city, *args, **kwargs):

        super().__init__(*args, **kwargs)

        train_labeled_image_paths = sorted(
            glob.glob(f'{self.data_path}/{city}/train/*_x.tif')
        )
        test_labeled_image_paths = sorted(
            glob.glob(f'{self.data_path}/{city}/test/*_x.tif')
        )
        self.labeled_image_paths = train_labeled_image_paths + test_labeled_image_paths

        train_label_paths = sorted(
            glob.glob(f'{self.data_path}/{city}/train/*_y.tif')
        )
        test_label_paths = sorted(
            glob.glob(f'{self.data_path}/{city}/test/*_y.tif')
        )
        self.label_paths = train_label_paths + test_label_paths

        self.unlabeled_image_paths = []

        self.default_train_val = (
            len(train_label_paths),
            len(test_label_paths)
        )
