from abc import ABC
from torch_datasets import Base, BaseLabeled, BaseUnlabeled
import glob
import numpy as np
import torch
import rasterio

class BaseCity(Base, ABC):

    """
    Inherits from the Base dataset class and overloads common methods of miniworld cities classes to minimize code
    redundancy in the actual cities dataset classes.
    """

    @staticmethod
    def colors_to_labels(colors):

        '''
        Creates one-hot encoded labels for binary classification.
        :param labels_color:
        :return:
        '''

        labels0 = np.zeros(shape=colors.shape[1:], dtype=float)
        labels1 = np.zeros(shape=colors.shape[1:], dtype=float)
        mask = np.any(colors != [0], axis=0)
        np.putmask(labels0, ~mask, 1.)
        np.putmask(labels1, mask, 1.)
        labels = np.stack([labels0, labels1], axis=0)

        return labels

    @staticmethod
    def labels_to_colors(labels):

        colors = np.zeros(shape=(labels.shape[0], labels.shape[1], labels.shape[2], 3), dtype=np.uint8)
        idx = np.array(labels == 1)
        colors[idx] = np.array([255,255,255])
        res = np.transpose(colors, axes=(0, 3, 1, 2))
        return torch.from_numpy(res).float()

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

        self.precompute_crops()


class BaseCityImage(Base, ABC):

    def __init__(self, image_path, label_path, *args, **kwargs):

        super(BaseCityImage, self).__init__(*args, **kwargs)
        self.labeled_image_paths = [image_path]
        self.label_paths = [label_path]
        self.path_idxs = [0]
        self.precompute_crops()

    def colors_to_labels(self, colors):

        return BaseCity.colors_to_labels(colors)

    def labels_to_colors(self, labels):

        return BaseCity.labels_to_colors(labels)


class BaseCityImageLabeled(BaseCityImage, BaseLabeled):

    def __init__(self, image_path, label_path, *args, **kwargs):

        super().__init__(image_path, label_path, *args, **kwargs)


class BaseCityImageUnlabeled(BaseCityImage, BaseUnlabeled):

    def __init__(self, image_path, label_path, *args, **kwargs):

        super().__init__(image_path, label_path, *args, **kwargs)
