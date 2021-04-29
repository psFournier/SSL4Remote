from abc import ABC
from torch_datasets import Base, BaseLabeled, BaseUnlabeled
import glob
import numpy as np


class Paris(Base, ABC):

    # default_train_val = (757, 391)
    default_train_val = (8, 4)
    nb_labeled_images = default_train_val[0] + default_train_val[1]
    nb_unlabeled_images = 0
    image_size = (390, 390)
    pixels_per_class = [1, 1]
    # mean_labeled_pixels = (0.4050, 0.4140, 0.3783)
    # std_labeled_pixels = (0.2102, 0.2041, 0.1965)


    @staticmethod
    def colors_to_labels(labels_color):

        labels = np.zeros(labels_color.shape[:2], dtype=int)
        labels[np.where(np.any(labels_color != [0, 0, 0], axis=2))] = 1

        return labels

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.labeled_image_paths = sorted(
            glob.glob(f'{self.data_path}/paris/train/*_x.png')
        ) + sorted(
            glob.glob(f'{self.data_path}/paris/test/*_x.png')
        )

        self.unlabeled_image_paths = []

        self.label_paths = sorted(
            glob.glob(f'{self.data_path}/paris/train/*_y.png')
        ) + sorted(
            glob.glob(f'{self.data_path}/paris/test/*_y.png')
        )


class ParisLabeled(Paris, BaseLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class ParisUnlabeled(Paris, BaseUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
