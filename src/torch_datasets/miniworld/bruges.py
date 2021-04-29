from abc import ABC
from torch_datasets import Base, BaseLabeled, BaseUnlabeled
import glob
import numpy as np


class Bruges(Base, ABC):

    nb_labeled_images = 6
    # nb_labeled_images = 9
    nb_unlabeled_images = 0
    image_size = (1000,1000)
    # pixels_per_class = [171095182, 8904818]
    # mean_labeled_pixels = (0.4050, 0.4140, 0.3783)
    # std_labeled_pixels = (0.2102, 0.2041, 0.1965)
    default_train_val = (4, 2)

    @staticmethod
    def colors_to_labels(labels_color):

        labels = np.zeros(labels_color.shape[:2], dtype=int)
        labels[np.where(np.any(labels_color != [0, 0, 0], axis=2))] = 1

        return labels

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.labeled_image_paths = sorted(
            glob.glob(f'{self.data_path}/bruges/train/*_x.png')
        ) + sorted(
            glob.glob(f'{self.data_path}/bruges/test/*_x.png')
        )

        self.unlabeled_image_paths = []

        self.label_paths = sorted(
            glob.glob(f'{self.data_path}/bruges/train/*_y.png')
        ) + sorted(
            glob.glob(f'{self.data_path}/bruges/test/*_y.png')
        )


class BrugesLabeled(Bruges, BaseLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class BrugesUnlabeled(Bruges, BaseUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
