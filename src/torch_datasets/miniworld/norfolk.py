from abc import ABC
from torch_datasets import Base, BaseLabeled, BaseUnlabeled
import glob
import numpy as np


class Norfolk(Base, ABC):

    nb_labeled_images = 2
    nb_unlabeled_images = 0
    image_size = (3000,3000)
    # pixels_per_class = [3704503, 295497]
    # mean_labeled_pixels = (0.4050, 0.4140, 0.3783)
    # std_labeled_pixels = (0.2102, 0.2041, 0.1965)
    default_train_val = (1, 1)

    @staticmethod
    def colors_to_labels(labels_color):

        labels = np.zeros(labels_color.shape[:2], dtype=int)
        labels[np.where(np.any(labels_color != [0, 0, 0], axis=2))] = 1

        return labels

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.labeled_image_paths = sorted(
            glob.glob(f'{self.data_path}/Norfolk/train/*_x.png')
        ) + sorted(
            glob.glob(f'{self.data_path}/Norfolk/test/*_x.png')
        )

        self.unlabeled_image_paths = []

        self.label_paths = sorted(
            glob.glob(f'{self.data_path}/Norfolk/train/*_y.png')
        ) + sorted(
            glob.glob(f'{self.data_path}/Norfolk/test/*_y.png')
        )


class NorfolkLabeled(Norfolk, BaseLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class NorfolkUnlabeled(Norfolk, BaseUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
