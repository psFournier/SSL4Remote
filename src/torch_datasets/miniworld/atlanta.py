from abc import ABC
from torch_datasets import Base, BaseLabeled, BaseUnlabeled
import glob
import numpy as np


class Atlanta(Base, ABC):

    nb_unlabeled_images = 0
    image_size = (1800,1200)
    # pixels_per_class = [151476453, 28523547]
    # mean_labeled_pixels = (0.4050, 0.4140, 0.3783)
    # std_labeled_pixels = (0.2102, 0.2041, 0.1965)
    default_train_val = (2, 1)
    nb_labeled_images = default_train_val[0] + default_train_val[1]

    @staticmethod
    def colors_to_labels(labels_color):

        labels = np.zeros(labels_color.shape[:2], dtype=int)
        labels[np.where(np.any(labels_color != [0, 0, 0], axis=2))] = 1

        return labels

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.labeled_image_paths = sorted(
            glob.glob(f'{self.data_path}/Atlanta/train/*_x.tif')
        ) + sorted(
            glob.glob(f'{self.data_path}/Atlanta/test/*_x.tif')
        )

        self.unlabeled_image_paths = []

        self.label_paths = sorted(
            glob.glob(f'{self.data_path}/Atlanta/train/*_y.tif')
        ) + sorted(
            glob.glob(f'{self.data_path}/Atlanta/test/*_y.tif')
        )


class AtlantaLabeled(Atlanta, BaseLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class AtlantaUnlabeled(Atlanta, BaseUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
