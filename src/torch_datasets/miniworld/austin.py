from abc import ABC
from torch_datasets import BaseCity, BaseLabeled, BaseUnlabeled
import glob
import numpy as np


class Austin(BaseCity, ABC):

    image_size = (3000, 3000)

    @staticmethod
    def colors_to_labels(labels_color):

        labels = np.zeros(labels_color.shape[:2], dtype=int)
        labels[np.where(np.any(labels_color != [0, 0, 0], axis=2))] = 1

        return labels

    def __init__(self, *args, **kwargs):

        super().__init__(city='austin', *args, **kwargs)


class AustinLabeled(Austin, BaseLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class AustinUnlabeled(Austin, BaseUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
