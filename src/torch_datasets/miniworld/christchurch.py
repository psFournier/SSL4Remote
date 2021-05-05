from abc import ABC
from torch_datasets import BaseCity, BaseLabeled, BaseUnlabeled
import glob
import numpy as np


class Christchurch(BaseCity, ABC):

    image_size = (1500, 1500)

    @staticmethod
    def colors_to_labels(labels_color):

        labels = np.zeros(labels_color.shape[:2], dtype=int)
        labels[np.where(np.any(labels_color != [0], axis=2))] = 1

        return labels

    def __init__(self, *args, **kwargs):

        super().__init__(city='christchurch', *args, **kwargs)


class ChristchurchLabeled(Christchurch, BaseLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class ChristchurchUnlabeled(Christchurch, BaseUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
