from abc import ABC
from torch_datasets import BaseCity, BaseLabeled, BaseUnlabeled
import glob
import numpy as np


class Tyrolw(BaseCity, ABC):

    image_size = (3000, 3000)

    def __init__(self, *args, **kwargs):

        super().__init__(city='tyrol-w', *args, **kwargs)

class TyrolwLabeled(Tyrolw, BaseLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class TyrolwUnlabeled(Tyrolw, BaseUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
