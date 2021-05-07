from abc import ABC
from torch_datasets import BaseCity, BaseLabeled, BaseUnlabeled
import glob
import numpy as np


class Chicago(BaseCity, ABC):

    image_size = (3000,3000)

    def __init__(self, *args, **kwargs):

        super().__init__(city='chicago', *args, **kwargs)


class ChicagoLabeled(Chicago, BaseLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class ChicagoUnlabeled(Chicago, BaseUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
