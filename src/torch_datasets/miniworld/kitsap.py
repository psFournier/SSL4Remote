from abc import ABC
from torch_datasets import BaseCity, BaseLabeled, BaseUnlabeled
import glob
import numpy as np


class Kitsap(BaseCity, ABC):

    image_size = (3000,3000)

    def __init__(self, *args, **kwargs):

        super().__init__(city='kitsap', *args, **kwargs)


class KitsapLabeled(Kitsap, BaseLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class KitsapUnlabeled(Kitsap, BaseUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
