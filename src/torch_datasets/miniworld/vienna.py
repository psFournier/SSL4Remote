from abc import ABC
from torch_datasets import BaseCity, BaseLabeled, BaseUnlabeled


class Vienna(BaseCity, ABC):

    image_size = (3000, 3000)

    def __init__(self, *args, **kwargs):

        super().__init__(city='vienna', *args, **kwargs)


class ViennaLabeled(Vienna, BaseLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class ViennaUnlabeled(Vienna, BaseUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
