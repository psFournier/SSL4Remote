from abc import ABC
from dl_toolbox.torch_datasets import BaseCity, BaseLabeled, BaseUnlabeled


class Kitsap(BaseCity, ABC):

    def __init__(self, *args, **kwargs):

        self.image_size = (3000,3000)
        super().__init__(city='kitsap', *args, **kwargs)


class KitsapLabeled(Kitsap, BaseLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class KitsapUnlabeled(Kitsap, BaseUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
