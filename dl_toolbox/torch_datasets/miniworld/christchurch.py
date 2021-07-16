from abc import ABC
from dl_toolbox.torch_datasets import BaseCity, BaseLabeled, BaseUnlabeled


class Christchurch(BaseCity, ABC):

    def __init__(self, *args, **kwargs):

        self.image_size = (1500, 1500)
        super().__init__(city='christchurch', *args, **kwargs)


class ChristchurchLabeled(Christchurch, BaseLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class ChristchurchUnlabeled(Christchurch, BaseUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
