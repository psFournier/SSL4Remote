from abc import ABC
from dl_toolbox.torch_datasets import BaseCity, BaseLabeled, BaseUnlabeled


class Chicago(BaseCity, ABC):

    def __init__(self, *args, **kwargs):

        self.image_size = (3000, 3000)
        super().__init__(city='chicago', *args, **kwargs)

class ChicagoLabeled(Chicago, BaseLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class ChicagoUnlabeled(Chicago, BaseUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
