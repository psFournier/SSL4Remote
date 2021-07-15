from abc import ABC
from dl_toolbox.torch_datasets import BaseCity, BaseLabeled, BaseUnlabeled


class Tyrolw(BaseCity, ABC):

    def __init__(self, *args, **kwargs):

        self.image_size = (3000, 3000)
        super().__init__(city='tyrol-w', *args, **kwargs)



class TyrolwLabeled(Tyrolw, BaseLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class TyrolwUnlabeled(Tyrolw, BaseUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
