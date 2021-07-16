from abc import ABC
from dl_toolbox.torch_datasets import BaseCity, BaseLabeled, BaseUnlabeled


class Austin(BaseCity, ABC):

    def __init__(self, *args, **kwargs):

        super().__init__(city='austin', *args, **kwargs)



class AustinLabeled(Austin, BaseLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class AustinUnlabeled(Austin, BaseUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
