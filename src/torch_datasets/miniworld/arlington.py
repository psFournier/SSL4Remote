from torch_datasets import MiniworldCity
from torch_datasets import BaseLabeled, BaseUnlabeled


class Arlington(Base, ABC):

    labeled_image_paths = ['test/{}_x.png'.format(i) for i in [0]] + \
                          ['train/{}_x.png'.format(i) for i in [0,1]]

    label_paths = ['test/{}_y.png'.format(i) for i in [0]] + \
                  ['train/{}_y.png'.format(i) for i in [0,1]]

    unlabeled_image_paths = []

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    @property
    def __image_size__(cls):

        return 3000*3000

class ArlingtonLabeled(Arlington, BaseLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

class ArlingtonUnlabeled(Arlington, BaseUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
