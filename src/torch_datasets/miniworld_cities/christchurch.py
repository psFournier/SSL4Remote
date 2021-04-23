from torch_datasets import MiniworldCity
from torch_datasets import BaseLabeled, BaseUnlabeled


class Christchurch(MiniworldCity):

    labeled_image_paths = ['test/{}_x.png'.format(i) for i in range(73)] + \
                          ['train/{}_x.png'.format(i) for i in range(730)]

    label_paths = ['test/{}_y.png'.format(i) for i in range(73)] + \
                  ['train/{}_y.png'.format(i) for i in range(730)]

    unlabeled_image_paths = []

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    @property
    def __image_size__(cls):

        return 1500*1500

class ChristchurchLabeled(Christchurch, BaseLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

class ChristchurchUnlabeled(Christchurch, BaseUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
