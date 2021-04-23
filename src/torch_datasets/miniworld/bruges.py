from torch_datasets import MiniworldCity
from torch_datasets import BaseLabeled, BaseUnlabeled


class Bruges(MiniworldCity):

    labeled_image_paths = ['test/{}_x.png'.format(i) for i in range(2)] + \
                          ['train/{}_x.png'.format(i) for i in range(4)]

    label_paths = ['test/{}_y.png'.format(i) for i in range(2)] + \
                  ['train/{}_y.png'.format(i) for i in range(4)]

    unlabeled_image_paths = []

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    @property
    def __image_size__(cls):

        return 1000*1000

class BrugesLabeled(Bruges, BaseLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

class BrugesUnlabeled(Bruges, BaseUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
