from torch_datasets import MiniworldCity
from torch_datasets import BaseLabeled, BaseUnlabeled


class Potsdam(MiniworldCity):

    labeled_image_paths = ['test/{}_x.png'.format(i) for i in range(10)] + \
                          ['train/{}_x.png'.format(i) for i in range(14)]

    label_paths = ['test/{}_y.png'.format(i) for i in range(10)] + \
                  ['train/{}_y.png'.format(i) for i in range(14)]

    unlabeled_image_paths = []

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    @property
    def __image_size__(cls):

        return 600*600

class PotsdamLabeled(Potsdam, BaseLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

class PotsdamUnlabeled(Potsdam, BaseUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
