from torch_datasets import MiniworldCity
from torch_datasets import BaseLabeled, BaseUnlabeled


class Shanghai(MiniworldCity):

    labeled_image_paths = ['test/{}_x.png'.format(i) for i in range(1558)] + \
                          ['train/{}_x.png'.format(i) for i in range(3024)]

    label_paths = ['test/{}_y.png'.format(i) for i in range(1558)] + \
                  ['train/{}_y.png'.format(i) for i in range(3024)]

    unlabeled_image_paths = []

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    @property
    def __image_size__(cls):

        return 390*390

class ShanghaiLabeled(Shanghai, BaseLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

class ShanghaiUnlabeled(Shanghai, BaseUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
