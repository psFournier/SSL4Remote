from torch_datasets import MiniworldCity
from torch_datasets import BaseDatasetLabeled, BaseDatasetUnlabeled


class SanFrancisco(MiniworldCity):

    labeled_image_paths = ['test/{}_x.png'.format(i) for i in range(1)] + \
                          ['train/{}_x.png'.format(i) for i in range(2)]

    label_paths = ['test/{}_y.png'.format(i) for i in range(1)] + \
                  ['train/{}_y.png'.format(i) for i in range(2)]

    unlabeled_image_paths = []

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    @property
    def __image_size__(cls):

        return 3000*3000

class SanFranciscoLabeled(SanFrancisco, BaseDatasetLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

class SanFranciscoUnlabeled(SanFrancisco, BaseDatasetUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
