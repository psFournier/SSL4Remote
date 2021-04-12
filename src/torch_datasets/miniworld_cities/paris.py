from torch_datasets import MiniworldCity
from torch_datasets import BaseDatasetLabeled, BaseDatasetUnlabeled


class Paris(MiniworldCity):

    labeled_image_paths = ['test/{}_x.png'.format(i) for i in range(391)] + \
                          ['train/{}_x.png'.format(i) for i in range(757)]

    label_paths = ['test/{}_y.png'.format(i) for i in range(391)] + \
                  ['train/{}_y.png'.format(i) for i in range(757)]

    unlabeled_image_paths = []

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    @property
    def __image_size__(cls):

        return 390*390

class ParisLabeled(Paris, BaseDatasetLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

class ParisUnlabeled(Paris, BaseDatasetUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
