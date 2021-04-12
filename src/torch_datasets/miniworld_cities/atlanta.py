from torch_datasets import MiniworldCity
from torch_datasets import BaseDatasetLabeled, BaseDatasetUnlabeled


class Atlanta(MiniworldCity):

    labeled_image_paths = ['test/{}_x.png'.format(i) for i in [0]] + \
                          ['train/{}_x.png'.format(i) for i in [0,1]]

    label_paths = ['test/{}_y.png'.format(i) for i in [0]] + \
                  ['train/{}_y.png'.format(i) for i in [0,1]]

    unlabeled_image_paths = []

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    @property
    def __image_size__(cls):

        return 1800*1200

class AtlantaLabeled(Atlanta, BaseDatasetLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

class AtlantaUnlabeled(Atlanta, BaseDatasetUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
