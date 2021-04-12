from torch_datasets import MiniworldCity
from torch_datasets import BaseDatasetLabeled, BaseDatasetUnlabeled


class Rio(MiniworldCity):

    labeled_image_paths = ['test/{}_x.png'.format(i) for i in range(2360)] + \
                          ['train/{}_x.png'.format(i) for i in range(4580)]

    label_paths = ['test/{}_y.png'.format(i) for i in range(2360)] + \
                  ['train/{}_y.png'.format(i) for i in range(4580)]

    unlabeled_image_paths = []

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    @property
    def __image_size__(cls):

        return 439*406

class RioLabeled(Rio, BaseDatasetLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

class RioUnlabeled(Rio, BaseDatasetUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
