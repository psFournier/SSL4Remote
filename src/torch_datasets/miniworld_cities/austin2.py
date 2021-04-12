from torch_datasets import MiniworldCity
from torch_datasets import BaseDatasetLabeled, BaseDatasetUnlabeled


class Austin2(MiniworldCity):

    labeled_image_paths = ['test/{}_x.png'.format(i) for i in range(1)] + \
                          ['train/{}_x.png'.format(i) for i in range(2)]

    label_paths = ['test/{}_y.png'.format(i) for i in range(1)] + \
                  ['train/{}_y.png'.format(i) for i in range(2)]

    unlabeled_image_paths = []

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    @property
    def __image_size__(cls):

        return 3063*3478

class Austin2Labeled(Austin2, BaseDatasetLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

class Austin2Unlabeled(Austin2, BaseDatasetUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
