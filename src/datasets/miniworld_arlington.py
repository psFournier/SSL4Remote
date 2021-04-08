from datasets import MiniworldCity
from datasets import BaseDatasetLabeled, BaseDatasetUnlabeled


class MiniworldArlington(MiniworldCity):

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

class MiniworldArlingtonLabeled(MiniworldArlington, BaseDatasetLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

class MiniworldArlingtonUnlabeled(MiniworldArlington, BaseDatasetUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
