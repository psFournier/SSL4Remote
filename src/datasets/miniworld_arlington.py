from datasets import MiniworldCity, MiniworldCityLabeled, MiniworldCityUnlabeled


class MiniworldArlington(MiniworldCity):

    labeled_image_paths = ['test/{}_x.png'.format(i) for i in [0]] + \
                          ['train/{}_x.png'.format(i) for i in [0,1]]

    label_paths = ['test/{}_y.png'.format(i) for i in [0]] + \
                  ['train/{}_y.png'.format(i) for i in [0,1]]

    unlabeled_image_paths = []

    def __init__(self, data_path, idxs, crop):

        super().__init__(data_path, idxs, crop)

    @property
    def __image_size__(cls):

        return 3000*3000

class MiniworldArlingtonLabeled(MiniworldArlington, MiniworldCityLabeled):

    def __init__(self, data_path, idxs, crop):

        super().__init__(data_path, idxs, crop)

class MiniworldArlingtonUnlabeled(MiniworldArlington, MiniworldCityUnlabeled):

    def __init__(self, data_path, idxs, crop):

        super().__init__(data_path, idxs, crop)
