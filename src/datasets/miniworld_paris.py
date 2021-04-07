from datasets import MiniworldCity, MiniworldCityUnlabeled, MiniworldCityLabeled


class MiniworldParis(MiniworldCity):

    labeled_image_paths = [
        'test/{}_x.png'.format(i) for i in range(391)
    ] + [
        'train/{}_x.png'.format(i) for i in range(757)
    ]

    label_paths = [
        'test/{}_y.png'.format(i) for i in range(391)
    ] + [
        'train/{}_y.png'.format(i) for i in range(757)
    ]

    unlabeled_image_paths = []

    def __init__(self, data_path, idxs, crop):

        super().__init__(data_path, idxs, crop)

    @property
    def __image_size__(cls):

        return 390*390

class MiniworldParisLabeled(MiniworldParis, MiniworldCityLabeled):

    def __init__(self, data_path, idxs, crop):

        super().__init__(data_path, idxs, crop)

class MiniworldParisUnlabeled(MiniworldParis, MiniworldCityUnlabeled):

    def __init__(self, data_path, idxs, crop):

        super().__init__(data_path, idxs, crop)