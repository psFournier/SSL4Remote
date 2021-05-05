from abc import ABC
from torch_datasets import Base
import glob


class BaseCity(Base, ABC):

    def __init__(self, city, *args, **kwargs):

        super().__init__(*args, **kwargs)

        train_labeled_image_paths = sorted(
            glob.glob(f'{self.data_path}/{city}/train/*_x.tif')
        )
        test_labeled_image_paths = sorted(
            glob.glob(f'{self.data_path}/{city}/test/*_x.tif')
        )
        self.labeled_image_paths = train_labeled_image_paths + test_labeled_image_paths

        train_label_paths = sorted(
            glob.glob(f'{self.data_path}/{city}/train/*_y.tif')
        )
        test_label_paths = sorted(
            glob.glob(f'{self.data_path}/{city}/test/*_y.tif')
        )
        self.label_paths = train_label_paths + test_label_paths

        self.unlabeled_image_paths = []

        self.default_train_val = (
            len(train_label_paths),
            len(test_label_paths)
        )
