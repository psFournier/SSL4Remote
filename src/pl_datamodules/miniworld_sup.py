from pl_datamodules import BaseSupervisedDatamodule
from torch_datasets import *
from torch import tensor

cities = {
    'christchurch': ChristchurchLabeled,
    'austin': AustinLabeled,
    'chicago': ChicagoLabeled,
    'kitsap': KitsapLabeled,
    'tyrol-w': TyrolwLabeled,
    'vienna': ViennaLabeled
}

class MiniworldSup(BaseSupervisedDatamodule):

    class_weights = tensor(
        [1., 1.]
    )

    def __init__(self, city, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.city = city

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--city", type=str, default='christchurch',
                            help="Which city to train on.")

        return parser

    def setup(self, stage=None):

        self.sup_train_set = cities[self.city](
            data_path=self.data_dir,
            crop=self.crop_size
        )

        self.val_set = cities[self.city](
            data_path=self.data_dir,
            crop=self.crop_size,
            fixed_crop=True
        )

        nb_labeled_images = len(self.sup_train_set.labeled_image_paths)
        labeled_idxs = list(range(nb_labeled_images))

        if self.train_val == (0, 0):
            train, val = self.sup_train_set.default_train_val
        else:
            train, val = self.train_val

        self.sup_train_set.path_idxs = labeled_idxs[:train]
        self.val_set.path_idxs = labeled_idxs[-val:]