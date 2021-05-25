from pl_datamodules import MiniworldSup, BaseSemisupDatamodule
from torch_datasets import *
from torch import tensor

cities_unlabeled = {
    'christchurch': ChristchurchUnlabeled,
    'austin': AustinUnlabeled,
    'chicago': ChicagoUnlabeled,
    'kitsap': KitsapUnlabeled,
    'tyrol-w': TyrolwUnlabeled,
    'vienna': ViennaUnlabeled
}

cities_labeled = {
    'christchurch': ChristchurchLabeled,
    'austin': AustinLabeled,
    'chicago': ChicagoLabeled,
    'kitsap': KitsapLabeled,
    'tyrol-w': TyrolwLabeled,
    'vienna': ViennaLabeled
}


class MiniworldSemisup(MiniworldSup, BaseSemisupDatamodule):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def setup(self, stage=None):

        self.sup_train_set = cities_labeled[self.city](
            data_path=self.data_dir,
            crop=self.crop_size
        )

        self.val_set = cities_labeled[self.city](
            data_path=self.data_dir,
            crop=self.crop_size,
            fixed_crop=True
        )

        self.unsup_train_set = cities_unlabeled[self.city](
            data_path = self.data_dir,
            crop=self.crop_size
        )

        train, val = self.train_val
        nb_labeled_images = len(self.sup_train_set.labeled_image_paths)
        labeled_idxs = list(range(nb_labeled_images))
        self.sup_train_set.path_idxs = labeled_idxs[:train]
        self.val_set.path_idxs = labeled_idxs[train:train+val]
        self.unsup_train_set.path_idxs = labeled_idxs[train+val:train+val+self.unsup_train]