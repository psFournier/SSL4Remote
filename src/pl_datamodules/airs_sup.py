from torch_datasets import Airs, AirsLabeled
from pl_datamodules import BaseSupervisedDatamodule
import random
from torch import tensor


class AirsSup(BaseSupervisedDatamodule):

    class_weights = tensor(
        [
            Airs.pixels_per_class[0] / ppc for ppc in
            Airs.pixels_per_class
        ]
    )

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def setup(self, stage=None):

        nb_labeled_images = Airs.nb_labeled_images
        labeled_idxs = list(range(nb_labeled_images))
        train, val = Airs.default_train_val

        self.sup_train_set = AirsLabeled(
            data_path=self.data_dir,
            idxs=labeled_idxs[:train][::self.prop_train],
            crop=self.crop_size,
            augmentations=self.train_augment
        )

        self.val_set = AirsLabeled(
            data_path=self.data_dir,
            idxs=labeled_idxs[train:train+val],
            crop=self.crop_size,
            augmentations=self.val_augment,
            fixed_crop=True
        )