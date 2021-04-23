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
        random.shuffle(labeled_idxs)

        self.sup_train_set = AirsLabeled(
            data_path=self.data_dir,
            idxs=labeled_idxs[:857][::self.prop_train],
            crop=self.crop_size,
            augmentations=self.train_augment
        )

        self.val_set = AirsLabeled(
            data_path=self.data_dir,
            idxs=labeled_idxs[857:],
            crop=self.crop_size,
            augmentations=self.val_augment
        )