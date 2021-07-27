from dl_toolbox.lightning_datamodules import BaseSupervisedDatamodule, BaseSemisupDatamodule
from dl_toolbox.torch_datasets import PhrPanDs
import glob
from torch.utils.data import ConcatDataset
import numpy as np
from functools import partial


class PhrPanDm(BaseSupervisedDatamodule):

    def __init__(self, image_path, label_path, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.image_path = image_path
        self.label_path = label_path

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--image_path", type=str)
        parser.add_argument("--label_path", type=str, default=None)

        return parser

    def setup(self, stage=None):

        sup_train_set = PhrPanDs(
            image_path=self.image_path,
            label_path=self.label_path,
            tile_size=2300,
            tile_step=2300,
            crop_size=self.crop_size
        )
        sup_train_set.idxs = sup_train_set.idxs[::3] + sup_train_set.idxs[1::3]
        val_set = PhrPanDs(
            image_path=self.image_path,
            label_path=self.label_path,
            tile_size=2300,
            tile_step=2300,
            crop_size=self.crop_size
        )
        val_set.idxs = val_set.idxs[2::3]


class PhrPanDmSemisup(PhrPanDm, BaseSemisupDatamodule):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def setup(self, stage=None):

        super(PhrPanDmSemisup, self).setup(stage=stage)

        self.unsup_train_set = PhrPanDs(
            image_path=self.image_path,
            tile_size=2300,
            tile_step=2300,
            crop_size=self.crop_size
        )