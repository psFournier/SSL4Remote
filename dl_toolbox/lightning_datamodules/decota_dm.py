from argparse import ArgumentParser
import os
import csv

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from rasterio.windows import Window

from dl_toolbox.lightning_datamodules import SupervisedDm
from dl_toolbox.utils import worker_init_function
from dl_toolbox.torch_collate import CustomCollate
from dl_toolbox.torch_datasets import *
from .utils import read_splitfile


class SemisupDm(SupervisedDm):

    def __init__(
        self,
        unsup_batch_size,
        unsup_crop_size,
        unsup_splitfile_path,
        unsup_train_folds,
        *args,
        **kwargs
    ):
        
        super().__init__(*args, **kwargs)
        self.unsup_batch_size = unsup_batch_size
        self.unsup_crop_size = unsup_crop_size
        self.unsup_splitfile_path = unsup_splitfile_path
        self.unsup_train_folds = unsup_train_folds

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--unsup_batch_size", type=int, default=16)
        parser.add_argument("--unsup_crop_size", type=int, default=160)
        parser.add_argument("--unsup_splitfile_path", type=str)
        parser.add_argument("--unsup_train_folds", nargs='+', type=int)
        return parser

    def setup(self, stage=None):

        super().setup(stage=stage)
        #with open(self.unsup_splitfile_path, newline='') as splitfile:
        #    train_sets, _ = build_split_from_csv(
        #        splitfile=splitfile,
        #        dataset_cls=self.dataset_cls,
        #        train_folds=self.unsup_train_folds,
        #        test_folds=(),
        #        img_aug=self.img_aug,
        #        data_path=self.data_path,
        #        crop_size = self.unsup_crop_size,
        #        one_hot=True
        #    )
        #self.unsup_train_set = ConcatDataset(train_sets)

        with open(self.unsup_splitfile_path, newline='') as splitfile:
            train_args, _ = read_splitfile(
                splitfile=splitfile,
                data_path=self.data_path,
                train_folds=self.unsup_train_folds,
                test_folds=()
            )

        if train_args:
            self.unsup_train_set = ConcatDataset([
                cls(
                    labels=self.labels,
                    label_merger=self.label_merger,
                    img_aug=self.img_aug,
                    crop_size=self.unsup_crop_size,
                    crop_step=self.unsup_crop_size,
                    one_hot=True,
                    **kwarg
                ) for cls, kwarg in train_args
            ])


       # unlabeled_paths = [
       #     #('Toulouse','normalized_mergedTO.tif'),
       #     #('Strasbourg','ORT_P1BPX-2018062038865324CP_epsg32632_decoup.tif'),
       #     #('Biarritz','biarritz_ortho_cropped.tif'),
       #     #('Paris','emprise_ORTHO_cropped.tif'),
       #     #('Montpellier','montpellier_ign_cropped.tif')
       #     ('Toulouse','toulouse_full_tiled.tif'),
       #     ('Strasbourg','strasbourg_full_tiled.tif'),
       #     ('Biarritz','biarritz_full_tiled.tif'),
       #     ('Paris','paris_full_tiled.tif'),
       #     ('Montpellier','montpellier_full_tiled.tif')

       # ]
       # unlabeled_sets = []

       # for path in unlabeled_paths:
       #     m = DigitanieDs.DATASET_DESC['min'][path[0]][:3]
       #     M = DigitanieDs.DATASET_DESC['max'][path[0]][:3]
       #     big_raster_path = os.path.join(self.data_path, path[1])
       #     width, height = imagesize.get(big_raster_path)
       #     tile = Window(0, 0, width, height)
       #     unlabeled_sets.append(
       #         DigitanieDs(
       #             image_path=big_raster_path,
       #             tile=tile,
       #             fixed_crops=False,
       #             read_window_fn=read_window_basic_gdal,
       #             norm_fn=partial(
       #                 minmax,
       #                 m=m,
       #                 M=M
       #             ),
       #             crop_size=self.unsup_crop_size,
       #             crop_step=self.unsup_crop_size,
       #             img_aug=self.img_aug
       #         )
       #     )
       # 
       # self.unsup_train_set = ConcatDataset(unlabeled_sets) 

    def train_dataloader(self):

        train_dataloader = super().train_dataloader()
        unsup_train_sampler = RandomSampler(
            data_source=self.unsup_train_set,
            replacement=True,
            num_samples=self.epoch_len
        )

        unsup_train_dataloader = DataLoader(
            dataset=self.unsup_train_set,
            batch_size=self.unsup_batch_size,
            sampler=unsup_train_sampler,
            collate_fn=CustomCollate(batch_aug='no'),
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_function
        )

        train_dataloaders = {
            "sup": train_dataloader,
            "unsup": unsup_train_dataloader
        }

        return train_dataloaders

def main():

    datamodule = SemisupDm(
        dataset_cls=SemcityBdsdDs,
        data_path='/d/pfournie/ai4geo/data/SemcityTLS_DL',
        splitfile_path='/d/pfournie/ai4geo/split_semcity.csv',
        #data_path='/d/pfournie/ai4geo/data/DIGITANIE',
        #splitfile_path='/d/pfournie/ai4geo/split_toulouse.csv',
        test_folds=(4,),
        train_folds=(0,1,2,3),
        crop_size=128,
        epoch_len=100,
        sup_batch_size=16,
        workers=0,
        img_aug='d4_color-0',
        batch_aug='no',
        unsup_splitfile_path='/d/pfournie/ai4geo/split_semcity_unlabeled.csv',
        unsup_batch_size=8,
        unsup_crop_size=140,
        unsup_train_folds=(0,1,2,3,4)
    )

    datamodule.setup()
    sup_dataloader = datamodule.train_dataloader()['sup']
    unsup_dataloader = datamodule.train_dataloader()['unsup']
    for batch in unsup_dataloader:

        print(batch['image'].shape)

if __name__ == '__main__':

    main()

