"""
In this file is a very simple code training a Unet on ISPRS_VAIHINGEN.
The Unet should come from segmentation_models.pytorch.
We do not use Pytorch.Lightning nor Hydra, but this files intends to make
clear what these two layers will replace later.
Being some kind of tutorial file, it should be self-contained.
The file follows the sampleISPRS from ai4geo_dl
"""

import os
import datetime
from pytorch_lightning import Trainer, loggers
from src.pl_modules import Semisup_segm
from src.pl_datamodules import Isprs_semisup
from src.networks import Unet
import albumentations as A
from albumentations.pytorch import ToTensorV2
def main():

    # Hyperparameters
    NB_EPOCHS = 10
    IN_CHANNELS = 4
    OUT_CHANNELS = 2
    BATCH_SIZE = 12
    DATA_PATH = '/home/pierre/Documents/ONERA/ai4geo/ISPRS_VAIHINGEN'
    CROP_SIZE = 128
    NB_PASS_PER_EPOCH = 2
    OUTPUT_PATH = '/home/pierre/PycharmProjects/RemoteSensing/outputs'

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    log_dir = os.path.expanduser(
        os.path.join(OUTPUT_PATH, '%s_%s' % (current_date, 'unet_isprs/')))

    TB_logger = loggers.TensorBoardLogger(save_dir=log_dir,
                                          name='tensorboard')
    CSV_logger = loggers.CSVLogger(save_dir=log_dir,
                                   name='csv')

    # Create network
    network = Unet(IN_CHANNELS, OUT_CHANNELS)

    transform = A.Compose([
        A.RandomCrop(CROP_SIZE, CROP_SIZE),
        ToTensorV2()
    ])

    pl_datamodule = Isprs_semisup(DATA_PATH,
                                  NB_PASS_PER_EPOCH,
                                  BATCH_SIZE,
                                  sup_train_transforms=transform,
                                  val_transforms=transform,
                                  unsup_train_transforms=transform)

    pl_module = Semisup_segm(network)

    trainer = Trainer(logger=[TB_logger,
                              CSV_logger],
                      default_root_dir=OUTPUT_PATH,
                      max_epochs=NB_EPOCHS,
                      log_every_n_steps=5)

    trainer.fit(model=pl_module, datamodule=pl_datamodule)

if __name__ == "__main__":
    # execute only if run as a script
    main()
