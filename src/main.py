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
import pytorch_lightning.metrics as M
import segmentation_models_pytorch as smp

def main():

    # Hyperparameters
    NB_EPOCHS = 10
    IN_CHANNELS = 4
    NUM_CLASSES = 2
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
    # CSV_logger = loggers.CSVLogger(save_dir=log_dir,
    #                                name='csv')

    # Create network
    # network = Unet(IN_CHANNELS, NUM_CLASSES)
    network = smp.Unet(
        encoder_name='efficientnet-b0',
        encoder_weights='imagenet',
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        decoder_attention_type='scse'
    )

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
    

    accuracy = M.Accuracy(
        top_k=1,
        subset_accuracy=False
    )
    # average_precision = M.AveragePrecision(
    #     num_classes=NUM_CLASSES,
    # )
    # global_precision = M.Precision(
    #     num_classes=NUM_CLASSES,
    #     mdmc_average='global',
    #     average='macro'
    # )
    per_class_precision = M.Precision(
        num_classes=NUM_CLASSES,
        mdmc_average='global',
        average='weighted'
    )
    per_class_F1 = M.F1(
        num_classes=NUM_CLASSES,
        average='macro'
    )
    IoU = M.IoU(
        num_classes=NUM_CLASSES,
        reduction='elementwise_mean'
    )
    
    scalar_metrics = M.MetricCollection([
        accuracy,
        # average_precision,
        # global_precision,
        per_class_precision,
        per_class_F1,
        IoU
    ])
    
    pl_module = Semisup_segm(network,
                             scalar_metrics=scalar_metrics,
                             num_classes=NUM_CLASSES)

    trainer = Trainer(logger=TB_logger,
                      default_root_dir=OUTPUT_PATH,
                      max_epochs=NB_EPOCHS,
                      log_every_n_steps=1,
                      multiple_trainloader_mode='min_size')

    trainer.fit(model=pl_module, datamodule=pl_datamodule)

if __name__ == "__main__":
    # execute only if run as a script
    main()
