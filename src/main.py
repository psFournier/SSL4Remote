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
# from src.metrics import MAPMetric
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning.metrics as M
import segmentation_models_pytorch as smp
from src.callbacks import Conf_mat #, Map
from argparse import ArgumentParser
import shutil

def main():

    parser = ArgumentParser()

    parser.add_argument("--data_dir",
                        type=str,
                        default=os.environ['TMPDIR'])

    parser.add_argument('--nb_epochs',
                        type=int,
                        default=10)

    parser.add_argument('--batch_size',
                        type=int,
                        default=12)

    parser.add_argument('--nb_pass_per_epoch',
                        type=int,
                        default=2)

    parser.add_argument('--crop_size',
                        type=int,
                        default=128)

    parser.add_argument('--output_dir',
                        type=str,
                        default='~/scratch')

    parser.add_argument('--in_channels',
                        type=int,
                        default=4)

    parser.add_argument('--num_classes',
                        type=int,
                        default=2)

    parser.add_argument('--log_every_n_step',
                        type=int,
                        default=10)

    parser = Semisup_segm.add_model_specific_args(parser)

    args = parser.parse_args()

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    log_dir = os.path.expanduser(
        os.path.join(
            args.output_dir, '%s_%s' % (current_date, 'unet_isprs/')
        )
    )
    TB_logger = loggers.TensorBoardLogger(save_dir=log_dir,
                                          name='tensorboard')

    # Create network
    network = Unet(args.in_channels, args.num_classes)
    # network = smp.Unet(
    #     encoder_name='efficientnet-b0',
    #     encoder_weights='imagenet',
    #     in_channels=args.in_channels,
    #     classes=args.num_classes,
    #     decoder_attention_type='scse'
    # )

    transform = A.Compose([
        A.RandomCrop(args.crop_size, args.crop_size),
        ToTensorV2()
    ])

    pl_datamodule = Isprs_semisup(
        args.data_dir,
        args.nb_pass_per_epoch,
        args.batch_size,
        sup_train_transforms=transform,
        val_transforms=transform,
        unsup_train_transforms=transform
    )

    accuracy = M.Accuracy(
        top_k=1,
        subset_accuracy=False
    )
    # average_precision = MAPMetric()
    # average_precision = M.AveragePrecision(
    #     num_classes=NUM_CLASSES
    # )
    # global_precision = M.Precision(
    #     num_classes=NUM_CLASSES,
    #     mdmc_average='global',
    #     average='macro'
    # )
    per_class_precision = M.Precision(
        num_classes=args.num_classes,
        mdmc_average='global',
        average='weighted'
    )
    per_class_F1 = M.F1(
        num_classes=args.num_classes,
        average='macro'
    )
    IoU = M.IoU(
        num_classes=args.num_classes,
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

    pl_module = Semisup_segm(
        network = network,
        scalar_metrics=scalar_metrics,
        unsup_loss_prop=args.unsup_loss_prop
    )
    # pl_module = Semisup_segm(
    #     network,
    #     scalar_metrics=scalar_metrics,
    # )

    cm = Conf_mat(
        num_classes=args.num_classes
    )

    trainer = Trainer.from_argparse_args(
        args,
        logger=TB_logger,
        multiple_trainloader_mode='min_size',
        callbacks=[
            cm
        ]
    )

    trainer.fit(model=pl_module, datamodule=pl_datamodule)

if __name__ == "__main__":
    # execute only if run as a script
    main()
