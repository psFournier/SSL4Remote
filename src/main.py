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
from pl_modules import Semisup_segm
from pl_datamodules import Isprs_semisup
from networks import Unet
# from metrics import MAPMetric
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning.metrics as M
from metrics import MyMetricCollection
import segmentation_models_pytorch as smp
from callbacks import Conf_mat_logger, Array_val_logger
from argparse import ArgumentParser
import shutil
from pytorch_lightning.profiler import PyTorchProfiler, SimpleProfiler
import torch.autograd.profiler as profiler
from copy import deepcopy

def main():

    parser = ArgumentParser(add_help=True)

    parser.add_argument('--output_dir',
                        type=str,
                        default='~/scratch',
                        help='Where to store results')

    parser = Unet.add_model_specific_args(parser)
    parser = Isprs_semisup.add_model_specific_args(parser)
    parser = Semisup_segm.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    log_dir = os.path.expanduser(
        os.path.join(
            args.output_dir, '%s_%s' % (current_date, 'unet_isprs/')
        )
    )
    TB_logger = loggers.TensorBoardLogger(save_dir=log_dir,
                                          name='tensorboard')
    CSV_logger = loggers.CSVLogger(save_dir=log_dir,
                                   name='csv')

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
        ToTensorV2()
    ])

    pl_datamodule = Isprs_semisup(
        args.data_dir,
        args.crop_size,
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

    global_precision = M.Precision(
        num_classes=args.num_classes,
        mdmc_average='global',
        average='macro'
    )

    IoU = M.IoU(
        num_classes=args.num_classes,
        reduction='elementwise_mean'
    )

    scalar_metrics_dict = {
        'acc': accuracy,
        'global_precision': global_precision,
        'IoU': IoU
    }
    train_scalar_metrics = MyMetricCollection(
        scalar_metrics_dict,
        "train_"
    )
    val_scalar_metrics = MyMetricCollection(
        deepcopy(scalar_metrics_dict),
        "val_"
    )


    pl_module = Semisup_segm(
        network = network,
        scalar_metrics={
            "train": train_scalar_metrics,
            "val": val_scalar_metrics
        },
        unsup_loss_prop=args.unsup_loss_prop
    )

    per_class_precision = M.Precision(
        num_classes=args.num_classes,
        mdmc_average='global',
        average='none'
    )
    per_class_precision_logger = Array_val_logger(
        array_metric=per_class_precision,
        name='per_class_precision'
    )

    per_class_F1 = M.F1(
        num_classes=args.num_classes,
        average='none'
    )
    per_class_F1_logger = Array_val_logger(
        array_metric=per_class_F1,
        name='per_class_F1'
    )
    cm = Conf_mat_logger(
        num_classes=args.num_classes
    )

    # profiler = PyTorchProfiler(
    #     output_filename=os.path.join(log_dir, 'profile'),
    #     # profile_memory=False,
    #     # use_cpu=True,
    #     # use_cuda=False
    # )

    trainer = Trainer.from_argparse_args(
        args,
        logger=TB_logger,
        callbacks=[
            cm,
            per_class_precision_logger,
            per_class_F1_logger
        ],
        # profiler=profiler
    )

    # with profiler.profile(with_stack=True, profile_memory=True) as prof:
    #     trainer.fit(
    #         model=pl_module,
    #         datamodule=pl_datamodule
    #     )
    # print(prof.key_averages(group_by_stack_n=5).table(
    #     sort_by='self_cpu_time_total', row_limit=5))

    trainer.fit(
        model=pl_module,
        datamodule=pl_datamodule
    )

if __name__ == "__main__":
    # execute only if run as a script
    main()
