import datetime
import os
from argparse import ArgumentParser
from copy import deepcopy

import albumentations as A
import pytorch_lightning.metrics as M
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import Trainer, loggers

from callbacks import Array_val_logger, Conf_mat_logger
from metrics import MyMetricCollection
from networks import Unet
from pl_datamodules import Isprs_semisup
from pl_modules import Semisup_segm


def main():

    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./outputs")

    # Each class of interest contains a method to add its specific arguments
    # to the parser
    parser = Unet.add_model_specific_args(parser)
    parser = Isprs_semisup.add_model_specific_args(parser)
    parser = Semisup_segm.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    log_dir = os.path.expanduser(
        os.path.join(args.output_dir, "%s_%s" % (current_date, "unet_isprs/"))
    )
    tensorboard = loggers.TensorBoardLogger(save_dir=log_dir, name="tensorboard")

    # The effect of using imagenet pre-training instead is to be measured, but
    # for now we don't.
    network = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=args.in_channels,
        classes=args.num_classes,
    )

    # Additional transforms should be employed: which ones?
    transform = A.Compose([ToTensorV2()])

    # The lightning datamodule deals with instantiating the proper dataloaders.
    pl_datamodule = Isprs_semisup(
        args.data_dir,
        args.crop_size,
        args.nb_pass_per_epoch,
        args.batch_size,
        sup_train_transforms=transform,
        val_transforms=transform,
        unsup_train_transforms=transform,
    )

    # Scalar metrics are separated because lightning can deal with logging
    # them automatically.
    accuracy = M.Accuracy(top_k=1, subset_accuracy=False)
    global_precision = M.Precision(
        num_classes=args.num_classes, mdmc_average="global", average="macro"
    )
    iou = M.IoU(num_classes=args.num_classes, reduction="elementwise_mean")

    scalar_metrics_dict = {
        "acc": accuracy,
        "global_precision": global_precision,
        "IoU": iou,
    }

    # Two things here:
    # 1. MyMetricCollection adds a prefix to metrics names, and should be
    # included in future versions of lightning
    # 2. The metric objects keep statistics during runs, and deepcopy should be
    # necessary to ensure these stats do not overlap
    train_scalar_metrics = MyMetricCollection(scalar_metrics_dict, "train_")
    val_scalar_metrics = MyMetricCollection(deepcopy(scalar_metrics_dict), "val_")

    # This lightning module is where the training schema is implemented.
    pl_module = Semisup_segm(
        network=network,
        scalar_metrics={"train": train_scalar_metrics, "val": val_scalar_metrics},
        unsup_loss_prop=args.unsup_loss_prop,
    )

    # Non-scalar metrics are bundled in callbacks that deal with logging them
    per_class_precision = M.Precision(
        num_classes=args.num_classes, mdmc_average="global", average="none"
    )
    per_class_precision_logger = Array_val_logger(
        array_metric=per_class_precision, name="per_class_precision"
    )
    per_class_F1 = M.F1(num_classes=args.num_classes, average="none")
    per_class_F1_logger = Array_val_logger(
        array_metric=per_class_F1, name="per_class_F1"
    )
    cm = Conf_mat_logger(num_classes=args.num_classes)

    # Using from_argparse_args enables to use any standard parameter of the
    # lightning Trainer class without having to manually add them to the parser.
    trainer = Trainer.from_argparse_args(
        args,
        logger=tensorboard,
        callbacks=[cm, per_class_precision_logger, per_class_F1_logger],
    )

    trainer.fit(model=pl_module, datamodule=pl_datamodule)


if __name__ == "__main__":

    main()
