from argparse import ArgumentParser
import segmentation_models_pytorch as smp

import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import rot90, no_grad
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    ExponentialLR,
    CyclicLR,
    MultiStepLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)
import copy
import pytorch_lightning.metrics as M
from metrics import MetricCollection
from callbacks import ArrayValLogger, ConfMatLogger
from common_utils.scheduler import get_scheduler
from common_utils.losses import get_loss
from torchvision.transforms import ConvertImageDtype
import torch
from pytorch_toolbelt.losses import DiceLoss

class SupervisedBaseline(pl.LightningModule):

    def __init__(self,
                 encoder,
                 pretrained,
                 in_channels,
                 num_classes,
                 inplaceBN,
                 learning_rate,
                 class_weights,
                 wce,
                 *args,
                 **kwargs):

        super().__init__()

        network = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
            decoder_use_batchnorm='inplace' if inplaceBN else True
        )

        self.network = network
        self.save_hyperparameters()

        self.learning_rate = learning_rate

        self.train_metrics = None
        self.val_metrics = None
        self.init_metrics(num_classes)

        self.callbacks = []
        self.init_callbacks(num_classes)

        self.class_weights = class_weights if wce else torch.FloatTensor(
            [1.] * num_classes
        )
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(mode="multiclass", log_loss=False)


    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument("--in_channels", type=int, default=3)
        parser.add_argument("--pretrained", action='store_true')
        parser.add_argument("--encoder", type=str, default='timm-efficientnet-b0')
        parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3,
                            help="Initial learning rate")
        parser.add_argument("--inplaceBN", action='store_true' )
        parser.add_argument("--wce", action='store_true')

        return parser

    def init_metrics(self, num_classes):

        # Scalar metrics are separated because lightning can deal with logging
        # them automatically.
        accuracy = M.Accuracy(top_k=1, subset_accuracy=False)
        global_precision = M.Precision(
            num_classes=num_classes, mdmc_average="global", average="macro"
        )
        iou = M.IoU(num_classes=num_classes, reduction="elementwise_mean")

        scalar_metrics_dict = {
            "acc": accuracy,
            # "global_precision": global_precision,
            "IoU": iou
        }

        # Two things here:
        # 1. MyMetricCollection adds a prefix to metrics names, and should be
        # included in future versions of lightning
        # 2. The metric objects keep statistics during runs, and deepcopy should be
        # necessary to ensure these stats do not overlap
        self.train_metrics = MetricCollection(
            scalar_metrics_dict,
            prefix="train_"
        )
        self.val_metrics = MetricCollection(
            copy.deepcopy(scalar_metrics_dict),
            prefix="val_"
        )

    def init_callbacks(self, num_classes):

        # Non-scalar metrics are bundled in callbacks that deal with logging them
        # per_class_precision = M.Precision(
        #     num_classes=num_classes, mdmc_average="global", average="none"
        # )
        # per_class_precision_logger = ArrayValLogger(
        #     array_metric=per_class_precision, name="per_class_precision"
        # )
        # self.callbacks.append(per_class_precision_logger)
        #
        # per_class_F1 = M.F1(num_classes=num_classes, average="none")
        # per_class_F1_logger = ArrayValLogger(
        #     array_metric=per_class_F1, name="per_class_F1"
        # )
        # self.callbacks.append(per_class_F1_logger)

        cm = ConfMatLogger(num_classes=num_classes)
        self.callbacks.append(cm)

    def forward(self, x):

        return self.network(x)

    def configure_optimizers(self):

        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = MultiStepLR(
            optimizer,
            milestones=[
                int(self.trainer.max_epochs * 0.5),
                int(self.trainer.max_epochs * 0.7),
                int(self.trainer.max_epochs * 0.9)],
            gamma=0.3
        )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        train_inputs, train_labels = batch
        outputs = self.network(train_inputs)
        train_labels = train_labels.long()
        train_loss1 = self.ce(outputs, train_labels)
        train_loss2 = self.dice(outputs, train_labels)
        train_loss = train_loss1 + train_loss2

        self.log('Cross entropy loss', train_loss1)
        self.log('Dice loss', train_loss2)

        self.train_metrics(outputs.softmax(dim=1), train_labels)
        self.log_dict(self.train_metrics)

        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):

        val_inputs, val_labels = batch
        outputs = self.network(val_inputs)
        val_labels = val_labels.long()
        val_loss1 = self.ce(outputs, val_labels)
        val_loss2 = self.dice(outputs, val_labels)
        val_loss = val_loss1 + val_loss2

        self.val_metrics(outputs.softmax(dim=1), val_labels)
        self.log("val_sup_loss", val_loss)
        self.log_dict(self.val_metrics)
