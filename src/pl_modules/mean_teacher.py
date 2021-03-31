from argparse import ArgumentParser
import segmentation_models_pytorch as smp

import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import rot90, no_grad
from torch.optim import Adam
import copy
import pytorch_lightning.metrics as M
from metrics import MyMetricCollection
from callbacks import ArrayValLogger, ConfMatLogger
from losses import SoftDiceLoss, DiceCE

class MeanTeacher(pl.LightningModule):
    def __init__(self, arguments):

        super().__init__()

        # The effect of using imagenet pre-training instead is to be measured, but
        # for now we don't.
        network = smp.Unet(
            encoder_name="timm-regnetx_002",
            encoder_depth=1,
            decoder_channels=[256],
            encoder_weights=None,
            in_channels=arguments.in_channels,
            classes=arguments.num_classes,
        )

        self.student_network = network
        self.teacher_network = copy.deepcopy(network)
        self.save_hyperparameters()

        self.train_metrics = None
        self.val_metrics = None
        self.init_metrics(arguments)

        self.callbacks = None
        self.init_callbacks(arguments)

        # For the linear combination of loss
        self.unsup_loss_prop = arguments.unsup_loss_prop

        # Exponential moving average
        self.ema = arguments.ema

        self.loss = DiceCE()

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--unsup_loss_prop", type=float, default=0.5)
        parser.add_argument("--ema", type=float, default=0.95)
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument("--in_channels", type=int, default=3)

        return parser

    def init_metrics(self, args):

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
        self.train_metrics = MyMetricCollection(
            scalar_metrics_dict,
            "train_"
        )
        self.val_metrics = MyMetricCollection(
            copy.deepcopy(scalar_metrics_dict),
            "val_"
        )

    def init_callbacks(self, args):

        # Non-scalar metrics are bundled in callbacks that deal with logging them
        per_class_precision = M.Precision(
            num_classes=args.num_classes, mdmc_average="global", average="none"
        )
        per_class_precision_logger = ArrayValLogger(
            array_metric=per_class_precision, name="per_class_precision"
        )
        per_class_F1 = M.F1(num_classes=args.num_classes, average="none")
        per_class_F1_logger = ArrayValLogger(
            array_metric=per_class_F1, name="per_class_F1"
        )
        cm = ConfMatLogger(num_classes=args.num_classes)

        self.callbacks = [
            cm, per_class_F1_logger, per_class_precision_logger
        ]

    def forward(self, x):

        # Should it be an inference on the teacher network instead ?
        return self.teacher_network(x)

    def configure_optimizers(self):

        return Adam(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx):

        sup_data, unsup_data = batch["sup"], batch["unsup"]
        sup_train_inputs, sup_train_labels = sup_data

        student_outputs = self.student_network(sup_train_inputs)

        # Supervised learning
        sup_loss = self.loss(student_outputs, sup_train_labels)
        self.train_metrics(student_outputs.softmax(dim=1), sup_train_labels)

        augmentation = np.random.randint(5)
        augmented_inputs = rot90(unsup_data, k=augmentation, dims=[2, 3])
        student_outputs = self.student_network(augmented_inputs)

        # Enforcing consistency on unlabeled data
        with no_grad():
            teacher_outputs = self.teacher_network(unsup_data)
        teacher_outputs = rot90(teacher_outputs, k=augmentation, dims=[2, 3])

        unsup_loss = F.mse_loss(student_outputs, teacher_outputs)

        total_loss = sup_loss + self.unsup_loss_prop * unsup_loss

        self.log("sup_loss", sup_loss)
        self.log_dict(self.train_metrics)
        self.log("train_unsup_loss", unsup_loss)

        # Update teacher model in place
        ema = min(1.0 - 1.0 / float(self.global_step + 1), self.ema)
        for param_t, param in zip(self.teacher_network.parameters(),
                                  self.student_network.parameters()):
            param_t.data.mul_(ema).add_(param.data, alpha=1 - ema)

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):

        val_inputs, val_labels = batch
        outputs = self.teacher_network(val_inputs)
        sup_loss = self.loss(outputs, val_labels)
        softmax = outputs.softmax(dim=1)
        self.val_metrics(softmax, val_labels)

        self.log("val_sup_loss", sup_loss)
        self.log_dict(self.val_metrics)
