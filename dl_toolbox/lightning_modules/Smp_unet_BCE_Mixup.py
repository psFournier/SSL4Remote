from argparse import ArgumentParser
import segmentation_models_pytorch as smp
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
import torch
import torchmetrics.functional as torchmetrics
from dl_toolbox.losses import DiceLoss
from copy import deepcopy
import torch.nn.functional as F

from dl_toolbox.lightning_modules.utils import *
from dl_toolbox.lightning_modules import BaseModule
from dl_toolbox.augmentations import Mixup
from dl_toolbox.utils import TorchOneHot

class Smp_Unet_BCE_Mixup(BaseModule):

    # BCE_Mixup = Binary Cross Entropy multilabel + mixup

    def __init__(self,
                 encoder,
                 in_channels,
                 pretrained=True,
                 initial_lr=0.05,
                 final_lr=0.001,
                 lr_milestones=(0.5,0.9),
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)
        
        self.network = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=self.num_classes,
            decoder_use_batchnorm=True
        )
        self.in_channels = in_channels
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.lr_milestones = list(lr_milestones)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.onehot = TorchOneHot(range(self.num_classes))
        self.mixup = Mixup(alpha=0.4)
        self.dice = DiceLoss(
            mode="multilabel",
            log_loss=False,
            from_logits=True
        )
        self.save_hyperparameters()

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--in_channels", type=int)
        parser.add_argument("--pretrained", action='store_true')
        parser.add_argument("--encoder", type=str)
        parser.add_argument("--initial_lr", type=float)
        parser.add_argument("--final_lr", type=float)
        parser.add_argument("--lr_milestones", nargs='+', type=float)

        return parser

    def forward(self, x):
        
        return self.network(x)

    def training_step(self, batch, batch_idx):

        inputs = batch['image']
        labels = batch['mask']
        onehot_labels = self.onehot(labels).float()
        mixed_inputs, mixed_labels = self.mixup(inputs, onehot_labels)
        
        mask = torch.ones_like(
            onehot_labels,
            dtype=onehot_labels.dtype,
            device=onehot_labels.device
        )
        mask -= mixed_labels[:, [self.ignore_index], ...]
        
        logits = self.network(mixed_inputs)
        bce = self.bce(logits, mixed_labels)
        bce = torch.sum(mask * bce) / torch.sum(mask)
        dice = self.dice(logits * mask, mixed_labels * mask)
        loss = loss1 + loss2
        
        self.log('Train_sup_BCE', loss1)
        self.log('Train_sup_Dice', loss2)
        self.log('Train_sup_loss', loss)

        return {'batch': batch, 'logits': logits.detach(), "loss": loss}

    def validation_step(self, batch, batch_idx):

        inputs = batch['image']
        labels = batch['mask']
        logits = self.forward(inputs).squeeze()
        probas = torch.sigmoid(logits)

        calib_error = torchmetrics.calibration_error(
            probas,
            labels
        )
        self.log('Calibration error', calib_error)

        stat_scores = torchmetrics.stat_scores(
            probas,
            labels,
            ignore_index=self.ignore_index if self.ignore_index >= 0 else None,
            mdmc_reduce='global',
            reduce='macro',
            threshold=0.5,
            top_k=1,
            num_classes=self.num_classes
        )
        
        labels = self.onehot(labels)
        loss1 = self.loss1(logits, labels)
        loss2 = self.loss2(logits, labels)
        loss = loss1 + loss2
        self.log('Val_BCE', loss1)
        self.log('Val_Dice', loss2)
        self.log('Val_loss', loss)

        return {'batch': batch,
                'logits': logits.detach(),
                'stat_scores': stat_scores.detach(),
                'probas': probas.detach()
                }
