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
from dl_toolbox.utils import TorchOneHot

class Unet_BCE_binary(BaseModule):

    # BCE = Binary Cross Entropy for binary classif

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
        self.num_classes = 1 
        self.ignore_index = -1
        self.network = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=1,
            decoder_use_batchnorm=True
        )
        self.in_channels = in_channels
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.lr_milestones = list(lr_milestones)
        self.bce = nn.BCEWithLogitsLoss(
            reduction='none'
        )
        self.dice = DiceLoss(
            mode="binary",
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
        mask = torch.ones_like(labels, dtype=labels.dtype, device=labels.device)
        logits = self.network(inputs).squeeze()
        bce = self.bce(logits, labels.float())
        bce = torch.sum(mask * bce) / torch.sum(mask)
        dice = self.dice(logits*mask, labels*mask)
        loss = bce + dice
        self.log('Train_sup_BCE', bce)
        self.log('Train_sup_Dice', dice)
        self.log('Train_sup_loss', loss)

        return {'batch': batch, 'logits': logits.detach(), "loss": loss}

    def validation_step(self, batch, batch_idx):

        inputs = batch['image']
        labels = batch['mask']
        mask = torch.ones_like(labels, dtype=labels.dtype, device=labels.device)
        logits = self.forward(inputs).squeeze()
        probas = torch.sigmoid(logits)
        probas = torch.stack([1-probas, probas], dim=1)
        calib_error = torchmetrics.calibration_error(
            probas,
            labels
        )
        self.log('Calibration error', calib_error)
        stat_scores = torchmetrics.stat_scores(
            probas,
            labels,
            ignore_index=None,
            mdmc_reduce='global',
            reduce='macro',
            threshold=0.5,
            top_k=1,
            num_classes=2
        )
        bce = self.bce(logits, labels.float())
        bce = torch.sum(mask * bce) / torch.sum(mask)
        dice = self.dice(logits*mask, labels*mask)
        loss = bce + dice
        self.log('Val_BCE', bce)
        self.log('Val_Dice', dice)
        self.log('Val_loss', loss)

        return {'batch': batch,
                'logits': logits.detach(),
                'stat_scores': stat_scores.detach(),
                'probas': probas.detach()
                }
