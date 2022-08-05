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
import numpy as np
from dl_toolbox.augmentations import Cutmix

class MT(BaseModule):

    # MT = Mean Teacher

    def __init__(self,
                 encoder,
                 in_channels,
                 final_alpha,
                 alpha_milestones,
                 pseudo_threshold,
                 ema,
                 pretrained=True,
                 initial_lr=0.05,
                 final_lr=0.001,
                 lr_milestones=(0.5,0.9),
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.network1 = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=self.num_classes,
            decoder_use_batchnorm=True
        )

        self.network2 = deepcopy(self.network1)

        self.in_channels = in_channels
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.lr_milestones = list(lr_milestones)

        self.loss1 = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index
        )
        self.loss2 = DiceLoss(
            mode="multiclass",
            log_loss=False,
            from_logits=True,
            ignore_index=self.ignore_index
        )

        self.unsup_loss = nn.CrossEntropyLoss(
            reduction='none'
        )

        self.ema = ema
        self.final_alpha = final_alpha
        self.alpha_milestones = alpha_milestones
        self.alpha = 0.
        self.pseudo_threshold = pseudo_threshold
        self.cutmix = Cutmix(alpha=0.4)
        self.save_hyperparameters()


    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--ema", type=float)
        parser.add_argument("--final_alpha", type=float)
        parser.add_argument("--alpha_milestones", nargs=2, type=int)
        parser.add_argument("--pseudo_threshold", type=float)
        parser.add_argument("--in_channels", type=int)
        parser.add_argument("--pretrained", action='store_true')
        parser.add_argument("--encoder", type=str)
        parser.add_argument("--initial_lr", type=float)
        parser.add_argument("--final_lr", type=float)
        parser.add_argument("--lr_milestones", nargs='+', type=float)

        return parser


    def forward(self, x):
 
        return self.network2(x)

    def on_train_epoch_start(self):

        start = self.alpha_milestones[0]
        end = self.alpha_milestones[1]
        e = self.trainer.current_epoch
        alpha = self.final_alpha

        if e <= start:
            self.alpha = 0.
        elif e <= end:
            self.alpha = ((e - start) / (end - start)) * alpha
        else:
            self.alpha = alpha

    def update_teacher(self):

        # Update teacher model in place AFTER EACH BATCH?
        ema = min(1.0 - 1.0 / float(self.global_step + 1), self.ema)
        for param_t, param in zip(self.network2.parameters(),
                                  self.network1.parameters()):
            param_t.data.mul_(ema).add_(param.data, alpha=1 - ema)

    def training_step(self, batch, batch_idx):

        batch, unsup_batch = batch["sup"], batch["unsup"]

        inputs = batch['image']
        labels = batch['mask']

        logits1 = self.network1(inputs)
        loss1 = self.loss1(logits1, labels) 
        loss2 = self.loss2(logits1, labels)
        loss = loss1 + loss2
        self.log('Train_sup_CE', loss1)
        self.log('Train_sup_Dice', loss2)
        self.log('Train_sup_loss', loss)

        if self.trainer.current_epoch > self.alpha_milestones[0]:
            
            unsup_inputs = unsup_batch['image']

            with torch.no_grad():
                teacher_probs = self.network2(unsup_inputs).softmax(dim=1)

            cutmixed_inputs, cutmixed_probs = self.cutmix(
                input_batch=unsup_inputs,
                target_batch=teacher_probs
            )

            cutmix_confs, cutmix_preds = torch.max(cutmixed_probs, dim=1)
            cutmixed_logits = self.network1(cutmixed_inputs)
            loss_no_reduce = self.unsup_loss(
                cutmixed_logits,
                cutmix_preds
            )

            cutmix_certain = cutmix_confs > self.pseudo_threshold
            certain = torch.sum(cutmix_certain)
            cutmix_loss = torch.sum(cutmix_certain * loss_no_reduce) / certain

            self.log('Cutmix consistency loss', cutmix_loss)

            loss += self.alpha * cutmix_loss

        self.update_teacher()
        self.log('Prop unsup train', self.alpha)
        self.log("Train_loss", loss)

        return {'batch': batch, "loss": loss}

    def validation_step(self, batch, batch_idx):

        res_dict = super().validation_step(batch, batch_idx)
        logits = res_dict['logits']
        labels = batch['mask']
        
        loss1 = self.loss1(logits, labels)
        loss2 = self.loss2(logits, labels)
        loss = loss1 + loss2
        self.log('Val_CE', loss1)
        self.log('Val_Dice', loss2)
        self.log('Val_loss', loss)

        probas = logits.softmax(dim=1)
        calib_error = torchmetrics.calibration_error(
            probas,
            labels
        )
        self.log('Calibration error', calib_error)

        return {**res_dict, **{'probas': probas.detach()}}

    def on_train_epoch_end(self):
        for param_group in self.optimizer.param_groups:
            self.log(f'learning_rate', param_group['lr'])
            break
