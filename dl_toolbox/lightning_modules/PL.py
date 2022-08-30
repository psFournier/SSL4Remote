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

class PL(BaseModule):
    
    # PL = Pseudo Labelling

    def __init__(self,
                 encoder,
                 in_channels,
                 final_alpha,
                 alpha_milestones,
                 pseudo_threshold,
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

        self.final_alpha = final_alpha
        self.alpha_milestones = alpha_milestones
        self.alpha = 0.
        self.pseudo_threshold = pseudo_threshold
        self.save_hyperparameters()

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
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

    def forward(self, x):
        
        return self.network(x)

    def training_step(self, batch, batch_idx):

        batch, unsup_batch = batch["sup"], batch["unsup"]

        inputs = batch['image']
        labels = batch['mask']

        logits = self.network(inputs)
        loss1 = self.loss1(logits, labels)
        loss2 = self.loss2(logits, labels)
        loss = loss1 + loss2

        self.log('Train_sup_CE', loss1)
        self.log('Train_sup_Dice', loss2)
        self.log('Train_sup_loss', loss)

        if self.trainer.current_epoch > self.alpha_milestones[0]:

            unsup_inputs = unsup_batch['image']
            unsup_outputs = self.network(unsup_inputs)
            
            pseudo_probs = unsup_outputs.detach().softmax(dim=1)
            
            pseudo_probas, pseudo_preds = torch.max(pseudo_probs, dim=1)
            loss_no_reduce = self.unsup_loss(
                unsup_outputs,
                pseudo_preds
            )
            pseudo_certain = pseudo_probas > self.pseudo_threshold
            certain = torch.sum(pseudo_certain)
            pseudo_loss = torch.sum(pseudo_certain * loss_no_reduce) / certain
            self.log('Pseudo label loss', pseudo_loss)
            loss += self.alpha * pseudo_loss

        self.log('Prop unsup train', self.alpha)
        self.log("Train_loss", loss)

        return {'batch': batch, "loss": loss}


    def validation_step(self, batch, batch_idx):

        inputs = batch['image']
        labels = batch['mask']
        logits = self.forward(inputs)
        probas = torch.sigmoid(logits)
        preds = torch.argmax(probas, dim=1)

        stat_scores = torchmetrics.stat_scores(
            preds,
            labels,
            ignore_index=self.ignore_index if self.ignore_index >= 0 else None,
            mdmc_reduce='global',
            reduce='macro',
            num_classes=self.num_classes
        )
        
        loss1 = self.loss1(logits, labels)
        loss2 = self.loss2(logits, labels)
        loss = loss1 + loss2
        self.log('Val_CE', loss1)
        self.log('Val_Dice', loss2)
        self.log('Val_loss', loss)

        return {'batch': batch,
                'logits': logits.detach(),
                'stat_scores': stat_scores.detach(),
                'probas': probas.detach(),
                'preds': preds.detach()
                }
