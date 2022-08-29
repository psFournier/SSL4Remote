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
from dl_toolbox.networks import *



class Smp_Unet_BCE_multilabel_2(BaseModule):

    # BCE_multilabel = Binary Cross Entropy for multilabel prediction

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
            classes=self.num_classes - 1,
            decoder_use_batchnorm=True
        )
        self.in_channels = in_channels
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.lr_milestones = list(lr_milestones)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.onehot = TorchOneHot(range(self.num_classes))
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
        onehot_labels = self.onehot(labels).float() # B,C,H,W

        final_labels = onehot_labels[:, 1:, ...] # B,C-1,H,W
        
        mask = torch.ones_like(
            final_labels,
            dtype=onehot_labels.dtype,
            device=onehot_labels.device
        )
        
        logits = self.network(inputs) # B,C-1,H,W
        bce = self.bce(logits, final_labels)
        bce = torch.sum(mask * bce) / torch.sum(mask)
        dice = self.dice(logits * mask, final_labels * mask)
        loss = bce + dice
        
        self.log('Train_sup_BCE', bce)
        self.log('Train_sup_Dice', dice)
        self.log('Train_sup_loss', loss)

        return {'batch': batch, 'logits': logits.detach(), "loss": loss}

    def validation_step(self, batch, batch_idx):

        inputs = batch['image']
        labels = batch['mask'] # B,H,W
        onehot_labels = self.onehot(labels).float() # B,C,H,W
        
        final_labels = onehot_labels[:, 1:, ...]
        
        mask = torch.ones_like(
            final_labels,
            dtype=onehot_labels.dtype,
            device=onehot_labels.device
        )
        
        logits = self.forward(inputs)
        probas = torch.sigmoid(logits) # B,C-1,H,W
        #preds = torch.argmax(probas, dim=1)

        stat_scores = torchmetrics.stat_scores(
            probas,
            labels,
            ignore_index=None,
            mdmc_reduce='global',
            reduce='macro',
            threshold=0.5,
            top_k=1,
            num_classes=self.num_classes-1
        )
        
        bce = self.bce(logits, final_labels)
        bce = torch.sum(mask * bce) / torch.sum(mask)
        dice = self.dice(logits * mask, final_labels * mask)
        loss = bce + dice
        
        self.log('Val_BCE', bce)
        self.log('Val_Dice', dice)
        self.log('Val_loss', loss)

        full_probas = torch.cat(
            [torch.zeros_like(
                labels,
                device=probas.device,
                dtype=probas.dtype
            ).unsqueeze(dim=1),
            probas],
            dim=1
        )

        return {'batch': batch,
                'logits': logits.detach(),
                'stat_scores': stat_scores.detach(),
                'probas': full_probas.detach()
                }
    
    def validation_epoch_end(self, outs):
        
        stat_scores = [out['stat_scores'] for out in outs]

        class_stat_scores = torch.sum(torch.stack(stat_scores), dim=0)
        f1_sum = 0
        tp_sum = 0
        supp_sum = 0
        nc = 0
        # ignore_index = 0
        for i in range(1, self.num_classes):
            tp, fp, tn, fn, supp = class_stat_scores[i-1, :]
            if supp > 0:
                nc += 1
                f1 = tp / (tp + 0.5 * (fp + fn))
                self.log(f'Val_f1_{i}', f1)
                f1_sum += f1
                tp_sum += tp
                supp_sum += supp
        
        self.log('Val_acc', tp_sum / supp_sum)
        self.log('Val_f1', f1_sum / nc) 

