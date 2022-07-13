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

class CPS(pl.LightningModule):

    # CPS = Cross Pseudo Supervision

    def __init__(self,
                 encoder,
                 in_channels,
                 num_classes,
                 final_alpha,
                 alpha_milestones,
                 pseudo_threshold,
                 ignore_index=0,
                 pretrained=True,
                 initial_lr=0.05,
                 final_lr=0.001,
                 lr_milestones=(0.5,0.9),
                 *args,
                 **kwargs):

        super().__init__()


        self.network1 = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
            decoder_use_batchnorm=True
        )

        self.network2 = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
            decoder_use_batchnorm=True
        )

        self.num_classes = num_classes
        self.ignore_index = None if ignore_index < 0 else ignore_index
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

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--final_alpha", type=float)
        parser.add_argument("--alpha_milestones", nargs=2, type=int)
        parser.add_argument("--pseudo_threshold", type=float)
        parser.add_argument("--num_classes", type=int)
        parser.add_argument("--ignore_index", type=int)
        parser.add_argument("--in_channels", type=int)
        parser.add_argument("--pretrained", action='store_true')
        parser.add_argument("--encoder", type=str)
        parser.add_argument("--initial_lr", type=float)
        parser.add_argument("--final_lr", type=float)
        parser.add_argument("--lr_milestones", nargs='+', type=float)

        return parser

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/Val_loss": 0})

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

    def configure_optimizers(self):

        self.optimizer = Adam(self.parameters(), lr=self.initial_lr)
        scheduler = MultiStepLR(
            self.optimizer,
            milestones=self.lr_milestones,
            gamma=0.1
        )

        return [self.optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        batch, unsup_batch = batch["sup"], batch["unsup"]

        inputs = batch['image']
        labels = batch['mask']

        logits1 = self.network1(inputs)
        logits2 = self.network2(inputs)
        loss1 = self.loss1(logits1, labels) 
        loss1 += self.loss1(logits2, labels)
        loss1 /= 2
        loss2 = self.loss2(logits1, labels)
        loss2 += self.loss2(logits2, labels)
        loss2 /= 2
        loss = loss1 + loss2

        self.log('Train_sup_CE', loss1)
        self.log('Train_sup_Dice', loss2)
        self.log('Train_sup_loss', loss)

        if self.trainer.current_epoch > self.alpha_milestones[0]:

            unsup_inputs = unsup_batch['image']
            unsup_outputs_1 = self.network1(unsup_inputs)
            unsup_outputs_2 = self.network2(unsup_inputs)

            # Supervising network 1 with pseudolabels from network 2
            
            pseudo_probs_2 = unsup_outputs_2.detach().softmax(dim=1)
            top_probs_2, pseudo_preds_2 = torch.max(pseudo_probs_2, dim=1)
            loss_no_reduce_1 = self.unsup_loss(
                unsup_outputs_1,
                pseudo_preds_2
            )
            pseudo_certain_2 = top_probs_2 > self.pseudo_threshold
            pseudo_loss_1 = torch.sum(pseudo_certain_2 * loss_no_reduce_1) / torch.sum(pseudo_certain_2)

            # Supervising network 2 with pseudolabels from network 1

            pseudo_probs_1 = unsup_outputs_1.detach().softmax(dim=1)
            top_probs_1, pseudo_preds_1 = torch.max(pseudo_probs_1, dim=1)
            loss_no_reduce_2 = self.unsup_loss(
                unsup_outputs_2,
                pseudo_preds_1
            )
            pseudo_certain_1 = top_probs_1 > self.pseudo_threshold
            pseudo_loss_2 = torch.sum(pseudo_certain_1 * loss_no_reduce_2) / torch.sum(pseudo_certain_1)

            pseudo_loss = (pseudo_loss_1 + pseudo_loss_2) / 2

            self.log('Pseudo label loss', pseudo_loss)
            loss += self.alpha * pseudo_loss

        self.log('Prop unsup train', self.alpha)
        self.log("Train_loss", loss)

        return {'batch': batch, "loss": loss}

    def validation_step(self, batch, batch_idx):

        inputs = batch['image']
        labels = batch['mask']
        logits = self.network1(inputs)
        loss1 = self.loss1(logits, labels)
        loss2 = self.loss2(logits, labels)
        loss = loss1 + loss2
        self.log('Val_CE', loss1)
        self.log('Val_Dice', loss2)
        self.log('hp/Val_loss', loss)

        preds = logits.argmax(dim=1)
        stat_scores = torchmetrics.stat_scores(
            preds,
            labels,
            ignore_index=self.ignore_index,
            mdmc_reduce='global',
            reduce='macro',
            num_classes=self.num_classes
        )

        return {'batch': batch, 'logits': logits.detach(), 'stat_scores': stat_scores.detach()}

    def on_train_epoch_end(self):
        for param_group in self.optimizer.param_groups:
            self.log(f'learning_rate', param_group['lr'])
            break

    def validation_epoch_end(self, outs):
        
        stat_scores = [out['stat_scores'] for out in outs]

        class_stat_scores = torch.sum(torch.stack(stat_scores), dim=0)
        f1_sum = 0
        tp_sum = 0
        supp_sum = 0
        nc = 0
        for i in range(self.num_classes):
            if i != self.ignore_index:
                tp, fp, tn, fn, supp = class_stat_scores[i, :]
                if supp > 0:
                    nc += 1
                    f1 = tp / (tp + 0.5 * (fp + fn))
                    self.log(f'Val_f1_{i}', f1)
                    f1_sum += f1
                    tp_sum += tp
                    supp_sum += supp
        
        self.log('Val_acc', tp_sum / supp_sum)
        self.log('Val_f1', f1_sum / nc) 

