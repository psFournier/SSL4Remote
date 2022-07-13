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

class PL(pl.LightningModule):
    
    # PL = Pseudo Labelling

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

        self.network = smp.Unet(
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

            with torch.no_grad():
                pseudo_logits = self.network(unsup_inputs)
            
            pseudo_probas, pseudo_preds = torch.max(pseudo_logits.softmax(dim=1), dim=1)
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

        return {'batch': batch, 'logits': logits.detach(), "loss": loss}

    def validation_step(self, batch, batch_idx):

        inputs = batch['image']
        labels = batch['mask']
        logits = self.network(inputs)
        loss1 = self.loss1(logits, labels)
        loss2 = self.loss2(logits, labels)
        loss = loss1 + loss2
        self.log('Val_CE', loss1)
        self.log('Val_Dice', loss2)
        self.log('hp/Val_loss', loss)

        probas = logits.softmax(dim=1)

        # Checking calibration
        top_pred_probas, preds = torch.max(probas, dim=1)
        calib_certain = top_pred_probas > self.pseudo_threshold
        self.log('Pseudo label certainty', torch.mean(calib_certain.float()))
        calib_error = torchmetrics.calibration_error(
            probas,
            labels
        )
        self.log('Calibration error', calib_error)

        stat_scores = torchmetrics.stat_scores(
            preds,
            labels,
            ignore_index=self.ignore_index,
            mdmc_reduce='global',
            reduce='macro',
            num_classes=self.num_classes
        )

        return {
            'batch': batch, 
            'logits': logits.detach(), 
            'stat_scores': stat_scores.detach()
        }

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

