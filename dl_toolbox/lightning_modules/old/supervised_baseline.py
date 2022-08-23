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

class Unet(pl.LightningModule):

    def __init__(self,
                 encoder,
                 in_channels,
                 num_classes,
                 train_with_void=False,
                 eval_with_void=False,
                 pretrained=True,
                 initial_lr=0.05,
                 final_lr=0.001,
                 weight_decay=0,
                 lr_milestones=(0.5,0.9),
                 *args,
                 **kwargs):

        super().__init__()

        self.num_classes = num_classes
        network = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=num_classes if train_with_void else num_classes-1,
            decoder_use_batchnorm=True
        )
        self.train_with_void = train_with_void
        self.eval_with_void = eval_with_void
        self.ignore_index = None if (self.train_with_void and self.eval_with_void) else 0
        self.in_channels = in_channels
        self.network = network
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.wd = weight_decay
        self.lr_milestones = list(lr_milestones)
        # Reduction = none is necessary to compute properly the mean when using
        # a masked loss
        #self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        # The Dice loss is not a pixel-wise loss, so it seems that the masked
        # loss works properly by just masking preds and labels
        self.dice_loss = DiceLoss(mode="multilabel", log_loss=False, from_logits=True)
        #self.dice_loss = DiceLoss(mode="multiclass", log_loss=False, from_logits=True)
        self.save_hyperparameters()

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_classes", type=int)
        parser.add_argument("--train_with_void", action='store_true')
        parser.add_argument("--eval_with_void", action='store_true')
        parser.add_argument("--in_channels", type=int)
        parser.add_argument("--pretrained", action='store_true')
        parser.add_argument("--encoder", type=str)
        parser.add_argument("--initial_lr", type=float)
        parser.add_argument("--final_lr", type=float)
        parser.add_argument("--weight_decay", type=float)
        parser.add_argument("--lr_milestones", nargs='+', type=float)

        return parser

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/Val_loss": 0})

    def forward(self, x):
        
        return self.network(x)

    def configure_optimizers(self):

        self.optimizer = Adam(self.parameters(), lr=self.initial_lr)
        scheduler = MultiStepLR(
            self.optimizer,
            milestones=self.lr_milestones,
            gamma=0.1
        )

        #self.optimizer = SGD(
        #    self.parameters(),
        #    lr=self.initial_lr,
        #    momentum=0.9,
        #    weight_decay=self.wd
        #)

        #def lambda_lr(epoch):
        #    return ramp_down(epoch,
        #                     initial_lr=self.initial_lr,
        #                     final_lr=self.final_lr,
        #                     max_epochs=self.trainer.max_epochs,
        #                     milestones=self.lr_milestones)

        #scheduler = LambdaLR(
        #    self.optimizer,
        #    lr_lambda=lambda_lr
        #)

        return [self.optimizer], [scheduler]

#    def get_masked_labels(self, mask):
#
#        if not self.train_with_void:
#            # Granted that the first label is the void/unknown label, this extracts
#            # from labels the mask to use to ignore this class
#            labels_onehot = mask[:, 1:, :, :]
#            loss_mask = 1. - mask[:, [0], :, :]
#        else:
#            b, c, h, w = mask.shape
#            labels_onehot, loss_mask = mask, torch.ones(
#                b, 1, h, w, 
#                dtype=mask.dtype,
#                device=mask.device
#            )
#
#        return labels_onehot, loss_mask
#
#    def compute_sup_loss(self, logits, labels_onehot, loss_mask):
#
#        loss1_noreduce = self.ce_loss(logits, labels_onehot)
#        # The mean over all pixels is replaced with a mean over unmasked ones
#        loss1 = torch.sum(loss_mask * loss1_noreduce) / torch.sum(loss_mask)
#        loss2 = self.dice_loss(logits * loss_mask, labels_onehot * loss_mask)
#
#        return loss1, loss2, loss1 + loss2
#
#    def compute_metrics(self, preds, labels):
#
#        ignore_index = None if self.eval_with_void else 0
#
##        iou = torchmetrics.iou(
##            preds + int(not self.train_with_void),
##            labels,
##            reduction='none',
##            num_classes=self.num_classes, 
##            ignore_index=ignore_index
##        )
#
#        accuracy = torchmetrics.accuracy(
#            preds + int(not self.train_with_void),
#            labels,
#            ignore_index=ignore_index
#        )
#
#
#        f1_score = torchmetrics.f1_score(
#            preds + int(not self.train_with_void),
#            labels,
#            ignore_index=ignore_index,
#            mdmc_average='global'
#        )
#
#
#        return f1_score, accuracy
#
#    def log_metric_per_class(self, mode, metrics):
#
#        class_names = self.trainer.datamodule.class_names[
#            int(not self.eval_with_void):
#        ]
#        for metric_name, vals in metrics.items():
#            for val, class_name in zip(vals, class_names):
#                self.log(f'{mode}_{metric_name}_{class_name}', val)

    def training_step(self, batch, batch_idx):

        inputs = batch['image']
        labels_onehot = batch['mask']
        logits = self.network(inputs)

        if self.train_with_void:
            b,c,h,w = labels_onehot.shape
            loss_mask = torch.ones(
                    b,1,h,w,
                    dtype=labels_onehot.dtype,
                    device=labels_onehot.device)
        else:
            loss_mask = 1. - labels_onehot[:,[0],...]
            labels_onehot = labels_onehot[:,1:,...]

        loss1_noreduce = self.bce_loss(logits, labels_onehot)
        loss1 = torch.sum(loss_mask * loss1_noreduce) / torch.sum(loss_mask)
        loss2 = self.dice_loss(logits * loss_mask, labels_onehot * loss_mask)

        loss = loss1 + loss2

        self.log('Train_sup_BCE', loss1)
        self.log('Train_sup_Dice', loss2)
        self.log('Train_sup_loss', loss)

        return {'batch': batch, 'logits': logits.detach(), "loss": loss}

    def validation_step(self, batch, batch_idx):

        inputs = batch['image']
        labels_onehot = batch['mask']
        logits = self.network(inputs)

        if self.train_with_void:
            b,c,h,w = labels_onehot.shape
            loss_mask = torch.ones(
                    b,1,h,w,
                    dtype=labels_onehot.dtype,
                    device=labels_onehot.device)
        else:
            loss_mask = 1. - labels_onehot[:,[0],...]
            labels_onehot = labels_onehot[:,1:,...]
        
        loss1_noreduce = self.bce_loss(logits, labels_onehot)
        loss1 = torch.sum(loss_mask * loss1_noreduce) / torch.sum(loss_mask)
        loss2 = self.dice_loss(logits * loss_mask, labels_onehot * loss_mask)

        loss = loss1 + loss2

        self.log('Val_BCE', loss1)
        self.log('Val_Dice', loss2)
        self.log('hp/Val_loss', loss)

        preds = logits.argmax(dim=1) + int(not self.train_with_void)
        labels = torch.argmax(batch['mask'], dim=1).long()

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
        for i in range(self.num_classes):
            tp, fp, tn, fn, supp = class_stat_scores[i, :]
            f1 = tp / (tp + 0.5 * (fp + fn))
            iou = tp / (tp + fp + fn)
            self.log(f'Val_f1_{i}', f1)
            self.log(f'Val_iou_{i}', iou)
        
        tp, fp, tn, fn, supp = torch.sum(class_stat_scores, dim=0)
        accuracy = tp / supp
        self.log('Val_acc', accuracy)

        #f1 = cum_stat_scores[i, 
        #val_precision_0 = cum_stat_scores[0, 0] / (cum_stat_scores[0, 0] + cum_stat_scores[0, 1])
        #val_precision_1 = cum_stat_scores[1, 0] / (cum_stat_scores[1, 0] + cum_stat_scores[1, 1])
        #self.log('Val_precision_0', val_precision_0)
        #self.log('Val_precision_1', val_precision_1)
        #for i, val in metrics.items():
        #    for val, class_name in zip(vals, class_names):
        #        self.log(f'{mode}_{metric_name}_{class_name}', val)
