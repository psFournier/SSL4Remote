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


class Unet(pl.LightningModule):

    def __init__(self,
                 encoder='efficientnet-b0',
                 pretrained=False,
                 in_channels=3,
                 num_classes=2,
                 learning_rate=0.001,
                 lr_milestones=(100,),
                 *args,
                 **kwargs):

        super().__init__()

        network = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
            decoder_use_batchnorm=True
        )
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.network = network
        self.learning_rate = learning_rate
        # Reduction = none is necessary to compute properly the mean when using
        # a masked loss
        self.ce_loss = nn.BCEWithLogitsLoss(reduction='none')
        # The Dice loss is not a pixel-wise loss, so it seems that the masked
        # loss works properly by just masking preds and labels
        self.dice_loss = DiceLoss(mode="multilabel", log_loss=False, from_logits=True)
        self.save_hyperparameters()
        self.lr_milestones = lr_milestones

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument("--in_channels", type=int, default=3)
        parser.add_argument("--pretrained", action='store_true')
        parser.add_argument("--encoder", type=str, default='efficientnet-b0')
        parser.add_argument("--learning_rate", type=float, default=5e-2)
        parser.add_argument("--lr_milestones", nargs='+', type=int, default=[100])

        return parser

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/Val_loss": 0})

    def forward(self, x):
        
        return self.network(x)

    def configure_optimizers(self):

        optimizer = SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.9
        )

        def lambda_lr(epoch):

            #s = self.trainer.max_steps
            #b = self.trainer.datamodule.sup_batch_size
            #l = self.trainer.datamodule.epoch_len
            #m = s * b / l
            m = self.trainer.max_epochs
            if epoch < 0.4*m:
                return 1
            elif 0.4*m <= epoch <= 0.9*m:
                return 1 + ((epoch - 0.4*m) / (0.9*m - 0.4*m)) * (0.01 - 1)
            else:
                return 0.01

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda_lr
        )

        return [optimizer], [scheduler]

    def get_masked_labels(self, mask):

        if self.ignore_void:
            # Granted that the first label is the void/unknown label, this extracts
            # from labels the mask to use to ignore this class
            labels_onehot = mask[:, 1:, :, :]
            loss_mask = 1. - mask[:, [0], :, :]
        else:
            labels_onehot, loss_mask = mask, torch.ones_like(mask)

        return labels_onehot, loss_mask

    def compute_sup_loss(self, logits, labels_onehot, loss_mask):

        loss1_noreduce = self.ce_loss(logits, labels_onehot)
        # The mean over all pixels is replaced with a mean over unmasked ones
        loss1 = torch.sum(loss_mask * loss1_noreduce) / torch.sum(loss_mask)
        loss2 = self.dice_loss(logits * loss_mask, labels_onehot * loss_mask)

        return loss1, loss2, loss1 + 2*loss2

    def compute_metrics(self, preds, labels):

        ignore_index = 0 if self.ignore_void else None

        iou = torchmetrics.iou(
            preds + int(self.ignore_void),
            labels,
            reduction='none',
            num_classes=self.num_classes + int(self.ignore_void),
            ignore_index=ignore_index
        )

        accuracy = torchmetrics.accuracy(
            preds + int(self.ignore_void),
            labels,
            ignore_index=ignore_index
        )

        return iou, accuracy

    def log_metric_per_class(self, mode, metrics):

        class_names = self.trainer.datamodule.class_names[int(self.ignore_void):]
        for metric_name, vals in metrics.items():
            for val, class_name in zip(vals, class_names):
                self.log(f'{mode}_{metric_name}_{class_name}', val)

    def training_step(self, batch, batch_idx):

        inputs = batch['image']
        labels_onehot, loss_mask = self.get_masked_labels(batch['mask'])
        logits = self.network(inputs)
        loss1, loss2, loss = self.compute_sup_loss(logits, labels_onehot, loss_mask)
        # preds = logits.argmax(dim=1)
        # labels = torch.argmax(batch['mask'], dim=1).long()
        # iou, accuracy = self.compute_metrics(preds, labels)

        self.log('Train_sup_BCE', loss1)
        self.log('Train_sup_Dice', loss2)
        self.log('Train_sup_loss', loss)
        # self.log_metrics(mode='Train', metrics={'iou': iou, 'acc': accuracy})

        return {'batch': batch, 'logits': logits.detach(), "loss": loss}

    def validation_step(self, batch, batch_idx):

        inputs = batch['image']
        labels_onehot, loss_mask = self.get_masked_labels(batch['mask'])
        logits = self.network(inputs)
        loss1, loss2, loss = self.compute_sup_loss(logits, labels_onehot, loss_mask)
        preds = logits.argmax(dim=1)
        labels = torch.argmax(batch['mask'], dim=1).long()

        iou, accuracy = self.compute_metrics(preds, labels)

        self.log('Val_BCE', loss1)
        self.log('Val_Dice', loss2)
        self.log('hp/Val_loss', loss)
        self.log_metric_per_class(mode='Val', metrics={'iou': iou})
        self.log(f'Val_iou', torch.mean(iou))
        self.log(f'Val_acc', accuracy)
        # self.log('epoch', self.trainer.current_epoch)

        return {'batch': batch, 'logits': logits, 'iou': iou, 'accuracy' : accuracy}

    @property
    def ignore_void(self):
        return self.trainer.datamodule.ignore_void


