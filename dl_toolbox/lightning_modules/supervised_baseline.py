from argparse import ArgumentParser
import segmentation_models_pytorch as smp
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch
import torchmetrics.functional as metrics
from dl_toolbox.losses import DiceLoss
from copy import deepcopy
import torch.nn.functional as F


class SupervisedBaseline(pl.LightningModule):

    def __init__(self,
                 encoder='efficientnet-b0',
                 pretrained=False,
                 in_channels=3,
                 num_classes=2,
                 learning_rate=0.001,
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
        # a masked loss ; we do not use BCEWithLogits because the masking must
        # be done after the computation of probabilities
        self.bce = nn.BCELoss(reduction='none')
        # The Dice loss is not a pixel-wise loss, so it seems that the masked
        # loss works properly by just masking preds and labels
        self.dice = DiceLoss(mode="multilabel", log_loss=False, from_logits=False)
        self.save_hyperparameters()

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument("--in_channels", type=int, default=3)
        parser.add_argument("--pretrained", action='store_true')
        parser.add_argument("--encoder", type=str, default='efficientnet-b0')
        parser.add_argument("--learning-rate", type=float, default=1e-3)

        return parser

    def forward(self, x):

        return self.network(x)

    def configure_optimizers(self):

        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = MultiStepLR(
            optimizer,
            milestones=[100],
            gamma=0.3
        )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        inputs, labels_onehot, loss_mask = batch['image'], batch['mask'], batch['loss_mask']

        if self.ignore_void:
            # Granted that the first label is the void/unknown label, this extracts
            # from labels the mask to use to ignore this class
            labels_onehot = labels_onehot[:, 1:, :, :]
            loss_mask = 1. - labels_onehot[:, [0], :, :]

        outputs = self.network(inputs)

        # Computing manually the activations just before masking and not after
        outputs = F.logsigmoid(outputs).exp()

        loss1_noreduce = self.bce(outputs * loss_mask, labels_onehot * loss_mask)
        # The mean over all pixels is replaced with a mean over unmasked ones
        loss1 = torch.sum(loss_mask * loss1_noreduce) / torch.sum(loss_mask)
        loss2 = self.dice(outputs * loss_mask, labels_onehot * loss_mask)

        loss = loss1 + loss2

        self.log('Train_sup_BCE', loss1)
        self.log('Train_sup_Dice', loss2)
        self.log('Train_sup_loss', loss)

        preds = outputs.argmax(dim=1)
        labels = torch.argmax(batch['mask'], dim=1).long()

        ignore_index = 0 if self.ignore_void else None
        IoU = metrics.iou(preds + int(self.ignore_void),
                          labels,
                          reduction='none',
                          num_classes=self.num_classes + int(self.ignore_void),
                          ignore_index=ignore_index)

        class_names = self.trainer.datamodule.class_names[int(self.ignore_void):]
        for i, name in enumerate(class_names):
            self.log('Train_IoU_{}'.format(name), IoU[i])
        self.log('Train_IoU', torch.mean(IoU))

        return {'batch': batch, 'preds': outputs, "loss": loss}

    def validation_step(self, batch, batch_idx):

        inputs, labels_onehot, loss_mask = batch['image'], batch['mask'], batch['loss_mask']

        if self.ignore_void:
            # Granted that the first label is the void/unknown label, this extracts
            # from labels the mask to use to ignore this class
            labels_onehot = labels_onehot[:, 1:, :, :]
            loss_mask = 1. - labels_onehot[:, [0], :, :]

        outputs = self.network(inputs)
        outputs = F.logsigmoid(outputs).exp()

        loss1_noreduce = self.bce(outputs * loss_mask, labels_onehot * loss_mask)
        loss1 = torch.sum(loss_mask * loss1_noreduce) / torch.sum(loss_mask)
        loss2 = self.dice(outputs * loss_mask, labels_onehot * loss_mask)
        loss = loss1 + loss2

        self.log('Val_BCE', loss1)
        self.log('Val_Dice', loss2)
        self.log('Val_loss', loss)

        preds = outputs.argmax(dim=1)
        labels = torch.argmax(batch['mask'], dim=1).long()

        ignore_index = 0 if self.ignore_void else None
        IoU = metrics.iou(preds + int(self.ignore_void),
                          labels,
                          reduction='none',
                          num_classes=self.num_classes + int(self.ignore_void),
                          ignore_index=ignore_index)
        accuracy = metrics.accuracy(preds + int(self.ignore_void),
                                    labels,
                                    ignore_index=ignore_index)

        class_names = self.trainer.datamodule.class_names[int(self.ignore_void):]
        for i, name in enumerate(class_names):
            self.log('Val_IoU_{}'.format(name), IoU[i])
        self.log('Val_IoU', torch.mean(IoU))

        self.log('Val_acc', accuracy)

        self.log('epoch', self.trainer.current_epoch)

        return {'batch': batch, 'preds': outputs, 'IoU': IoU, 'accuracy' : accuracy}

    # def test_step(self, batch, batch_idx):
    #
    #     inputs, labels_onehot = batch['image'], batch['mask']
    #
    #     outputs = self.network(inputs)
    #
    #     preds = outputs.argmax(dim=1) + 1
    #     labels = torch.argmax(labels_onehot, dim=1).long()
    #
    #     IoU = metrics.iou(preds,
    #                       labels,
    #                       reduction='none',
    #                       num_classes=self.num_classes+1,
    #                       ignore_index=0)
    #
    #     accuracy = metrics.accuracy(preds,
    #                                 labels,
    #                                 ignore_index=0)
    #
    #     return {'preds': outputs, 'accuracy': accuracy, 'IoU': IoU}
    #
    # def test_epoch_end(self, outputs):
    #
    #     avg_IoU = torch.stack([output['IoU'] for output in outputs]).mean(dim=0)
    #     self.test_results = {'IoU': avg_IoU.mean()}
    #
    #     return avg_IoU
    #
    @property
    def ignore_void(self):
        return self.trainer.datamodule.ignore_void


