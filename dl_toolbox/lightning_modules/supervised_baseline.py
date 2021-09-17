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
        self.network = network
        self.learning_rate = learning_rate
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(mode="multilabel", log_loss=False, from_logits=True)
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
            milestones=[100, 300],
            gamma=0.3
        )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        inputs, labels_onehot = batch['image'], batch['mask']

        labels = torch.argmax(labels_onehot, dim=1).long()

        outputs = self.network(inputs)
        loss1 = self.bce(outputs, labels_onehot)
        loss2 = self.dice(outputs, labels_onehot)

        loss = loss1 + loss2

        self.log('Train_sup_BCE', loss1)
        self.log('Train_sup_Dice', loss2)
        self.log('Train_sup_loss', loss)

        probas = outputs.softmax(dim=1)
        IoU = metrics.iou(probas,
                          labels,
                          reduction='none',
                          num_classes=self.num_classes)
        for i in range(self.num_classes):
            self.log('Train_IoU_{}'.format(i), IoU[i])
        self.log('Train_IoU', torch.mean(IoU))

        return {'preds': outputs, "loss": loss}

    def validation_step(self, batch, batch_idx):

        val_inputs, val_labels_one_hot = batch['image'], batch['mask']
        val_labels = torch.argmax(val_labels_one_hot, dim=1).long()

        outputs = self.network(val_inputs)
        val_loss1 = self.bce(outputs, val_labels_one_hot)
        val_loss2 = self.dice(outputs, val_labels_one_hot)
        val_loss = val_loss1 + val_loss2

        self.log('Val_BCE', val_loss1)
        self.log('Val_Dice', val_loss2)
        self.log('Val_loss', val_loss)

        probas = outputs.softmax(dim=1)
        accuracy = metrics.accuracy(probas, val_labels)
        self.log('Val_acc', accuracy)

        IoU = metrics.iou(probas,
                          val_labels,
                          reduction='none',
                          num_classes=self.num_classes)
        for i in range(self.num_classes):
            self.log('Val_IoU_{}'.format(i), IoU[i])
        self.log('Val_IoU', torch.mean(IoU))

        self.log('epoch', self.trainer.current_epoch)

        return {'preds': outputs, 'IoU': IoU}

    def test_step(self, batch, batch_idx):

        test_inputs, test_labels_one_hot = batch['image'], batch['mask']
        test_labels = torch.argmax(test_labels_one_hot, dim=1).long()

        outputs = self.network(test_inputs)
        probas = outputs.softmax(dim=1)
        accuracy = metrics.accuracy(probas, test_labels)
        IoU = metrics.iou(probas,
                          test_labels,
                          reduction='none',
                          num_classes=self.num_classes)

        return {'preds': outputs, 'accuracy': accuracy, 'IoU': IoU}

    def test_epoch_end(self, outputs):

        avg_IoU = torch.stack([output['IoU'] for output in outputs]).mean(dim=0)
        self.test_results = {'IoU': avg_IoU.mean()}

        return avg_IoU


