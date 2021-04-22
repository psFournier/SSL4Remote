from argparse import ArgumentParser
import segmentation_models_pytorch as smp
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch
from pytorch_toolbelt.losses import DiceLoss
import torchmetrics.functional as metrics

class SupervisedBaseline(pl.LightningModule):

    def __init__(self,
                 encoder,
                 pretrained,
                 in_channels,
                 num_classes,
                 inplaceBN,
                 learning_rate,
                 class_weights,
                 wce,
                 *args,
                 **kwargs):

        super().__init__()

        network = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
            decoder_use_batchnorm='inplace' if inplaceBN else True
        )
        self.num_classes = num_classes

        self.network = network
        self.save_hyperparameters()
        self.callbacks = []

        self.learning_rate = learning_rate # Initial learning rate

        self.class_weights = class_weights if wce else torch.FloatTensor(
            [1.] * num_classes
        )
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(mode="multiclass", log_loss=False)


    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument("--in_channels", type=int, default=3)
        parser.add_argument("--pretrained", action='store_true')
        parser.add_argument("--encoder", type=str, default='efficientnet-b0')
        parser.add_argument("--learning-rate", type=float, default=1e-3)
        parser.add_argument("--inplaceBN", action='store_true' )
        parser.add_argument("--wce", action='store_true')

        return parser

    def forward(self, x):

        return self.network(x)

    def configure_optimizers(self):

        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = MultiStepLR(
            optimizer,
            milestones=[
                int(self.trainer.max_epochs * 0.5),
                int(self.trainer.max_epochs * 0.7),
                int(self.trainer.max_epochs * 0.9)],
            gamma=0.3
        )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        train_inputs, train_labels = batch
        outputs = self.network(train_inputs)
        probas = outputs.softmax(dim=1)
        train_labels = train_labels.long()

        train_loss1 = self.ce(outputs, train_labels)
        train_loss2 = self.dice(outputs, train_labels)
        train_loss = train_loss1 + train_loss2
        self.log('Cross entropy loss', train_loss1)
        self.log('Dice loss', train_loss2)

        accuracy = metrics.accuracy(probas, train_labels)
        self.log('Train acc', accuracy)

        # Could all these be made faster by making sure they rely on the same
        # computation for fp, fn, etc ?
        IoU = metrics.iou(probas, train_labels, reduction='none')
        self.log('Train IoU class 0', IoU[0])
        self.log('Train IoU class 1', IoU[1])

        precision, recall = metrics.precision_recall(probas,
                                                     train_labels,
                                                     mdmc_average='global',
                                                     average='none',
                                                     num_classes=self.num_classes)
        self.log('Train precision class 1', precision[1])
        self.log('Train recall class 1', recall[1])

        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):

        val_inputs, val_labels = batch
        val_labels = val_labels.long()

        outputs = self.network(val_inputs)
        probas = outputs.softmax(dim=1)

        val_loss1 = self.ce(outputs, val_labels)
        val_loss2 = self.dice(outputs, val_labels)
        val_loss = val_loss1 + val_loss2
        self.log("val_sup_loss", val_loss)

        accuracy = metrics.accuracy(probas, val_labels)
        self.log('Train acc', accuracy)

        # Could all these be made faster by making sure they rely on the same
        # computation for fp, fn, etc ?
        IoU = metrics.iou(probas, val_labels, reduction='none')
        self.log('Train IoU class 0', IoU[0])
        self.log('Train IoU class 1', IoU[1])

        precision, recall = metrics.precision_recall(probas,
                                                     val_labels,
                                                     mdmc_average='global',
                                                     average='none',
                                                     num_classes=self.num_classes)
        self.log('Train precision class 1', precision[1])
        self.log('Train recall class 1', recall[1])


