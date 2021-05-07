from argparse import ArgumentParser
import segmentation_models_pytorch as smp
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch
from pytorch_toolbelt.losses import DiceLoss
import torchmetrics.functional as metrics
# import torch.functional as F

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
        self.learning_rate = learning_rate # Initial learning rate
        self.class_weights = class_weights if wce else torch.FloatTensor(
            [1.] * self.num_classes
        )
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(mode="multilabel", log_loss=False, from_logits=True)


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

        train_inputs, train_labels_one_hot = batch
        # train_labels = torch.argmax(train_labels_one_hot, dim=1).long()

        outputs = self.network(train_inputs)
        train_loss1 = self.bce(outputs, train_labels_one_hot)
        train_loss2 = self.dice(outputs, train_labels_one_hot)
        train_loss = train_loss1 + train_loss2

        self.log('Train_BCE', train_loss1)
        self.log('Train_Dice', train_loss2)
        self.log('Train_loss', train_loss)

        # probas = outputs.softmax(dim=1)
        # IoU = metrics.iou(probas,
        #                   train_labels,
        #                   reduction='none',
        #                   num_classes=self.num_classes)
        # self.log('Train_IoU_0', IoU[0])
        # self.log('Train_IoU_1', IoU[1])
        # self.log('Train_IoU', torch.mean(IoU))

        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):

        val_inputs, val_labels_one_hot = batch
        val_labels = torch.argmax(val_labels_one_hot, dim=1).long()

        outputs = self.network(val_inputs)
        val_loss1 = self.bce(outputs, val_labels_one_hot)
        val_loss2 = self.dice(outputs, val_labels_one_hot)
        val_loss = val_loss1 + val_loss2

        self.log('Val_BCE', val_loss1)
        self.log('Val_Dice', val_loss2)
        self.log('Val_loss', val_loss)

        swa_outputs = self.trainer.callbacks[1]._average_model.network(val_inputs)
        swa_probas = swa_outputs.softmax(dim=1)
        swa_IoU = metrics.iou(swa_probas,
                              val_labels,
                              reduction='none',
                              num_classes=self.num_classes)
        self.log('Swa_Val_IoU_0', swa_IoU[0])
        self.log('Swa_Val_IoU_1', swa_IoU[1])
        self.log('Swa_Val_IoU', torch.mean(swa_IoU))

        probas = outputs.softmax(dim=1)
        accuracy = metrics.accuracy(probas, val_labels)
        self.log('Val_acc', accuracy)

        # Could all these be made faster by making sure they rely on the same
        # computation for fp, fn, etc ?
        IoU = metrics.iou(probas,
                          val_labels,
                          reduction='none',
                          num_classes=self.num_classes)
        self.log('Val_IoU_0', IoU[0])
        self.log('Val_IoU_1', IoU[1])
        self.log('Val_IoU', torch.mean(IoU))

        precision, recall = metrics.precision_recall(probas,
                                                     val_labels,
                                                     mdmc_average='global',
                                                     average='none',
                                                     num_classes=self.num_classes)
        self.log('Val_precision_1', precision[1])
        self.log('Val_recall_1', recall[1])

        # figure = plot_confusion_matrix(cm.numpy(), class_names=["0", "1"])
        # trainer.logger.experiment.add_figure(
        #     "Confusion matrix", figure, global_step=trainer.global_step
        # )

    # def test_step(self, batch, batch_idx):
    #
    #     self.validation_step(batch, batch_idx)

