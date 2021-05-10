from argparse import ArgumentParser
import segmentation_models_pytorch as smp
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch
from pytorch_toolbelt.losses import DiceLoss
import torchmetrics.functional as metrics
import torchvision.transforms as T
import utils

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
        self.dice = utils.DiceLoss(mode="multilabel", log_loss=False, from_logits=True)
        # self.mixup_alpha = mixup_alpha
        # self.tta_augment = get_augment(tta_augment, always_apply=True)
        # self.train_augment = A.Compose(get_augment(augment))
        # self.batch_augment = get_batch_augment(batch_augment)
        # self.end_augment = A.Compose([
        #     A.Normalize(),
        #     ToTensorV2(transpose_mask=True)
        # ])

        self.im_aug = utils.Compose([
            utils.D4(),
            utils.ColorJitter(brightness=0.2)
        ])
        self.save_hyperparameters()

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
        parser.add_argument('--mixup_alpha', type=float, default=0.4)
        parser.add_argument('--augment', nargs='+', type=str, default=[])
        parser.add_argument('--batch_augment', nargs='+', type=str, default=[])
        parser.add_argument('--tta_augment', nargs='+', type=str, default=[])

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

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/Val_IoU": 0,
                                                   "hp/Swa_Val_IoU": 0,
                                                   "hp/TTA_Val_IoU":0})

    def training_step(self, batch, batch_idx):

        inputs, labels_onehot = batch
        labels = torch.argmax(labels_onehot, dim=1).long()

        inputs, labels_onehot = self.im_aug(inputs, labels_onehot)

        outputs = self.network(inputs)
        loss1 = self.bce(outputs, labels_onehot)
        loss2 = self.dice(outputs, labels_onehot)

        loss = loss1 + loss2

        self.log('Train_BCE', loss1)
        self.log('Train_Dice', loss2)
        self.log('Train_loss', loss)

        probas = outputs.softmax(dim=1)
        IoU = metrics.iou(probas,
                          labels,
                          reduction='none',
                          num_classes=self.num_classes)
        self.log('Train_IoU_0', IoU[0])
        self.log('Train_IoU_1', IoU[1])
        self.log('Train_IoU', torch.mean(IoU))

        return {"loss": loss}

    # def tta(self, batch, aug):



    def tta_and_collate(self, batch, tta_augment):

        tta_batches = batch.copy()
        for aug in tta_augment:
            for image, mask in batch:
                tta = aug(image=image, mask=mask)
                tta_batches.append((tta['image'], tta['mask']))

        end_batch = [self.end_augment(
            image=image,
            mask=mask
        ) for image, mask in tta_batches]

        end_batch = default_collate(
            [(elem["image"], elem["mask"]) for elem in end_batch]
        )

        return end_batch

    # def apply_im_aug(self, batch, aug):
    #
    #     aug_batch = [
    #         aug(
    #             image=image,
    #             mask=label
    #         )
    #         for image, label in batch
    #     ]
    #     batch = [(elem["image"], elem["mask"], torch.ones_like(elem["image"])) for elem in aug_batch]
    #
    #     return batch
    #
    # def apply_batch_aug(self, batch, aug):
    #
    #

    def augment_and_collate(self, batch, image_augment, batch_augment):

        image_augment_batch = [
            image_augment(
                image=image,
                mask=mask
            )
            for image, mask in batch
        ]
        batch = [(elem["image"], elem["mask"]) for elem in image_augment_batch]

        for aug in batch_augment:
            batch_augment_batch = aug(batch=batch)
            # not necessarily self.batch_size if drop_last=False in dataloader
            batch_size = len(batch)
            idx = np.random.choice(2*batch_size, size=batch_size, replace=False)
            batch = [(batch+batch_augment_batch)[i] for i in idx]

        end_batch = [self.end_augment(
            image=image,
            mask=mask
        ) for image, mask in batch]
        end_batch = [(elem["image"], elem["mask"]) for elem in end_batch]

        return default_collate(end_batch)

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
        self.log('hp/Val_IoU', torch.mean(IoU))

        precision, recall = metrics.precision_recall(probas,
                                                     val_labels,
                                                     mdmc_average='global',
                                                     average='none',
                                                     num_classes=self.num_classes)
        self.log('Val_precision_1', precision[1])
        self.log('Val_recall_1', recall[1])

        swa_callback = self.trainer.callbacks[1]
        if self.trainer.current_epoch < swa_callback._swa_epoch_start:
            swa_IoU = IoU
        else:
            swa_outputs = swa_callback._average_model.network(val_inputs)
            swa_probas = swa_outputs.softmax(dim=1)
            swa_IoU = metrics.iou(swa_probas,
                                  val_labels,
                                  reduction='none',
                                  num_classes=self.num_classes)
        self.log('Swa_Val_IoU_0', swa_IoU[0])
        self.log('Swa_Val_IoU_1', swa_IoU[1])
        self.log('hp/Swa_Val_IoU', torch.mean(swa_IoU))

        tta_batches = [outputs]
        for tta in self.tta:
            tta_batches.append(self.network(tta(val_inputs)))
        tta_probas = torch.stack(tta_batches).mean(dim=0).softmax(dim=1)

        #     tta_batches.append()
        # tta_factor = len(self.trainer.datamodule.tta_augment) + 1
        # input_chunks = torch.chunk(inputs, chunks=tta_factor, dim=0)
        # label_chunks = torch.chunk(labels, chunks=tta_factor, dim=0)
        # val_labels_one_hot = label_chunks[0]
        # val_inputs = input_chunks[0]
        #
        #
        # output_chunks = torch.stack(
        #     [outputs]+[self.network(chunk) for chunk in input_chunks[1:]]
        # )
        # tta_outputs = torch.mean(output_chunks, dim=0)
        # tta_probas = tta_outputs.softmax(dim=1)
        tta_IoU = metrics.iou(tta_probas,
                          val_labels,
                          reduction='none',
                          num_classes=self.num_classes)
        self.log('TTA_Val_IoU_0', tta_IoU[0])
        self.log('TTA_Val_IoU_1', tta_IoU[1])
        self.log('hp/TTA_Val_IoU', torch.mean(tta_IoU))