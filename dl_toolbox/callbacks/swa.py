import pytorch_lightning as pl
import torch
import torchmetrics.functional as metrics
from pytorch_lightning.callbacks import StochasticWeightAveraging
from copy import deepcopy

class CustomSwa(StochasticWeightAveraging):

    def __init__(self, *args, **kwargs):

        super(CustomSwa, self).__init__(*args, **kwargs)

    def on_validation_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ):

        if trainer.current_epoch < self._swa_epoch_start:
            swa_IoU = outputs['IoU']
            swa_accuracy = outputs['accuracy']
        else:
            inputs, labels_onehot = batch['image'], batch['mask']
            labels = torch.argmax(labels_onehot, dim=1).long()
            swa_logits = self._average_model.network(inputs.to(pl_module.device))
            swa_preds = swa_logits.argmax(dim=1)

            ignore_index = 0 if pl_module.ignore_void else None
            swa_IoU = metrics.iou(
                swa_preds + int(pl_module.ignore_void),
                labels,
                reduction='none',
                num_classes=pl_module.num_classes + int(pl_module.ignore_void),
                ignore_index=ignore_index
            )
            swa_accuracy = metrics.accuracy(
                swa_preds + int(pl_module.ignore_void),
                labels,
                ignore_index=ignore_index
            )

        class_names = trainer.datamodule.class_names[int(pl_module.ignore_void):]
        for i, name in enumerate(class_names):
            pl_module.log('Swa_Val_IoU_{}'.format(name), swa_IoU[i])
        pl_module.log('Swa_Val_IoU', torch.mean(swa_IoU))
        pl_module.log('Swa_Val_acc', swa_accuracy)
