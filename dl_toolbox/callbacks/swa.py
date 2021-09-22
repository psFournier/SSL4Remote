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
            swa_outputs = self._average_model.network(inputs.to(pl_module.device)).cpu()
            swa_preds = swa_outputs.argmax(dim=1) + 1
            swa_IoU = metrics.iou(swa_preds,
                                  labels,
                                  reduction='none',
                                  num_classes=pl_module.num_classes + 1,
                                  ignore_index=0)
            swa_accuracy = metrics.accuracy(swa_preds,
                                            labels,
                                            ignore_index=0)
        for i in range(pl_module.num_classes):
            class_name = trainer.datamodule.val_set.labels_desc[i+1][2]
            pl_module.log('Swa_Val_IoU_{}'.format(class_name), swa_IoU[i])
        pl_module.log('Swa_Val_IoU', torch.mean(swa_IoU))