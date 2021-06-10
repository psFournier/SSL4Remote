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
        else:
            val_inputs, val_labels_one_hot = batch
            val_labels = torch.argmax(val_labels_one_hot, dim=1).long()
            swa_outputs = self._average_model.network(val_inputs.to(pl_module.device)).cpu()
            swa_probas = swa_outputs.softmax(dim=1)
            swa_IoU = metrics.iou(swa_probas,
                                  val_labels,
                                  reduction='none',
                                  num_classes=pl_module.num_classes)
        pl_module.log('callback_Swa_Val_IoU_0', swa_IoU[0])
        pl_module.log('callback_Swa_Val_IoU_1', swa_IoU[1])
        pl_module.log('callback_Swa_Val_IoU', torch.mean(swa_IoU))