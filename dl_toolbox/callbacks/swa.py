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

        if trainer.current_epoch <= self._swa_epoch_start:
            swa_iou = outputs['iou']
            swa_accuracy = outputs['accuracy']
        else:
            print(self._average_model.device)
            print(pl_module.device)
            print(self._device)
            inputs = batch['image']
            swa_logits = self._average_model.network(inputs.to(pl_module.device))
            swa_preds = swa_logits.argmax(dim=1)
            swa_labels = torch.argmax(batch['mask'], dim=1).long()
            swa_iou, swa_accuracy = pl_module.compute_metrics(swa_preds, swa_labels)
        
        pl_module.log_metric_per_class(mode='Val', metrics={'swa_iou': swa_iou})
        pl_module.log(f'Val_swa_acc', swa_accuracy)



