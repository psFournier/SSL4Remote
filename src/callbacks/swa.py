import pytorch_lightning as pl
import torch
import torchmetrics.functional as metrics

class CustomSwa(pl.callbacks.StochasticWeightAveraging):

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
            swa_outputs = self._average_model.network(val_inputs)
            swa_probas = swa_outputs.softmax(dim=1)
            swa_IoU = metrics.iou(swa_probas,
                                  val_labels,
                                  reduction='none',
                                  num_classes=pl_module.num_classes)
        pl_module.log('Swa_Val_IoU_0', swa_IoU[0])
        pl_module.log('Swa_Val_IoU_1', swa_IoU[1])
        pl_module.log('Swa_Val_IoU', torch.mean(swa_IoU))

    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'):

        if self._model_contains_batch_norm and trainer.current_epoch == self.swa_end + 1:
            # BatchNorm epoch update. Reset state
            trainer.accumulate_grad_batches = self._accumulate_grad_batches
            trainer.num_training_batches -= 1
            trainer.max_epochs -= 1
            self.reset_momenta()
        elif trainer.current_epoch == self.swa_end:
            # Last SWA epoch. Transfer weights from average model to pl_module
            self.transfer_weights(self._average_model.network, pl_module.swa_network)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        return {
            'average_model': self._average_model
        }

    def on_load_checkpoint(self, trainer, pl_module, callback_state):

        self._average_model = callback_state['average_model']

