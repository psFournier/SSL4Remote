import pytorch_lightning as pl
import torch


class ArrayValLogger(pl.Callback):
    def __init__(self, array_metric, name):
        self.array_metric = array_metric
        self.metric_name = name

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        image, target = batch
        predictions = pl_module(image.to(pl_module.device).float())
        target = target.to(pl_module.device)

        predictions = torch.argmax(predictions, dim=1)

        self.array_metric(predictions, target)

    def on_validation_epoch_end(self, trainer, pl_module):
        array_val = self.array_metric.compute()
        for i, val in enumerate(array_val):
            trainer.logger.experiment.add_scalar(
                self.metric_name + "_{}".format(i), val, global_step=trainer.global_step
            )
