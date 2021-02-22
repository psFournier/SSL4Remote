import pytorch_lightning as pl
from src.metrics import MAPMetric
import torch

class Map(pl.Callback):

    def __init__(self):
        self.map_metric = MAPMetric()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx):
        image, target = batch
        predictions = pl_module(image.to(pl_module.device).float())
        target = target.to(pl_module.device)

        # Get maximum of energy
        predictions = torch.argmax(predictions, dim=1)

        self.map_metric(predictions, target)

    def on_validation_epoch_end(self, trainer, pl_module):
        cm = self.map_metric.compute()
        print(cm.shape)
        for i, val in enumerate(cm):
            trainer.logger.experiment.add_scalar(
                'Mean Avg Precision, class {}'.format(i),
                val,
                global_step=trainer.global_step
            )

