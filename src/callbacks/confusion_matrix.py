import pytorch_lightning as pl
from pytorch_lightning.metrics import ConfusionMatrix
import torch
from src.utils.utils import plot_confusion_matrix

class Conf_mat(pl.Callback):
    
    def __init__(self, num_classes):
        
        self.conf_mat = ConfusionMatrix(
            num_classes=num_classes,
            normalize='true',
            compute_on_step=False
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx):

        image, target = batch
        predictions = pl_module(image.to(pl_module.device).float())
        target = target.to(pl_module.device)

        # Get maximum of energy
        predictions = torch.argmax(predictions, dim=1)
        
        self.conf_mat(predictions, target)

    def on_validation_epoch_end(self, trainer, pl_module):

        cm = self.conf_mat.compute()
        figure = plot_confusion_matrix(cm.numpy(), class_names=['0', '1'])
        trainer.logger.experiment.add_figure('Confusion matrix', figure)
