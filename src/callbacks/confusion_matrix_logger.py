import pytorch_lightning as pl
from pytorch_lightning.metrics import ConfusionMatrix
import torch
from utils.utils import plot_confusion_matrix

class Conf_mat_logger(pl.Callback):
    
    def __init__(self, num_classes):
        
        self.conf_mat = ConfusionMatrix(
            num_classes=num_classes,
            normalize='true',
            compute_on_step=False
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx):

        image, target = batch
        image = image.to(pl_module.device)
        target = target.to(pl_module.device)
        predictions = pl_module(image)

        predictions = torch.argmax(predictions, dim=1)
        
        self.conf_mat(predictions.cpu(), target.cpu())

    def on_validation_epoch_end(self, trainer, pl_module):

        cm = self.conf_mat.compute()
        figure = plot_confusion_matrix(cm.numpy(), class_names=['0', '1'])
        trainer.logger.experiment.add_figure('Confusion matrix',
                                             figure,
                                             global_step=trainer.global_step)
