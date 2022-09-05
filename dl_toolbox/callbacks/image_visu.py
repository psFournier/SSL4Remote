# Third-party libraries
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_warn
import numpy as np


class SegmentationImagesVisualisation(pl.Callback):
    """Generate images based on classifier predictions and log a batch to predefined logger.

    .. warning:: This callback supports only tensorboard right now

    """

    NB_COL: int = 4

    def __init__(self, visu_fn, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.visu_fn = visu_fn 

    def on_train_batch_end(
            self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ) -> None:

        if trainer.current_epoch % 50 == 0 and batch_idx == 0:

            self.display_batch(
                trainer,
                outputs['batch'],
                prefix='Train'
            )

            if 'unsup_batch' in outputs.keys():
                self.display_batch(
                    trainer,
                    outputs['unsup_batch'],
                    prefix='UnsupTrain'
                )
            
    def display_batch(self, trainer, batch, prefix):

        img = batch['image'].cpu()
        orig_img = batch['orig_image'].cpu()
        logits = batch['logits'].cpu()

        probs = torch.softmax(logits, dim=1)
        top_probs, preds = torch.max(probs, dim=1)
        preds_rgb = self.visu_fn(preds).transpose((0,3,1,2))
        np_preds_rgb = torch.from_numpy(preds_rgb).float()

        if 'mask' in batch.keys(): 
            labels = batch['mask'].cpu()
            labels_rgb = self.visu_fn(labels).transpose((0,3,1,2))
            np_labels_rgb = torch.from_numpy(labels_rgb).float()

        # Number of grids to log depends on the batch size
        quotient, remainder = divmod(img.shape[0], self.NB_COL)
        nb_grids = quotient + int(remainder > 0)

        for idx in range(nb_grids):

            start = self.NB_COL * idx
            if start + self.NB_COL <= img.shape[0]:
                end = start + self.NB_COL
            else:
                end = start + remainder

            img_grid = torchvision.utils.make_grid(img[start:end, :, :, :], padding=10, normalize=True)
            orig_img_grid = torchvision.utils.make_grid(orig_img[start:end, :, :, :], padding=10, normalize=True)
            out_grid = torchvision.utils.make_grid(np_preds_rgb[start:end, :, :, :], padding=10, normalize=True)
            grids = [orig_img_grid, img_grid, out_grid]

            if 'mask' in batch.keys():
                mask_grid = torchvision.utils.make_grid(np_labels_rgb[start:end, :, :, :], padding=10, normalize=True)
                grids.append(mask_grid)

            final_grid = torch.cat(grids, dim=1)

            trainer.logger.experiment.add_image(f'Images/{prefix}_batch_art_{idx}', final_grid, global_step=trainer.global_step)
            break

    def on_validation_batch_end(
            self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ) -> None:
        """Called when the validation batch ends."""
 
        if trainer.current_epoch % 50 == 0 and batch_idx == 0:
            self.display_batch(trainer, outputs['batch'], prefix='Val')

