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

        if trainer.current_epoch % 10 == 0 and trainer.global_step % 100 == 0:

            self.display_batch(trainer, pl_module, outputs, batch_idx=0,
                               prefix='Train')
            
    def display_batch(self, trainer, pl_module, outputs, batch_idx, prefix):

        img, mask = outputs['batch']['image'].cpu(), outputs['batch']['mask'].cpu()
        orig_img = outputs['batch']['orig_image'].cpu()
        logits = outputs['logits'].cpu()

        labels = torch.argmax(mask, dim=1) # torch tensor of dim B,H,W
        preds = torch.argmax(logits, dim=1)
        preds += int(not pl_module.train_with_void) # torch tensor of dim B,H,W
        labels_rgb = self.visu_fn(labels).transpose((0,3,1,2))
        preds_rgb = self.visu_fn(preds).transpose((0,3,1,2))
        mask_rgb = torch.from_numpy(labels_rgb).float()
        out_rgb = torch.from_numpy(preds_rgb).float()

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
            mask_grid = torchvision.utils.make_grid(mask_rgb[start:end, :, :, :], padding=10, normalize=True)
            out_grid = torchvision.utils.make_grid(out_rgb[start:end, :, :, :], padding=10, normalize=True)
            final_grid = torch.cat((orig_img_grid, img_grid, mask_grid, out_grid), dim=1)

            trainer.logger.experiment.add_image(f'Images/{prefix}_batch_{batch_idx}_part_{idx}', final_grid, global_step=trainer.global_step)
            break

    def on_validation_batch_end(
            self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ) -> None:
        """Called when the validation batch ends."""

        nb_val_batch = trainer.datamodule.nb_val_batch
        
        if trainer.current_epoch % 10 == 0 and batch_idx % (nb_val_batch // 5) == 0:
            self.display_batch(trainer, pl_module, outputs, batch_idx=batch_idx, prefix='Val')

