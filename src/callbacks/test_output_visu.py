# Third-party libraries
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_warn


class TestOutputVisu(pl.Callback):
    """Generate images based on classifier predictions and log a batch to predefined logger.

    .. warning:: This callback supports only tensorboard right now

    """

    def __init__(self):
        super(TestOutputVisu, self).__init__()


    def on_validation_batch_end(
            self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ) -> None:
        """Called when the validation batch ends."""
        # Skip log if no valid logger or wrong batch
        if self._log_available is False or batch_idx != 0:
            return

        # Forward
        img, mask = batch

        # Segmentation maps
        labels = torch.argmax(mask, dim=1)
        preds = torch.argmax(outputs['preds'], dim=1)
        mask_rgb = trainer.datamodule.val_set.labels_to_colors(labels)
        out_rgb = trainer.datamodule.val_set.labels_to_colors(preds)

        # Number of grids to log depends on the batch size
        quotient, remainder = divmod(img.shape[0], self.NB_COL)
        nb_grids = quotient + int(remainder > 0)

        for idx in range(nb_grids):

            # Make grids (1st row = images, 2nd row = ground truth, 3rd row = predictions)
            start = self.NB_COL * idx
            if start + self.NB_COL <= img.shape[0]:
                end = start + self.NB_COL
            else:
                end = start + remainder

            img_grid = torchvision.utils.make_grid(img[start:end, :, :, :], padding=10, normalize=True)
            mask_grid = torchvision.utils.make_grid(mask_rgb[start:end, :, :, :], padding=10, normalize=True)
            out_grid = torchvision.utils.make_grid(out_rgb[start:end, :, :, :], padding=10, normalize=True)

            # Concat image, ground truth and predictions
            final_grid = torch.cat((img_grid, mask_grid, out_grid), dim=1)

            # Log to tensorboard
            if isinstance(trainer.logger, TensorBoardLogger):
                trainer.logger.experiment.add_image(f'Images/Batch {idx}', final_grid, global_step=trainer.global_step)
