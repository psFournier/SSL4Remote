# Third-party libraries
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_warn

# def decode_segmentation_maps(tensor: torch.Tensor, pl_module: BaseModule = None) -> torch.Tensor:
#     """Decode model outputs to RGB segmentation maps.
#
#     Args:
#         tensor: Segmentation models predictions as a [N, W, H] tensor.
#         pl_module: Lightning module to decode classes.
#
#     Returns:
#         RGB segmentation maps
#
#     """
#     array = tensor.cpu().numpy()
#
#     # Create RGB channels
#     r = np.zeros_like(array).astype(np.uint8)
#     g = np.zeros_like(array).astype(np.uint8)
#     b = np.zeros_like(array).astype(np.uint8)
#
#     if pl_module is None:
#         raise ValueError('You must provide a Lightning Module.')
#
#     # Construct segmentation map
#     for label_idx in range(0, pl_module.datamodule.num_classes):
#         idx = array == label_idx
#         r[idx] = pl_module.datamodule.classes[label_idx].color[0]
#         g[idx] = pl_module.datamodule.classes[label_idx].color[1]
#         b[idx] = pl_module.datamodule.classes[label_idx].color[2]
#
#     rgb = np.stack([r, g, b], axis=1)
#     return torch.from_numpy(rgb).float()


class SegmentationImagesVisualisation(pl.Callback):
    """Generate images based on classifier predictions and log a batch to predefined logger.

    .. warning:: This callback supports only tensorboard right now

    """

    SUPPORTED_LOGGERS = (TensorBoardLogger,)
    NB_COL: int = 8

    def __init__(self):
        self._log_available = False

    def setup(self, trainer, pl_module, stage):
        """Called when fit or test begins."""
        available = True
        if not trainer.logger:
            rank_zero_warn("Cannot log images because Trainer has no logger.")
            available = False
        if not isinstance(trainer.logger, self.SUPPORTED_LOGGERS):
            rank_zero_warn(
                f"{self.__class__.__name__} does not support logging with {trainer.logger.__class__.__name__}."
                f" Supported loggers are: {', '.join(map(lambda x: str(x.__name__), self.SUPPORTED_LOGGERS))}"
            )
            available = False

        self._log_available = available

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
