# Third-party libraries
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_warn
import numpy as np

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


# def binary_labels_to_rgb(labels):
#
#     labels = labels.cpu().numpy()
#     colors = np.zeros(shape=(labels.shape[0], labels.shape[1], labels.shape[2], 3), dtype=np.uint8)
#     idx = np.array(labels == 1)
#     colors[idx] = np.array([255,255,255])
#     res = np.transpose(colors, axes=(0, 3, 1, 2))
#     return torch.from_numpy(res).float()


class SegmentationImagesVisualisation(pl.Callback):
    """Generate images based on classifier predictions and log a batch to predefined logger.

    .. warning:: This callback supports only tensorboard right now

    """

    NB_COL: int = 4

    # def on_train_batch_end(
    #         self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    # ) -> None:
    #     """Called when the validation batch ends."""
    #
    #     if trainer.global_step % 1 == 0:
    #         # Forward
    #         img, mask = batch['image'], batch['mask']
    #         orig_img = batch['orig_image']
    #
    #         # Segmentation maps
    #         labels = torch.argmax(mask, dim=1)
    #         preds = torch.argmax(outputs['preds'], dim=1)
    #         mask_rgb = binary_labels_to_rgb(labels)
    #         out_rgb = binary_labels_to_rgb(preds)
    #
    #         # Number of grids to log depends on the batch size
    #         quotient, remainder = divmod(img.shape[0], self.NB_COL)
    #         nb_grids = quotient + int(remainder > 0)
    #
    #         for idx in range(nb_grids):
    #
    #             start = self.NB_COL * idx
    #             if start + self.NB_COL <= img.shape[0]:
    #                 end = start + self.NB_COL
    #             else:
    #                 end = start + remainder
    #
    #             img_grid = torchvision.utils.make_grid(img[start:end, :, :, :], padding=10, normalize=True)
    #             orig_img_grid = torchvision.utils.make_grid(orig_img[start:end, :, :, :], padding=10, normalize=True)
    #             mask_grid = torchvision.utils.make_grid(mask_rgb[start:end, :, :, :], padding=10, normalize=True)
    #             out_grid = torchvision.utils.make_grid(out_rgb[start:end, :, :, :], padding=10, normalize=True)
    #
    #             final_grid = torch.cat((orig_img_grid, img_grid, mask_grid, out_grid), dim=1)
    #
    #             trainer.logger.experiment.add_image(f'Images/Train_batch {idx}', final_grid, global_step=trainer.global_step)
    #
    #             break

    def on_validation_batch_end(
            self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ) -> None:
        """Called when the validation batch ends."""

        # Forward
        img, mask = batch['image'], batch['mask']
        orig_img = batch['orig_image']

        # Segmentation maps
        labels = torch.argmax(mask, dim=1)
        preds = torch.argmax(outputs['preds'], dim=1)
        mask_rgb = trainer.datamodule.val_set.label_to_rgb(labels.cpu().numpy())
        out_rgb = trainer.datamodule.val_set.label_to_rgb(preds.cpu().numpy())

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

            trainer.logger.experiment.add_image(f'Images/Val_batch {idx}', final_grid, global_step=trainer.global_step)

            break