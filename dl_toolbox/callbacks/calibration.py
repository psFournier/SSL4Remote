import pytorch_lightning as pl
import torch
from torchmetrics import CalibrationError
import matplotlib.pyplot as plt
import numpy as np
import itertools
from torchmetrics.functional.classification.calibration_error import _binning_bucketize
from torchmetrics.utilities.data import dim_zero_cat

# Necessary for imshow to run on machines with no graphical interface.
plt.switch_backend("agg")

def plot_reliability_diagram(acc_bin, conf_bin):
    """

    """
    figure = plt.figure(figsize=(8, 8))
    plt.plot([0,1], [0,1], "k:", label="Perfectly calibrated")
    plt.plot(conf_bin, acc_bin, "s-", label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.title("Calibration curve")
    plt.tight_layout()
    plt.show()

    return figure


class CalibrationLogger(pl.Callback):

    def on_fit_start(self, trainer, pl_module):

        self.n_bins = 10
        self.calibration = CalibrationError(
            n_bins=self.n_bins
        )

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):

        _, labels = batch['image'], batch['mask']
        probas = outputs['probas']
        self.calibration(probas.cpu(), labels.cpu())

    def on_validation_epoch_end(self, trainer, pl_module):

        confidences = dim_zero_cat(self.calibration.confidences)
        accuracies = dim_zero_cat(self.calibration.accuracies)
        bin_boundaries = self.calibration.bin_boundaries.cpu()

        acc_bin = torch.zeros(len(bin_boundaries) - 1, device=confidences.device, dtype=confidences.dtype)
        conf_bin = torch.zeros(len(bin_boundaries) - 1, device=confidences.device, dtype=confidences.dtype)
        count_bin = torch.zeros(len(bin_boundaries) - 1, device=confidences.device, dtype=confidences.dtype)

        indices = torch.bucketize(confidences, bin_boundaries) - 1

        count_bin.scatter_add_(dim=0, index=indices, src=torch.ones_like(confidences))

        conf_bin.scatter_add_(dim=0, index=indices, src=confidences)
        conf_bin = torch.nan_to_num(conf_bin / count_bin)

        acc_bin.scatter_add_(dim=0, index=indices, src=accuracies)
        acc_bin = torch.nan_to_num(acc_bin / count_bin)

        figure = plot_reliability_diagram(
            acc_bin.numpy(),
            conf_bin.numpy(),
        )
        trainer.logger.experiment.add_figure(
            "Reliability diagram",
            figure,
            global_step=trainer.global_step
        )
        self.calibration.confidences = []
        self.calibration.accuracies = []

