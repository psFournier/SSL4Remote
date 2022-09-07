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

def compute_calibration_bins(bin_boundaries, labels, confs, preds, ignore_index):

    """
    All inputs must be flattened torch tensors.
    """
      
    accus = preds.eq(labels).float()

    indices = torch.bucketize(confs, bin_boundaries) - 1
    print(indices)
    print(torch.max(confs))
    n_bins = len(bin_boundaries) - 1

    count_bins = torch.zeros(n_bins, dtype=confs.dtype)
    count_bins.scatter_add_(dim=0, index=indices, src=torch.ones_like(confs))
    
    conf_bins = torch.zeros(n_bins, dtype=confs.dtype)
    conf_bins.scatter_add_(dim=0, index=indices, src=confs)
    conf_bins = torch.nan_to_num(conf_bins / count_bins)

    acc_bins = torch.zeros(n_bins, dtype=accus.dtype)
    acc_bins.scatter_add_(dim=0, index=indices, src=accus)
    acc_bins = torch.nan_to_num(acc_bins / count_bins)

    prop_bins = count_bins / count_bins.sum()

    return acc_bins, conf_bins, prop_bins
   

class CalibrationLogger(pl.Callback):

    def on_fit_start(self, trainer, pl_module):

        self.n_bins = 20
        self.bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
        print(self.bin_boundaries)
        self.acc_bins = torch.zeros(self.n_bins)
        self.conf_bins = torch.zeros(self.n_bins)
        self.prop_bins = torch.zeros(self.n_bins)
        self.nb_step = 0

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):

        labels = batch['mask'].cpu().flatten()
        confs = batch['confs'].cpu().flatten()
        preds = batch['preds'].cpu().flatten()
        acc_bins, conf_bins, prop_bins = compute_calibration_bins(
            self.bin_boundaries,
            labels,
            confs,
            preds,
            pl_module.ignore_index
        )
        self.acc_bins += acc_bins
        self.conf_bins += conf_bins
        self.prop_bins += prop_bins
        self.nb_step += 1

    def on_validation_epoch_end(self, trainer, pl_module):

        self.acc_bins /= self.nb_step
        self.conf_bins /= self.nb_step
        self.prop_bins /= self.nb_step

        figure = plot_reliability_diagram(
            self.acc_bins.numpy(),
            self.conf_bins.numpy(),
        )
        trainer.logger.experiment.add_figure(
            "Reliability diagram",
            figure,
            global_step=trainer.global_step
        )
        self.acc_bins = torch.zeros(self.n_bins)
        self.conf_bins = torch.zeros(self.n_bins)
        self.prop_bins = torch.zeros(self.n_bins)
        self.nb_step = 0

