import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

# Necessary for imshow to run on machines with no graphical interface.
plt.switch_backend("agg")

# Taken from https://www.tensorflow.org/tensorboard/image_summaries
def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


class ConfMatLogger(pl.Callback):
    def __init__(self, num_classes):

        self.conf_mat = ConfusionMatrix(
            num_classes=num_classes, normalize="true", compute_on_step=False
        )

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):

        image, target = batch
        image = image.to(pl_module.device)
        target = target.to(pl_module.device)
        predictions = pl_module(image)

        predictions = torch.argmax(predictions, dim=1)

        self.conf_mat(predictions.cpu(), target.cpu())

    def on_validation_epoch_end(self, trainer, pl_module):

        cm = self.conf_mat.compute()
        figure = plot_confusion_matrix(cm.numpy(), class_names=["0", "1"])
        trainer.logger.experiment.add_figure(
            "Confusion matrix", figure, global_step=trainer.global_step
        )
