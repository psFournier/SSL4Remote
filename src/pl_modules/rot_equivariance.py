from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import rot90
from torch.optim import Adam


class RotEquivariance(pl.LightningModule):
    def __init__(self, network, unsup_loss_prop, scalar_metrics):

        super(RotEquivariance, self).__init__()

        self.network = network
        self.save_hyperparameters()
        self.train_metrics = scalar_metrics["train"]
        self.val_metrics = scalar_metrics["val"]

        # For the linear combination of loss
        self.unsup_loss_prop = unsup_loss_prop

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--unsup_loss_prop", type=float, default=0.5)

        return parser

    def forward(self, x):

        return self.network(x)

    def configure_optimizers(self):

        return Adam(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx):

        sup_data, unsup_data = batch["sup"], batch["unsup"]
        sup_train_inputs, sup_train_labels = sup_data
        outputs = self.network(sup_train_inputs)

        # Supervised learning
        sup_loss = F.cross_entropy(outputs, sup_train_labels)
        self.train_metrics(outputs.softmax(dim=1), sup_train_labels)

        # Enforcing rotation equivariance
        rotation_1, rotation_2 = np.random.choice([0, 1, 2, 3], size=2, replace=False)
        augmented_1 = rot90(unsup_data, k=rotation_1, dims=[2, 3])
        augmented_2 = rot90(unsup_data, k=rotation_2, dims=[2, 3])
        outputs_1 = self.network(augmented_1)
        outputs_2 = self.network(augmented_2)
        unaugmented_1 = rot90(outputs_1, k=-rotation_1, dims=[2, 3])
        unaugmented_2 = rot90(outputs_2, k=-rotation_2, dims=[2, 3])
        unsup_loss = F.mse_loss(unaugmented_1, unaugmented_2)

        total_loss = sup_loss + self.unsup_loss_prop * unsup_loss

        self.log("train_sup_loss", sup_loss)
        self.log_dict(self.train_metrics)
        self.log("train_unsup_loss", unsup_loss)

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):

        val_inputs, val_labels = batch
        outputs = self.network(val_inputs)
        sup_loss = F.cross_entropy(outputs, val_labels)
        softmax = outputs.softmax(dim=1)
        self.val_metrics(softmax, val_labels)

        self.log("val_sup_loss", sup_loss)
        self.log_dict(self.val_metrics)
