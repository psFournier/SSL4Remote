from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import rot90, no_grad
from torch.optim import Adam
import copy

class MeanTeacher(pl.LightningModule):
    def __init__(self, network, unsup_loss_prop, ema, scalar_metrics):

        super(MeanTeacher, self).__init__()

        self.student_network = network
        self.teacher_network = copy.deepcopy(network)
        self.save_hyperparameters()
        self.train_metrics = scalar_metrics["train"]
        self.val_metrics = scalar_metrics["val"]

        # For the linear combination of loss
        self.unsup_loss_prop = unsup_loss_prop

        # Exponential moving average
        self.ema = ema

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--unsup_loss_prop", type=float, default=0.5)
        parser.add_argument("--ema", type=float, default=0.95)

        return parser

    def forward(self, x):

        # Should it be an inference on the teacher network instead ?
        return self.teacher_network(x)

    def configure_optimizers(self):

        return Adam(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx):

        sup_data, unsup_data = batch["sup"], batch["unsup"]
        sup_train_inputs, sup_train_labels = sup_data

        student_outputs = self.student_network(sup_train_inputs)

        # Supervised learning
        sup_loss = F.cross_entropy(student_outputs, sup_train_labels)
        self.train_metrics(student_outputs.softmax(dim=1), sup_train_labels)

        augmentation = np.random.randint(5)
        augmented_inputs = rot90(unsup_data, k=augmentation, dims=[2, 3])
        student_outputs = self.student_network(augmented_inputs)

        # Enforcing consistency on unlabeled data
        with no_grad():
            teacher_outputs = self.teacher_network(unsup_data)
        teacher_outputs = rot90(teacher_outputs, k=augmentation, dims=[2, 3])

        unsup_loss = F.mse_loss(student_outputs, teacher_outputs)

        total_loss = sup_loss + self.unsup_loss_prop * unsup_loss

        self.log("train_sup_loss", sup_loss)
        self.log_dict(self.train_metrics)
        self.log("train_unsup_loss", unsup_loss)

        # Update teacher model in place
        ema = min(1.0 - 1.0 / float(self.global_step + 1), self.ema)
        for param_t, param in zip(self.teacher_network.parameters(),
                                  self.student_network.parameters()):
            param_t.data.mul_(ema).add_(param.data, alpha=1 - ema)

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):

        val_inputs, val_labels = batch
        outputs = self.teacher_network(val_inputs)
        sup_loss = F.cross_entropy(outputs, val_labels)
        softmax = outputs.softmax(dim=1)
        self.val_metrics(softmax, val_labels)

        self.log("val_sup_loss", sup_loss)
        self.log_dict(self.val_metrics)
