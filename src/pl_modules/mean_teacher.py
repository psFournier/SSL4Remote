import torch.nn as nn
import copy
from pl_modules import SupervisedBaseline
import random
from torch import rot90, no_grad
import torchmetrics.functional as metrics

class MeanTeacher(SupervisedBaseline):

    def __init__(self,
                 ema,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        # The student network is self.network
        self.teacher_network = copy.deepcopy(self.network)

        # Exponential moving average
        self.ema = ema

        # Unsupervised leaning loss
        self.mse = nn.MSELoss()

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--ema", type=float, default=0.95)

        return parser

    def training_step(self, batch, batch_idx):

        sup_data, unsup_data = batch["sup"], batch["unsup"]
        sup_train_inputs, sup_train_labels = sup_data
        sup_train_labels = sup_train_labels.long()

        student_outputs = self.network(sup_train_inputs)
        sup_train_loss1 = self.ce(student_outputs, sup_train_labels)
        sup_train_loss2 = self.dice(student_outputs, sup_train_labels)
        sup_train_loss = sup_train_loss1 + sup_train_loss2

        augmentation = random.randint(0,4)
        augmented_inputs = rot90(unsup_data, k=augmentation, dims=[2, 3])
        student_outputs = self.network(augmented_inputs)
        with no_grad():
            teacher_outputs = self.teacher_network(unsup_data)
        teacher_outputs = rot90(teacher_outputs, k=augmentation, dims=[2, 3])
        unsup_train_loss = self.mse(student_outputs, teacher_outputs)

        train_loss = sup_train_loss + unsup_train_loss

        self.log('Cross entropy loss', sup_train_loss1)
        self.log('Dice loss', sup_train_loss2)
        self.log("sup_train_loss", sup_train_loss)
        self.log("unsup_train_loss", unsup_train_loss)
        probas = student_outputs.softmax(dim=1)
        IoU = metrics.iou(probas,
                          sup_train_labels,
                          reduction='none',
                          num_classes=self.num_classes)
        self.log('Train_IoU_0', IoU[0])
        self.log('Train_IoU_1', IoU[1])
        self.log('Train_IoU', IoU[0]+IoU[1])

        # Update teacher model in place AFTER EACH BATCH?
        ema = min(1.0 - 1.0 / float(self.global_step + 1), self.ema)
        for param_t, param in zip(self.teacher_network.parameters(),
                                  self.student_network.parameters()):
            param_t.data.mul_(ema).add_(param.data, alpha=1 - ema)

        return {"loss": train_loss}