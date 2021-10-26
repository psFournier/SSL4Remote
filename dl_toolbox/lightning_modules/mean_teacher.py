import torch.nn as nn
import copy
from dl_toolbox.lightning_modules import SupervisedBaseline
import torch
import torchmetrics.functional as metrics
from dl_toolbox.augmentations import Cutmix

class MeanTeacher(SupervisedBaseline):

    def __init__(self,
                 ema,
                 consistency_aug,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        # The student network is self.network
        self.teacher_network = copy.deepcopy(self.network)

        # Exponential moving average
        self.ema = ema
        self.consistency_aug = Cutmix()

        # Unsupervised leaning loss
        self.mse = nn.MSELoss()

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--ema", type=float, default=0.95)
        parser.add_argument('--consistency_aug', nargs='+', type=str, default=[],
                            help='list of augmentations names to perform Consistency Regularization with on unlabeled data.')

        return parser

    def training_step(self, batch, batch_idx):

        sup_data, unsup_data = batch["sup"], batch["unsup"]
        sup_inputs, sup_labels_onehot = sup_data['image'], sup_data['mask']
        sup_labels = torch.argmax(sup_labels_onehot, dim=1).long()

        student_outputs = self.network(sup_inputs)
        sup_loss1 = self.bce(student_outputs, sup_labels_onehot)
        sup_loss2 = self.dice(student_outputs, sup_labels_onehot)

        sup_loss = sup_loss1 + sup_loss2

        self.log('Train_sup_BCE', sup_loss1)
        self.log('Train_sup_Dice', sup_loss2)
        self.log('Train_sup_loss', sup_loss)

        if self.trainer.current_epoch > 10:
            unsup_inputs, _ = unsup_data
            with torch.no_grad():
                unsup_targets = self.teacher_network(unsup_inputs)
            unsup_inputs, unsup_targets = self.consistency_aug(unsup_inputs, unsup_targets)
            unsup_outputs = self.network(unsup_inputs)
            unsup_loss = self.mse(unsup_outputs.softmax(dim=1), unsup_targets.softmax(dim=1))
        else:
            unsup_loss = 0

        self.log("Train_unsup_loss", unsup_loss)

        probas = student_outputs.softmax(dim=1)
        IoU = metrics.iou(probas,
                          sup_labels,
                          reduction='none',
                          num_classes=self.num_classes)
        self.log('Train_IoU_0', IoU[0])
        self.log('Train_IoU_1', IoU[1])
        self.log('Train_IoU', torch.mean(IoU))

        # Update teacher model in place AFTER EACH BATCH?
        ema = min(1.0 - 1.0 / float(self.global_step + 1), self.ema)
        for param_t, param in zip(self.teacher_network.parameters(),
                                  self.network.parameters()):
            param_t.data.mul_(ema).add_(param.data, alpha=1 - ema)

        loss = sup_loss + unsup_loss

        return {"loss": loss}