import torch.nn as nn
import copy
from dl_toolbox.lightning_modules import SupervisedBaseline
import torch
import dl_toolbox.augmentations as aug

class MeanTeacher(SupervisedBaseline):

    def __init__(self,
                 ema,
                 consistency_aug,
                 supervised_warmup,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.student_network = self.network
        self.teacher_network = copy.deepcopy(self.student_network)

        # Exponential moving average
        self.ema = ema
        self.supervised_warmup = supervised_warmup
        self.consistency_aug = aug.get_transforms(consistency_aug)

        # Unsupervised leaning loss
        self.unsup_loss = nn.MSELoss(reduction='none')

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--ema", type=float, default=0.95)
        parser.add_argument('--consistency_aug', type=str, default='no')
        parser.add_argument("--supervised_warmup", type=int, default=20)

        return parser

    def update_teacher(self):

        # Update teacher model in place AFTER EACH BATCH?
        ema = min(1.0 - 1.0 / float(self.global_step + 1), self.ema)
        for param_t, param in zip(self.teacher_network.parameters(),
                                  self.student_network.parameters()):
            param_t.data.mul_(ema).add_(param.data, alpha=1 - ema)

    def training_step(self, batch, batch_idx):

        sup_batch, unsup_batch = batch["sup"], batch["unsup"]

        sup_inputs = sup_batch['image']
        sup_labels_onehot, sup_loss_mask = self.get_masked_labels(sup_batch['mask'])
        sup_logits = self.student_network(sup_inputs)
        sup_loss_1, sup_loss_2, sup_loss = self.compute_sup_loss(sup_logits, sup_labels_onehot, sup_loss_mask)
        sup_preds = sup_logits.argmax(dim=1)
        sup_labels = torch.argmax(sup_batch['mask'], dim=1).long()
        iou, accuracy = self.compute_metrics(sup_preds, sup_labels)

        self.log('Train_sup_BCE', sup_loss_1)
        self.log('Train_sup_Dice', sup_loss_2)
        self.log('Train_sup_loss', sup_loss)
        self.log_metrics(mode='Train', metrics={'iou': iou, 'acc': accuracy})

        if self.trainer.current_epoch > self.supervised_warmup:

            unsup_inputs = unsup_batch['image']
            with torch.no_grad():
                teacher_outputs = self.teacher_network(unsup_inputs)
            consistency_inputs, consistency_targets = self.consistency_aug(unsup_inputs, teacher_outputs)
            student_outputs = self.student_network(consistency_inputs)
            unsup_loss_no_reduce = self.unsup_loss(student_outputs.softmax(dim=1), consistency_targets.softmax(dim=1))
            unsup_loss = torch.mean(unsup_loss_no_reduce)
        else:
            unsup_loss = 0

        self.log("Train_unsup_loss", unsup_loss)

        self.update_teacher()

        loss = sup_loss + unsup_loss

        return {'batch': sup_batch, 'logits': sup_logits, "loss": loss}