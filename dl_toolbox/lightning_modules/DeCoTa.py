import torch.nn as nn
import copy
from dl_toolbox.lightning_modules import BaseModel
import torch
from dl_toolbox.torch_datasets.utils import *
from dl_toolbox.augmentations import Mixup

class PseudoLabelling(BaseModel):

    def __init__(self,
                 ema,
                 supervised_warmup,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.network_1 = self.network
        self.network_2 = copy.deepcopy(self.network_1)

        # Exponential moving average
        self.ema = ema
        self.supervised_warmup = supervised_warmup
        
        # Unsupervised leaning loss
        self.unsup_loss = nn.CrossEntropyLoss(reduction='none')
        self.save_hyperparameters()

        self.alpha = 0

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--ema", type=float, default=0.95)
        parser.add_argument("--supervised_warmup", type=int, default=10)

        return parser

    def on_train_epoch_start(self) -> None:

        #s = self.trainer.max_steps
        #b = self.trainer.datamodule.sup_batch_size
        #l = self.trainer.datamodule.epoch_len
        #m = s * b / l # max number of epochs
        m = self.trainer.max_epochs           
        w = self.supervised_warmup
        e = self.trainer.current_epoch
        if e <= w:
            self.alpha = 0.
        elif e <= 0.7 * m:
            self.alpha = ((e - w) / (0.7 * m - w)) * 100.
        else:
            self.alpha = 100.

    def update_teacher(self):

        # Update teacher model in place AFTER EACH BATCH?
        ema = min(1.0 - 1.0 / float(self.global_step + 1), self.ema)
        for param_t, param in zip(self.teacher_network.parameters(),
                                  self.student_network.parameters()):
            param_t.data.mul_(ema).add_(param.data, alpha=1 - ema)

    def training_step(self, batch, batch_idx):

        sup_batch, unsup_batch = batch["sup"], batch["unsup"]

        sup_inputs = sup_batch['image']
        sup_labels_onehot = sup_batch['mask']
        sup_logits = self.student_network(sup_inputs)
        sup_loss_1 = self.bce_loss(sup_logits, sup_labels_onehot)
        sup_loss_2 = self.dice_loss(sup_logits, sup_labels_onehot)
        sup_loss = sup_loss_1 + sup_loss_2

        self.log('Train_sup_BCE', sup_loss_1)
        self.log('Train_sup_Dice', sup_loss_2)
        self.log('Train_sup_loss', sup_loss)

        unsup_loss = 0.
        if self.trainer.current_epoch >= self.supervised_warmup:

            unsup_inputs = unsup_batch['image']

            with torch.no_grad():
                pseudo_labels = self.network_2(unsup_inputs)



            with torch.no_grad():
                teacher_outputs = self.teacher_network(unsup_inputs)

            if self.consistency_training:
                consistency_inputs, consistency_targets = self.consistency_aug(
                    unsup_inputs,
                    teacher_outputs
                )
                consistency_outputs = self.student_network(consistency_inputs)
                loss_no_reduce = self.unsup_loss(
                    consistency_outputs.softmax(dim=1),
                    consistency_targets.softmax(dim=1)
                )
                consistency_loss = torch.mean(loss_no_reduce)
                self.log('Consistency loss', consistency_loss)

                unsup_loss += consistency_loss

            if self.pseudo_labelling:
                student_outputs = self.student_network(unsup_inputs)
                teacher_probs, teacher_preds = torch.max(teacher_outputs.softmax(dim=1), dim=1)
                loss_no_reduce = self.unsup_loss(
                    student_outputs,
                    teacher_preds
                )
                teacher_certain = teacher_probs > 0.5
                pseudo_loss = torch.sum(teacher_certain * loss_no_reduce) / torch.sum(teacher_certain)
                self.log('Pseudo label loss', pseudo_loss)
                unsup_loss += pseudo_loss


        self.update_teacher()
        self.log('Prop unsup train', self.alpha)
        loss = sup_loss + self.alpha * unsup_loss
        self.log("Train_loss", loss)

        return {'batch': sup_batch, 'logits': sup_logits.detach(), "loss": loss}
