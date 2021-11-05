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
                 crop_size,
                 pseudo_labelling=False,
                 consistency_training=False,
                 do_semisup=False,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.student_network = self.network
        self.teacher_network = copy.deepcopy(self.student_network)

        # Exponential moving average
        self.ema = ema
        self.supervised_warmup = supervised_warmup
        self.pseudo_labelling = pseudo_labelling
        self.consistency_training = consistency_training
        self.consistency_aug = aug.get_transforms(consistency_aug)
        self.crop_size = crop_size
        self.do_semisup = do_semisup

        # Unsupervised leaning loss
        self.unsup_loss = nn.MSELoss(reduction='none')
        self.save_hyperparameters()

        self.alpha = 0

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--ema", type=float, default=0.95)
        parser.add_argument("--do_semisup", action='store_true')
        parser.add_argument("--consistency_training", action='store_true')
        parser.add_argument('--consistency_aug', type=str)
        parser.add_argument('--pseudo_labelling', action='store_true')
        parser.add_argument("--supervised_warmup", type=int, default=20)

        return parser

    def on_train_epoch_start(self) -> None:

        if self.do_semisup:
            s = self.trainer.max_steps
            b = self.trainer.datamodule.sup_batch_size
            l = self.trainer.datamodule.epoch_len
            m = s * b / l # max number of epochs
            w = self.supervised_warmup
            e = self.trainer.current_epoch
            if e <= w:
                self.alpha = 0
            elif e <= 0.5 * m:
                self.alpha = ((e - w) / (0.5 * m - w)) * 0.9
            else:
                self.alpha = 0.9
        else:
            self.alpha = 0

    def update_teacher(self):

        # Update teacher model in place AFTER EACH BATCH?
        ema = min(1.0 - 1.0 / float(self.global_step + 1), self.ema)
        for param_t, param in zip(self.teacher_network.parameters(),
                                  self.student_network.parameters()):
            param_t.data.mul_(ema).add_(param.data, alpha=1 - ema)

    def compute_intersection(self, i1, i2, j1, j2):

        min_i = min(0, i2-i1)
        max_i = max(0, i2-i1)
        min_j = min(0, j2-j1)
        max_j = max(0, j2-j1)

        return min_i, max_i, min_j, max_j

    def training_step(self, batch, batch_idx):

        sup_batch, unsup_batch = batch["sup"], batch["unsup"]

        sup_inputs = sup_batch['image']
        sup_labels_onehot, sup_loss_mask = self.get_masked_labels(sup_batch['mask'])
        sup_logits = self.student_network(sup_inputs)
        sup_loss_1, sup_loss_2, sup_loss = self.compute_sup_loss(
            sup_logits,
            sup_labels_onehot,
            sup_loss_mask
        )
        # sup_preds = sup_logits.argmax(dim=1)
        # sup_labels = torch.argmax(sup_batch['mask'], dim=1).long()
        # iou, accuracy = self.compute_metrics(sup_preds, sup_labels)

        self.log('Train_sup_BCE', sup_loss_1)
        self.log('Train_sup_Dice', sup_loss_2)
        self.log('Train_sup_loss', sup_loss)
        # self.log_metrics(mode='Train', metrics={'iou': iou, 'acc': accuracy})

        unsup_loss = 0
        if self.trainer.current_epoch > self.supervised_warmup:

            w, h = unsup_batch['image'].shape[-1], unsup_batch['image'].shape[-2]
            i1 = torch.randint(0, h - self.crop_size + 1, size=(1, )).item()
            j1 = torch.randint(0, w - self.crop_size + 1, size=(1, )).item()
            unsup_inputs = unsup_batch['image'][..., i1:i1 + self.crop_size, j1:j1 + self.crop_size]
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

            if h != self.crop_size or w != self.crop_size:

                i2 = torch.randint(0, h - self.crop_size + 1, size=(1, )).item()
                j2 = torch.randint(0, w - self.crop_size + 1, size=(1, )).item()
                student_crop =  unsup_batch['image'][..., i2:i2 + self.crop_size, j2:j2 + self.crop_size]
                student_outputs = self.student_network(student_crop)

                min_i1, max_i1, min_j1, max_j1 = self.compute_intersection(i1, i2, j1, j2)
                intersection_teacher = teacher_outputs[..., max_i1:self.crop_size+min_i1, max_j1:self.crop_size+min_j1]
                min_i2, max_i2, min_j2, max_j2 = self.compute_intersection(i2, i1, j2, j1)
                intersection_student = student_outputs[..., max_i2:self.crop_size+min_i2, max_j2:self.crop_size+min_j2]

                translate_loss_no_reduce = self.unsup_loss(
                    intersection_student.softmax(dim=1),
                    intersection_teacher.softmax(dim=1)
                )
                translate_loss = torch.mean(translate_loss_no_reduce)
                self.log('Translation loss', translate_loss)
                unsup_loss += translate_loss

            if self.pseudo_labelling:
                student_outputs = self.student_network(unsup_inputs)
                loss_no_reduce = self.unsup_loss(
                    student_outputs.softmax(dim=1),
                    teacher_outputs.softmax(dim=1)
                )
                pseudo_loss = torch.mean(loss_no_reduce)
                self.log('Pseudo label loss', pseudo_loss)
                unsup_loss += pseudo_loss

        self.log("Train_unsup_loss", unsup_loss)

        self.update_teacher()
        self.log('Prop unsup train', self.alpha)
        loss = (1 - self.alpha) * sup_loss + self.alpha * unsup_loss

        return {'batch': sup_batch, 'logits': sup_logits, "loss": loss}