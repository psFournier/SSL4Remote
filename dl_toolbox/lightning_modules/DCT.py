from argparse import ArgumentParser
import segmentation_models_pytorch as smp
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
import torch
import torchmetrics.functional as torchmetrics
from dl_toolbox.losses import DiceLoss
from copy import deepcopy
import torch.nn.functional as F

from dl_toolbox.lightning_modules.utils import *
from dl_toolbox.lightning_modules import BaseModule
import numpy as np

from utils import FGSM, VAT

class DCT(BaseModule):

    # DCT = Deep Co-Training

    def __init__(self,
                 encoder,
                 in_channels,
                 final_alpha,
                 alpha_milestones,
                 pseudo_threshold,
                 sup_eps,
                 unsup_eps,
                 num_power_iter,
                 pretrained=True,
                 initial_lr=0.05,
                 final_lr=0.001,
                 lr_milestones=(0.5,0.9),
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.network1 = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=self.num_classes,
            decoder_use_batchnorm=True
        )

        self.network2 = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=self.num_classes,
            decoder_use_batchnorm=True
        )

        self.in_channels = in_channels
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.lr_milestones = list(lr_milestones)

        self.loss1 = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index
        )
        self.loss2 = DiceLoss(
            mode="multiclass",
            log_loss=False,
            from_logits=True,
            ignore_index=self.ignore_index
        )

        self.unsup_loss = nn.CrossEntropyLoss(
            reduction='none'
        )

        # Adversarial attacks
        self.sup_eps = sup_esp
        self.unsup_eps = unsup_eps
        self.num_power_iter = num_power_iter

        self.fgsm1 = FGSM(model=self.network1, loss_fn=nn.BCEWithLogitsLoss(), epsilon=self.sup_eps)
        self.fgsm2 = FGSM(model=self.network2, loss_fn=nn.BCEWithLogitsLoss(), epsilon=self.sup_eps)
        self.vat1 = VAT(model=self.network1, xi=1e-6, eps=self.unsup_eps, num_power_iter=self.num_power_iter)
        self.vat2 = VAT(model=self.network2, xi=1e-6, eps=self.unsup_eps, num_power_iter=self.num_power_iter)

        self.final_alpha = final_alpha
        self.alpha_milestones = alpha_milestones
        self.alpha = 0.
        self.pseudo_threshold = pseudo_threshold
        self.save_hyperparameters()


    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)

        parser.add_argument("--unsup_eps", type=float, default=10.)
        parser.add_argument("--sup_eps", type=float, default=0.03)
        parser.add_argument("--num_power_iter", type=int, default=1)
        parser.add_argument("--final_alpha", type=float)
        parser.add_argument("--alpha_milestones", nargs=2, type=int)
        parser.add_argument("--pseudo_threshold", type=float)
        parser.add_argument("--in_channels", type=int)
        parser.add_argument("--pretrained", action='store_true')
        parser.add_argument("--encoder", type=str)
        parser.add_argument("--initial_lr", type=float)
        parser.add_argument("--final_lr", type=float)
        parser.add_argument("--lr_milestones", nargs='+', type=float)

        return parser


    def forward(self, x):

        logits1 = self.network1(x)
        logits2 = self.network2(x)
        
        return (logits1 + logits2) / 2

    def on_train_epoch_start(self):

        start = self.alpha_milestones[0]
        end = self.alpha_milestones[1]
        e = self.trainer.current_epoch
        alpha = self.final_alpha

        if e <= start:
            self.alpha = 0.
        elif e <= end:
            self.alpha = ((e - start) / (end - start)) * alpha
        else:
            self.alpha = alpha

    def training_step(self, batch, batch_idx):

        batch1, batch2, unsup_batch = batch["sup1"], batch["sup2"], batch["unsup"]

        inputs1 = batch1['image']
        labels1 = batch1['mask']

        inputs2 = batch2['image']
        labels2 = batch2['mask']

        logits1 = self.network1(inputs1)
        logits2 = self.network2(inputs2)
        loss1 = self.loss1(logits1, labels1) 
        loss1 += self.loss1(logits2, labels2)
        loss1 /= 2
        loss2 = self.loss2(logits1, labels1)
        loss2 += self.loss2(logits2, labels2)
        loss2 /= 2
        loss = loss1 + loss2

        self.log('Train_sup_CE', loss1)
        self.log('Train_sup_Dice', loss2)
        self.log('Train_sup_loss', loss)

        self.eval()
        
        # Making network2 robust on sup adv inputs of network1
        adv_sup_inputs1 = self.fgsm1(inputs1, labels1)
        adv_sup_outputs2 = self.network2(adv_sup_inputs1)
        fgsm_loss = self.loss1(adv_sup_output2, logits1.softmax(dim=1))

        # Making networ2 robust on unsup adv inputs of network1
        adv_unsup_inputs1 = self.vat1(unsup_inputs, unsup_outputs1)
        adv_unsup_outputs2 = self.network2(adv_unsup_inputs1)
        vat_loss = self.loss1(adv_unsup_output2, unsup_outputs1.softmax(dim=1))

        # Making network1 robust on sup adv inputs of network2
        adv_sup_inputs2 = self.fgsm2(inputs2, labels2)
        adv_sup_outputs1 = self.network1(adv_sup_inputs2)
        fgsm_loss += self.loss1(adv_sup_output1, logits2.softmax(dim=1))

        # Making networ1 robust on unsup adv inputs of network2
        adv_unsup_inputs2 = self.vat2(unsup_inputs, unsup_outputs2)
        adv_unsup_outputs1 = self.network1(adv_unsup_inputs2)
        vat_loss += self.loss1(adv_unsup_output1, unsup_outputs2.softmax(dim=1))

        diversity_loss = fgsm_loss + vat_loss

        self.log("Agreement loss", agreement_loss)
        self.log("Fgsm_loss", fgsm_loss)
        self.log("Vat_loss", vat_loss)

        self.train()

        if self.trainer.current_epoch > self.alpha_milestones[0]:
            
            unsup_inputs = unsup_batch['image']

            unsup_outputs1, unsup_outputs2 = self.network1(unsup_inputs), self.network2(unsup_inputs)

            mean_probs = unsup_outputs1.softmax(dim=1) + unsup_outputs2.softmax(dim=1)
            mean_probs /= 2

            agreement_loss = self.loss1(unsup_outputs1, mean_probs)
            agreement_loss += self.loss1(unsup_outputs2, mean_probs)

            # For diversity loss:
            # First, calculate adversarial samples


            loss += self.alpha * (agreement_loss + diversity_loss)
            
        self.log('Prop unsup train', self.alpha)
        self.log("Train_loss", loss)

        return {'batch': batch, "loss": loss}

    def validation_step(self, batch, batch_idx):

        res_dict = super().validation_step(batch, batch_idx)
        logits = res_dict['logits']
        labels = batch['mask']
        
        loss1 = self.loss1(logits, labels)
        loss2 = self.loss2(logits, labels)
        loss = loss1 + loss2
        self.log('Val_CE', loss1)
        self.log('Val_Dice', loss2)
        self.log('Val_loss', loss)

        probas = logits.softmax(dim=1)
        calib_error = torchmetrics.calibration_error(
            probas,
            labels
        )
        self.log('Calibration error', calib_error)

        return {**res_dict, **{'probas': probas.detach()}}

    def on_train_epoch_end(self):
        for param_group in self.optimizer.param_groups:
            self.log(f'learning_rate', param_group['lr'])
            break
