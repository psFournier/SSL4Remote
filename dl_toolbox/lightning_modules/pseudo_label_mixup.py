import torch.nn as nn
import copy
from dl_toolbox.lightning_modules import Unet
import torch
from dl_toolbox.torch_datasets.utils import *
from dl_toolbox.augmentations import Mixup

class PseudoLabelling(Unet):

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

    def training_step(self, batch, batch_idx):

        sup_batch, semisup_batch = batch["sup"], batch["unsup"]

        res_dict = super().training_step(sup_batch, batch_idx)

        if self.trainer.current_epoch >= self.supervised_warmup:

            unsup_inputs = unsup_batch['image']
            with torch.no_grad():
                pseudo_logits = self.network_2(unsup_inputs)

            pseudo_probas, pseudo_preds = torch.max(pseudo_logits.softmax(dim=1), dim=1)
            pseudo_certain = pseudo_probas > 0.5
    
            lam = 0.8
            #Calculer les onehot pseudo labels + choix : garder les onehot pseudolabels confiants et les mélanger avec les sup 
            # ou tous les garder, mélanger avec les sup et filtrer après
            mixed_inputs = lam * sup_inputs + (1 - lam) * unsup_inputs
            mixed_targets = lam * sup_labels_onehot
            lam = np.random.beta(self.alpha, self.alpha)
            batchsize = input_batch.size()[0]
            idx = torch.randperm(batchsize)
            mixed_inputs = lam * input_batch + (1 - lam) * input_batch[idx, :]
            mixed_targets = lam * target_batch + (1 - lam) * target_batch[idx, :]
            all_inputs = torch.vstack([input_batch, mixed_inputs])
            all_targets = torch.vstack([target_batch, mixed_targets])
            idx = np.random.choice(2*batchsize, size=batchsize, replace=False)
            batch = (all_inputs[idx, :], all_targets[idx, :])

            unsup_outputs = self.network_1(unsup_inputs)


            
            loss_no_reduce = self.unsup_loss(
                student_outputs,
                pseudo_preds
            )
            pseudo_loss = torch.sum(pseudo_certain * loss_no_reduce) / torch.sum(pseudo_certain)
            self.log('Pseudo label loss', pseudo_loss)

        self.log('Prop unsup train', self.alpha)
        loss = sup_loss + self.alpha * pseudo_loss
        self.log("Train_loss", loss)

        return {'batch': sup_batch, 'logits': sup_logits.detach(), "loss": loss}
