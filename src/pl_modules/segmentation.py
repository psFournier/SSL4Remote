from torch import rot90
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam

class Semisup_segm(pl.LightningModule):

    def __init__(self,
                 network):

        super(Semisup_segm, self).__init__()
        self.network = network
        self.save_hyperparameters()

    def forward(self, x):

        return self.network(x)

    def configure_optimizers(self):

        return Adam(self.parameters(), lr=0.01)

    def accuracy(self, pred, label):

        return (pred.argmax(dim=1) == label).float().mean()

    def training_step(self, batch):

        sup_data, unsup_data = batch
        sup_train_inputs, sup_train_labels = sup_data
        outputs = self.network(sup_train_inputs)
        sup_loss = F.cross_entropy(outputs, sup_train_labels)
        acc = self.accuracy(outputs, sup_train_labels)

        rotation_1, rotation_2 = np.random.choice(
            [0, 1, 2, 3],
            size=2,
            replace=False
        )
        augmented_1 = rot90(unsup_data, k=rotation_1, dims=[2, 3])
        augmented_2 = rot90(unsup_data, k=rotation_2, dims=[2, 3])
        outputs_1 = self.network(augmented_1)
        outputs_2 = self.network(augmented_2)
        unaugmented_1 = rot90(outputs_1, k=-rotation_1, dims=[2, 3])
        unaugmented_2 = rot90(outputs_2, k=-rotation_2, dims=[2, 3])

        unsup_loss = F.mse_loss(
            unaugmented_1,
            unaugmented_2
        )

        total_loss = sup_loss + unsup_loss


        log_dict = {
            'sup_loss': sup_loss,
            'unsup_loss': unsup_loss,
            'acc': acc
        }

        return {'loss': total_loss, 'log': log_dict}

    def validation_step(self, batch):
        print(batch)
        val_inputs, val_labels = batch
        outputs = self.network(val_inputs)
        sup_loss = F.cross_entropy(outputs, val_labels)
        acc = self.accuracy(outputs, val_labels)
        log_dict = {'sup_loss': sup_loss,
                    'acc': acc}

        # return {'loss': sup_loss, 'log': log_dict}

class SegmentationModule(pl.LightningModule):

    def __init__(self, backbone):

        super(SegmentationModule, self).__init__()
        self.save_hyperparameters()
        self.backbone = backbone

    def forward(self, x):

        return self.backbone(x)

    def configure_optimizers(self):

        opt = Adam(self.parameters(), lr=0.01)
        return opt

    def accuracy(self, pred, label):

        return (pred.argmax(dim=1) == label).float().mean()

    def training_step(self, batch, batch_idx):

        x, y = batch
        output = self.forward(x)
        loss = F.cross_entropy(output,y)
        acc = self.accuracy(output, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        output = self.forward(x)
        loss = F.cross_entropy(output,y)
        acc = self.accuracy(output, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)