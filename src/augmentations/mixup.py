import numpy as np
import torch

class Mixup():

    def __init__(self, alpha=0.4):

        self.alpha = alpha

    def __call__(self, batch):

        lam = np.random.beta(self.alpha, self.alpha)
        inputs, targets = batch
        batchsize = inputs.size()[0]
        idx = torch.randperm(batchsize)
        mixed_inputs = lam * inputs + (1 - lam) * inputs[idx, :]
        mixed_targets = lam * targets + (1 - lam) * targets[idx, :]
        all_inputs = torch.vstack([inputs, mixed_inputs])
        all_targets = torch.vstack([targets, mixed_targets])
        idx = np.random.choice(2*batchsize, size=batchsize, replace=False)
        batch = (all_inputs[idx, :], all_targets[idx, :])

        return batch