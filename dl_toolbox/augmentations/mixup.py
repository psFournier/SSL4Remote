import numpy as np
import torch

class Mixup():

    def __init__(self, alpha=0.4):

        self.alpha = alpha

    def __call__(self, input_batch, target_batch):

        lam = np.random.beta(self.alpha, self.alpha)
        batchsize = input_batch.size()[0]
        idx = torch.randperm(batchsize)
        mixed_inputs = lam * input_batch + (1 - lam) * input_batch[idx, :]
        mixed_targets = lam * target_batch + (1 - lam) * target_batch[idx, :]
        all_inputs = torch.vstack([input_batch, mixed_inputs])
        all_targets = torch.vstack([target_batch, mixed_targets])
        idx = np.random.choice(2*batchsize, size=batchsize, replace=False)
        batch = (all_inputs[idx, :], all_targets[idx, :])

        return batch

class Mixup2():

    def __init__(self, alpha=0.4):

        self.alpha = alpha

    def __call__(
        self,
        input_batch_1,
        target_batch_1,
        input_batch_2,
        target_batch_2
    ):

        lam = np.random.beta(self.alpha, self.alpha)
        mixed_inputs = lam * input_batch_1 + (1 - lam) * input_batch_2
        mixed_targets = lam * target_batch_1 + (1 - lam) * target_batch_2
        batch = (mixed_inputs, mixed_targets)

        return batch

