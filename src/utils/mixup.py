import numpy as np

def mixup_data(batch, alpha=1.0):

    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    idx = np.random.permutation(len(batch))
    perm = [batch[i] for i in idx]
    mixed_batch = [(lam * x1 + (1 - lam) * x2, lam * y1 + (1 - lam) * y2) for (x1, y1), (x2, y2) in zip(batch, perm)]

    return mixed_batch