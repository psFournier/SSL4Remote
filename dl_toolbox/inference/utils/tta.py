import numpy as np
from dl_toolbox.augmentations import D4

def apply_tta(
        tta,
        device,
        network,
        batch
):

    test_inputs = batch['image'].to(device)
    pred_list = []
    window_list = []

    if 'd4' in tta:
        d4 = D4()
        for t in d4.transforms:
            aug_inputs = t(test_inputs)[0]
            aug_pred = network(aug_inputs)
            pred = t(aug_pred)[0].cpu()
            pred_list += [np.squeeze(e, axis=0) for e in np.split(pred, pred.shape[0], axis=0)]
            window_list += batch['window']

    return pred_list, window_list