import numpy as np
from augmentations import image_level_aug

anti_t_dict = {
    'hflip': 'hflip',
    'vflip': 'vflip',
    'd1flip': 'd1flip',
    'd2flip': 'd2flip',
    'rot90': 'rot270',
    'rot180': 'rot180',
    'rot270': 'rot90'
}

def apply_tta(
        tta,
        device,
        module,
        batch
):

    test_inputs = batch['image'].to(device)
    pred_list = []
    window_list = []

    for t_name in tta:
        print(t_name)
        print(test_inputs.shape)
        print(test_inputs.device)
        t = image_level_aug[t_name](p=1)
        aug_inputs = t(test_inputs)[0]
        print(aug_inputs.shape)
        print(aug_inputs.device)
        pred = module.forward(aug_inputs).cpu()
        if t_name in anti_t_dict:
            anti_t = image_level_aug[anti_t_dict[t_name]](p=1)
            pred = anti_t(pred)[0]
        pred_list += [np.squeeze(e, axis=0) for e in np.split(pred, pred.shape[0], axis=0)]
        window_list += batch['window']

    # if 'd4' in tta:
    #     d4 = D4()
    #     for t in d4.transforms:
    #         aug_inputs = t(test_inputs)[0]
    #         aug_pred = network(aug_inputs)
    #         pred = t(aug_pred)[0].cpu()
    #         pred_list += [np.squeeze(e, axis=0) for e in np.split(pred, pred.shape[0], axis=0)]
    #         window_list += batch['window']

    return pred_list, window_list
