from torch.utils.data._utils.collate import default_collate
import torch
from dl_toolbox.torch_datasets.utils import *

class CustomCollate():

    def __init__(self, batch_aug):

        self.batch_aug = get_transforms(batch_aug)

    def __call__(self, batch, *args, **kwargs):

        windows = [elem['window'] for elem in batch]
        keys_to_collate = ['image', 'orig_image', 'mask', 'orig_mask']
        to_collate = [{k: v for k, v in elem.items() if (k in keys_to_collate) and (v is not None)} for elem in batch]
        batch = default_collate(to_collate)
        if 'mask' not in batch.keys():
            batch['mask'] = None
        batch['image'], batch['mask'] = self.batch_aug(batch['image'], batch['mask'])
        batch['window'] = windows

        return batch
