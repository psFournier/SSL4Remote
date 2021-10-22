from torch.utils.data._utils.collate import default_collate
import torch
import dl_toolbox.augmentations as aug

class CustomCollate():

    def __init__(self, batch_aug=None):

        self.batch_aug = aug.get_transforms(batch_aug)

    def __call__(self, batch, *args, **kwargs):

        windows = [elem['window'] for elem in batch]
        to_collate = [{k: v for k, v in elem.items() if k in ['image', 'orig_image', 'mask']} for elem in batch]
        batch = default_collate(to_collate)
        if 'mask' not in batch.keys():
            batch['mask'] = None
        batch['image'], batch['mask'] = self.batch_aug(batch['image'], batch['mask'])
        batch['window'] = windows
        batch['loss_mask'] = torch.ones_like(batch['mask'])

        return batch
