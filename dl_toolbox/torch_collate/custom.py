from torch.utils.data._utils.collate import default_collate
import torch

class CustomCollate():

    def __init__(self, img_aug='', batch_aug=''):

        self.img_aug = img_aug
        self.batch_aug = batch_aug

    def __call__(self, batch, *args, **kwargs):

        windows = [elem['window'] for elem in batch]
        to_collate = [{k: v for k, v in elem.items() if k in ['image', 'orig_image', 'mask']} for elem in batch]
        batch = default_collate(to_collate)
        if 'mask' not in batch.keys():
            batch['mask'] = None
        # batch = self.img_aug(img=batch['image'], label=batch['mask'])
        # batch = self.batch_aug(*batch)
        # if len(batch) < 3:
        #     s = batch[0].size()
        #     batch = (*batch, torch.ones(size=(s[0], s[2], s[3])))
        batch['window'] = windows
        batch['loss_mask'] = torch.ones_like(batch['mask'])

        return batch
