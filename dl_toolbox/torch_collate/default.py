from torch.utils.data._utils.collate import default_collate


class CollateDefault():

    def __call__(self, batch, *args, **kwargs):

        windows = [elem['window'] for elem in batch]
        to_collate = [
            {
                'image': elem['image'],
            }
            for elem in batch
        ]
        batch = default_collate(to_collate)
        if 'mask' not in batch.keys():
            batch['mask'] = None
        batch['window'] = windows

        return batch
