from torch.utils.data._utils.collate import default_collate


class Collate1():

    def __init__(self, transforms):

        self.transforms = transforms

    def __call__(self, batch, *args, **kwargs):

        windows = [elem['window'] for elem in batch]
        to_collate = [
            {
                'image': self.transforms(
                    image = elem['image'].transpose((1,2,0))
                )['image'],
            }
            for elem in batch
        ]
        batch = default_collate(to_collate)
        if 'mask' not in batch.keys():
            batch['mask'] = None
        batch['window'] = windows

        return batch
