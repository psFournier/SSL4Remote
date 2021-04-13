import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from torch.utils.data._utils.collate import default_collate
import torch

class RandomDataset(Dataset):
    def __getitem__(self, index):
        # return np.array([random.randint(0, 1000),
        #                  random.randint(0,1000),
        #                  random.randint(0, 1000)])
        return np.random.randint(0, 1000, 3)

    def __len__(self):
        return 4

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def wif(id):
    uint64_seed = torch.initial_seed()
    np.random.seed([uint64_seed >> 32, uint64_seed & 0xffff_ffff])

def collate(batch):

    # We apply transforms here because transforms are method-dependent
    # while the dataset class should be method independent.
    print('after collate: ', np.random.randint(0, 1000, 3))
    return default_collate(batch)

dataset = RandomDataset()
ds = DataLoader(dataset, batch_size=2, num_workers=2, shuffle=False,
                        collate_fn=collate,
                        worker_init_fn=wif
                )

for epoch in range(5):
    print("epoch {}".format(epoch))
    np.random.seed()
    for batch in ds:
        # print(batch)
        pass