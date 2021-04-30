from torch_datasets import ParisLabeled, AustinLabeled
import albumentations as A
from torch.utils.data import DataLoader, RandomSampler
import time
from argparse import ArgumentParser
import numpy as np
import torch

parser = ArgumentParser()
parser.add_argument("--city", type=str)
args = parser.parse_args()

cities = {
    'paris': ParisLabeled,
    'austin': AustinLabeled,
}

args.city = 'austin'

before = time.time()
ds = cities[args.city](
    data_path='/home/pierre/Documents/ONERA/ai4geo/miniminiworld/',
    idxs=list(range(1)),
    crop=1,
    augmentations=A.NoOp()
)

sampler = RandomSampler(
    data_source=ds,
    replacement=True,
    num_samples=1000
)

def wif(id):
    uint64_seed = torch.initial_seed()
    np.random.seed([uint64_seed >> 32, uint64_seed & 0xffff_ffff])

dl = DataLoader(
    ds,
    sampler=sampler,
    batch_size=32,
    num_workers=0,
    pin_memory=True,
    worker_init_fn=wif
)

i = 0
for image, label in dl:
    i+=1
    if i%10==0: print(i)
print("time elapsed: ", time.time()-before)