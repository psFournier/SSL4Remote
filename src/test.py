import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pl_modules import *
from torch_datasets import *
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

datasets = {
    'chicago': ChicagoLabeled,
}

parser = ArgumentParser()
parser.add_argument("--ckpt_path", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--data_dir", type=str)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()
args_dict = vars(args)

ckpt_path = os.path.join(
    '/home/pierre/PycharmProjects/RemoteSensing/outputs/',
    args.ckpt_path
)
pl_module = SupervisedBaseline.load_from_checkpoint(ckpt_path)

augment = A.Compose([
    A.Normalize(),
    ToTensorV2(transpose_mask=False)
])

dataset = datasets[args.dataset](
    data_path=args.data_dir,
    idxs=[2,3],
    crop=128,
    augmentations=augment,
    fixed_crop=True
)

def wif(id):
    uint64_seed = torch.initial_seed()
    np.random.seed([uint64_seed >> 32, uint64_seed & 0xffff_ffff])

dataloader = DataLoader(
    dataset=dataset,
    shuffle=False,
    batch_size=16,
    num_workers=8,
    pin_memory=True,
    worker_init_fn=wif
)

trainer = Trainer.from_argparse_args(
    args
)
trainer.test(model=pl_module, test_dataloaders=dataloader, verbose=True)

