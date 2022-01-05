from argparse import ArgumentParser
from pytorch_lightning import Trainer
from lightning_modules import *
from torch_datasets import *
from torch.utils.data import DataLoader, RandomSampler
import torch
from utils import worker_init_function
import glob
from augmentations import NoOp
from functools import partial
from torch_collate import CollateDefault

modules = {
    'sup': SupervisedBaseline,
    'MT': MeanTeacher
}


def main():

    parser = ArgumentParser()

    parser.add_argument("--module", type=str, default="sup")
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--city", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--tile_size", type=int, default=128)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args_dict = vars(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt_path, map_location=device)

    pl_module = modules[args.module](**args_dict)
    pl_module.load_state_dict(ckpt['state_dict'])
    pl_module.eval()

    image_list = sorted(glob.glob(f'{args.data_dir}/{args.city}/test/*_x.tif'))
    label_list = sorted(glob.glob(f'{args.data_dir}/{args.city}/test/*_y.tif'))
    test_set_transforms = NoOp()

    test_set = MiniworldCityDs(
        city=args.city,
        images_paths=image_list,
        labels_paths=label_list,
        crop_size=args.tile_size,
        transforms=test_set_transforms,
    )

    test_sampler = RandomSampler(
        data_source=test_set,
        replacement=True,
        num_samples=10000
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        collate_fn=CollateDefault(),
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=worker_init_function
    )

    trainer = Trainer.from_argparse_args(
        args,
        logger=[],
    )

    trainer.test(model=pl_module, test_dataloaders=test_loader, verbose=False)

    print('Test metrics: ', pl_module.test_results)

if __name__ == "__main__":

    main()
