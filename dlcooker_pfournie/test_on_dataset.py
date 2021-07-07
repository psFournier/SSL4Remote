from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pl_modules import *
from torch_datasets import *
from torch.utils.data import DataLoader
import torch
from torch.utils.data._utils.collate import default_collate
from utils import worker_init_function
from pl_datamodules import Miniworld2

def main():

    parser = ArgumentParser()

    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--label_path", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--crop_size", type=int, default=128)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt_path, map_location=torch.device(device))
    module = SupervisedBaseline()
    module.load_state_dict(ckpt['state_dict'])

    labels_formatter = Miniworld2.colors_to_labels
    # dataset = OneLabeledImage(
    #     image_path=args.image_path,
    #     label_path=args.label_path,
    #     idxs=None,
    #     tile_size=(args.crop_size, args.crop_size),
    #     crop=args.crop_size,
    #     labels_formatter=labels_formatter
    # )

    data_dir = '/home/pierre/Documents/ONERA/ai4geo/miniworld_tif'
    dataset = MultipleImagesLabeled(
        images_paths=sorted(glob.glob(f'{data_dir}/vienna/test/*_x.tif')),
        labels_paths=sorted(glob.glob(f'{data_dir}/vienna/test/*_y.tif')),
        crop=args.crop_size,
        labels_formatter=labels_formatter
    )

    def collate(batch):

        to_collate = [{k: v for k, v in elem.items() if k in ['image', 'mask']} for elem in batch]
        batch = default_collate(to_collate)
        if 'mask' not in batch.keys():
            batch['mask'] = None

        return batch

    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        collate_fn=collate,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=worker_init_function
    )

    trainer = Trainer.from_argparse_args(
        args,
        logger=[],
    )

    trainer.test(model=module, test_dataloaders=dataloader, verbose=False)

    print('Test metrics: ', module.test_results)

if __name__ == "__main__":

    main()
