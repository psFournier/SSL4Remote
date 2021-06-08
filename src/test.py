import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pl_modules import *
from torch_datasets import *
from torch.utils.data import DataLoader
from utils import get_image_level_aug
import torch
from torch.utils.data._utils.collate import default_collate
from callbacks import *
from augmentations import Compose

# datasets = {
#     # 'christchurch': ChristchurchLabeled,
#     'austin': AustinLabeled,
#     'chicago': ChicagoLabeled,
#     'kitsap': KitsapLabeled,
#     'tyrol-w': TyrolwLabeled,
#     'vienna': ViennaLabeled,
#     'image': BaseCityImageLabeled
# }

def main():

    parser = ArgumentParser()

    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--label_path", type=str)
    parser.add_argument("--with_swa", action='store_true')
    parser.add_argument("--tta", nargs='+', type=str, default=[])
    parser.add_argument("--store_pred", action='store_true')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument('--img_aug', nargs='+', type=str, default=[])

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt_path)
    module = SupervisedBaseline()
    module.load_state_dict(ckpt['state_dict'])

    crop_size = 128
    if 'translate' in args.tta:
        crop_step = crop_size // 2
    else:
        crop_step = crop_size

    dataset = BaseCityImageLabeled(
        image_path=args.image_path,
        label_path=args.label_path,
        crop=crop_size,
        crop_step=crop_step,
        fixed_crop=True
    )

    def wif(id):
        uint64_seed = torch.initial_seed()
        np.random.seed([uint64_seed >> 32, uint64_seed & 0xffff_ffff])

    img_aug = Compose(get_image_level_aug(names=args.img_aug, p=1))
    def test_collate(batch):

        to_collate = [{k: v for k, v in elem.items() if k in ['image', 'mask']} for elem in batch]
        windows = [elem['window'] for elem in batch]
        batch = default_collate(to_collate)
        batch['window'] = windows
        if 'mask' not in batch.keys():
            batch['mask'] = None
        batch['image'], batch['mask'] = img_aug(img=batch['image'], label=batch['mask'])

        return batch

    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        collate_fn=test_collate,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=wif
    )

    if args.store_pred:
        output_name = '_'.join(os.path.splitext(args.image_path)[0].split('/'))[1:]+'_pred.tif'
        ckpt_dir = os.path.dirname(args.ckpt_path)
        save_output_path = os.path.join(ckpt_dir, output_name)
    else:
        save_output_path = None
    whole_image_pred = WholeImagePred(
        tta=args.tta,
        save_output_path=save_output_path,
    )

    callbacks = [whole_image_pred]
    if args.with_swa:
        module.network = module.swa_network
        # callbacks.append(CustomSwa(device=None))

    trainer = Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=[],
    )

    trainer.test(model=module, test_dataloaders=dataloader)

    print(module.test_results)
    print(whole_image_pred.tta_metrics)

if __name__ == "__main__":

    main()
