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

# datasets = {
#     # 'christchurch': ChristchurchLabeled,
#     'austin': AustinLabeled,
#     'chicago': ChicagoLabeled,
#     'kitsap': KitsapLabeled,
#     'tyrol-w': TyrolwLabeled,
#     'vienna': ViennaLabeled,
#     'image': BaseCityImageLabeled
# }

parser = ArgumentParser()
parser.add_argument("--ckpt_path", type=str)
parser.add_argument("--image_path", type=str)
parser.add_argument("--label_path", type=str)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()
args_dict = vars(args)

ckpt = torch.load(args.ckpt_path)
module = SupervisedBaseline()
module.load_state_dict(ckpt['state_dict'])

dataset = BaseCityImageLabeled(
    image_path=args.image_path,
    label_path=args.label_path,
    crop=128,
    crop_step=128,
    fixed_crop=True
)

def wif(id):
    uint64_seed = torch.initial_seed()
    np.random.seed([uint64_seed >> 32, uint64_seed & 0xffff_ffff])

def test_collate(batch):

    to_collate = [{k: v for k, v in elem.items() if k in ['image', 'mask']} for elem in batch]
    windows = [elem['window'] for elem in batch]
    batch = default_collate(to_collate)
    batch['window'] = windows
    if 'mask' not in batch.keys():
        batch['mask'] = None

    return batch

dataloader = DataLoader(
    dataset=dataset,
    shuffle=False,
    collate_fn=test_collate,
    batch_size=32,
    num_workers=8,
    pin_memory=True,
    worker_init_fn=wif
)


output_name = '_'.join(os.path.splitext(args.image_path)[0].split('/'))[1:]+'_pred.tif'
ckpt_dir = os.path.dirname(args.ckpt_path)
save_output_path = os.path.join(ckpt_dir, output_name)
tta = WholeImagePred(
    image_size=dataset.image_size,
    label_path=dataset.label_paths[0],
    colors_to_label=dataset.colors_to_labels,
    tta=None,
    swa=None,
    save_output_path=save_output_path
)

trainer = Trainer.from_argparse_args(
    args,
    callbacks=[tta],
    logger=[],
)

trainer.test(model=module, test_dataloaders=dataloader)

print(module.test_results)
print(tta.tta_metrics)