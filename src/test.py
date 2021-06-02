import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pl_modules import *
from torch_datasets import *
from torch.utils.data import DataLoader
from utils import get_image_level_aug
import torch

datasets = {
    # 'christchurch': ChristchurchLabeled,
    'austin': AustinLabeled,
    'chicago': ChicagoLabeled,
    'kitsap': KitsapLabeled,
    'tyrol-w': TyrolwLabeled,
    'vienna': ViennaLabeled
}

parser = ArgumentParser()
parser.add_argument("--ckpt_path", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument('--tta', nargs='+', type=str, default=[],
                    help='list of augmentation name to perform Test Time Augmentation (TTA) with')
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()
args_dict = vars(args)

ckpt_path = os.path.join(
    '/home/pierre/PycharmProjects/RemoteSensing/outputs/',
    args.ckpt_path
)
ckpt = torch.load(ckpt_path)
module = SupervisedBaseline()
module.load_state_dict(ckpt['state_dict'])


dataset = datasets[args.dataset](
    data_path=args.data_dir,
    idxs=[2,3],
    crop=128,
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

tta = get_image_level_aug(args.tta, p=1)

for batch, batch_idx in dataloader:

    inputs, labels_one_hot = batch
    labels = torch.argmax(labels_one_hot, dim=1).long()
    outputs = module.network(inputs)
    outputs_swa = module.swa_network(inputs)

    tta_batches = [outputs_swa]
    for aug in tta:
        tta_inputs, tta_targets = aug(inputs, labels_one_hot)
        tta_outputs = module.swa_network(tta_inputs)

        tta_batches.append()
    tta_probas = torch.stack(tta_batches).mean(dim=0).softmax(dim=1)

    tta_IoU = metrics.iou(tta_probas,
                          val_labels,
                          reduction='none',
                          num_classes=self.num_classes)
    self.log('TTA_Val_IoU_0', tta_IoU[0])
    self.log('TTA_Val_IoU_1', tta_IoU[1])
    self.log('hp/TTA_Val_IoU', torch.mean(tta_IoU))

trainer = Trainer.from_argparse_args(args)
trainer.validate(model=pl_module, test_dataloaders=dataloader, verbose=True)

