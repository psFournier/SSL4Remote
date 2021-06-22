from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pl_modules import *
from torch_datasets import *
from torch.utils.data import DataLoader
import torch
from torch.utils.data._utils.collate import default_collate
from utils import worker_init_function
from pl_datamodules import Miniworld2
import imagesize
import torchmetrics.functional as M
from augmentations import *

def _apply_tta(
        tta,
        device,
        network,
        batch
):

    test_inputs = batch['image'].to(device)
    pred_list = []
    window_list = []

    if 'd4' in tta:
        d4 = D4()
        for t in d4.transforms:
            aug_inputs = t(test_inputs)[0]
            aug_pred = network(aug_inputs)
            pred = t(aug_pred)[0].cpu()
            pred_list += [np.squeeze(e, axis=0) for e in np.split(pred, pred.shape[0], axis=0)]
            window_list += batch['window']

    return pred_list, window_list


def main():

    parser = ArgumentParser()

    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--label_path", type=str)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--tta", nargs='+', type=str, default=[])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--crop_step", type=int, default=64)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt_path, map_location=torch.device(device))
    module = SupervisedBaseline()
    module.load_state_dict(ckpt['state_dict'])

    labels_formatter = Miniworld2.colors_to_labels
    dataset = OneLabeledImage(
        image_path=args.image_path,
        label_path=args.label_path,
        idxs=None,
        tile_size=(args.crop_size, args.crop_size),
        tile_step=(args.crop_step, args.crop_step),
        crop=args.crop_size,
        labels_formatter=labels_formatter
    )

    def collate(batch):

        windows = [elem['window'] for elem in batch]
        to_collate = [{k: v for k, v in elem.items() if k in ['image', 'mask']} for elem in batch]
        batch = default_collate(to_collate)
        if 'mask' not in batch.keys():
            batch['mask'] = None
        batch['window'] = windows

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

    height, width = imagesize.get(args.image_path)
    pred_sum = torch.zeros(size=(module.num_classes, height, width))
    metrics = {}
    module.eval()

    for batch in dataloader:

        inputs, _, windows = batch['image'], batch['mask'], batch['window']

        with torch.no_grad():
            outputs = module.network(inputs.to(device)).cpu()

        split_pred = np.split(outputs, outputs.shape[0], axis=0)
        pred_list = [np.squeeze(e, axis=0) for e in split_pred]
        window_list = windows[:]

        if args.tta:
            tta_preds, tta_windows = _apply_tta(args.tta, device, module.network, batch)
            pred_list += tta_preds
            window_list += tta_windows

        for pred, window in zip(pred_list, window_list):
            pred_sum[:, window.row_off:window.row_off + window.width, window.col_off:window.col_off + window.height] += pred

    avg_probs = pred_sum.softmax(dim=0)
    labels = rasterio.open(args.label_path).read(out_dtype=np.float32)
    labels_one_hot = torch.from_numpy(labels_formatter(labels))
    test_labels = torch.argmax(labels_one_hot, dim=0).long()
    IoU = M.iou(torch.unsqueeze(avg_probs, 0),
                      torch.unsqueeze(test_labels, 0),
                      reduction='none',
                      num_classes=module.num_classes)
    metrics['IoU_0'] = IoU[0]
    metrics['IoU_1'] = IoU[1]
    metrics['IoU'] = IoU.mean()
    print(metrics)

    if args.output_path is not None:
        pred_profile = rasterio.open(args.image_path).profile
        pred_profile.update(count=1)
        pred_profile.update(nodata=None)
        with rasterio.open(args.output_path, 'w', **pred_profile) as dst:
            dst.write(np.uint8(np.argmax(pred_sum, axis=0)), indexes=1)


if __name__ == "__main__":

    main()
