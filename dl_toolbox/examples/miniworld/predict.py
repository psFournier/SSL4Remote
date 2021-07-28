from argparse import ArgumentParser
from torch.utils.data import DataLoader
import torch
import imagesize
# from omegaconf import OmegaConf
import numpy as np
import rasterio
from torch import nn
from dl_toolbox.torch_datasets import OneImage
from dl_toolbox.torch_collate import CollateDefault
from dl_toolbox.lightning_modules import SupervisedBaseline
from dl_toolbox.utils import worker_init_function
from dl_toolbox.inference import apply_tta
import torchmetrics.functional as  M
from dl_toolbox.torch_datasets import inria_label_formatter

# class DummyModule(nn.Module):
#     def __init__(self, model, config_loss):
#         super().__init__()
#         self.model = model
#         # https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723/8
#         self.train_loss = instantiate(config_loss)
#         self.val_loss = instantiate(config_loss)
#         self.test_loss = instantiate(config_loss)
#
#     def forward(self, x):
#         return self.model.forward(x)

def main():

    """"
    See https://confluence.cnes.fr/pages/viewpage.action?pageId=85723974 for how to use
    this script.
    """

    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--output_path", type=str)

    # Optional arguments (label file for metrics computation + forward and tta paramters)
    parser.add_argument("--label_path", type=str, default=None)
    parser.add_argument("--tta", nargs='+', type=str, default=[])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--tile_size", type=int, default=128)
    parser.add_argument("--tile_step", type=int, default=64)

    args = parser.parse_args()

    # Retrieving the full configuration
    # config = OmegaConf.load(args.config_path)

    # Loading the module used for training with the weights from the checkpoint.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt_path, map_location=device)

    module = SupervisedBaseline()
    # module = DummyModule(model=instantiate(config.model), config_loss=config.loss)
    module.load_state_dict(ckpt['state_dict'])
    module.eval()
    module.to(device)

    dataset = OneImage(
        image_path=args.image_path,
        label_path=args.label_path,
        tile_size=args.tile_size,
        tile_step=args.tile_step,
        crop_size=args.tile_size,
    )

    # The collate function is needed to process the read windows from
    # the dataset __get_item__ method.
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        collate_fn=CollateDefault(),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=worker_init_function
    )

    # The full matrix used to cumulate results from overlapping tiles or tta
    width, height = imagesize.get(args.image_path)
    pred_sum = torch.zeros(size=(module.num_classes, height, width))
    metrics = {}

    with torch.no_grad():

        for batch in dataloader:

            inputs, _, windows = batch['image'], batch['mask'], batch['window']

            with torch.no_grad():
                outputs = module.forward(inputs.to(device)).cpu()

            split_pred = np.split(outputs, outputs.shape[0], axis=0)
            pred_list = [np.squeeze(e, axis=0) for e in split_pred]
            window_list = windows[:]

            if args.tta:
                tta_preds, tta_windows = apply_tta(args.tta, device, module, batch)
                pred_list += tta_preds
                window_list += tta_windows

            for pred, window in zip(pred_list, window_list):
                pred_sum[:, window.row_off:window.row_off + window.width, window.col_off:window.col_off + window.height] += pred
            # break


    avg_probs = pred_sum.softmax(dim=0)
    labels = rasterio.open(args.label_path).read(out_dtype=np.float32)
    labels = inria_label_formatter(labels)
    labels_one_hot = torch.from_numpy(labels)
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
        profile = {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': None,
            'width': pred_profile['width'],
            'height': pred_profile['height'],
            'count': 1,
            'crs': pred_profile['crs'],
            'transform': pred_profile['transform'],
            'tiled': False,
            'interleave': 'pixel'
        }
        with rasterio.open(args.output_path, 'w', **profile) as dst:
            dst.write(np.uint8(np.argmax(pred_sum, axis=0)), indexes=1)


if __name__ == "__main__":

    main()
