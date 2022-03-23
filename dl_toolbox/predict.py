from argparse import ArgumentParser
from torch.utils.data import DataLoader, ConcatDataset
import torch
import imagesize
import numpy as np
import rasterio
from rasterio.windows import Window
from torch import nn
from torch_datasets import SemcityBdsdDs, DigitanieDs
from torch_collate import CollateDefault
from lightning_modules import Unet
from utils import worker_init_function
import torchmetrics.functional as  M
from utils import get_tiles
from augmentations import image_level_aug

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

anti_t_dict = {
    'hflip': 'hflip',
    'vflip': 'vflip',
    'd1flip': 'd1flip',
    'd2flip': 'd2flip',
    'rot90': 'rot270',
    'rot180': 'rot180',
    'rot270': 'rot90'
}


def main():

    """"
    See https://confluence.cnes.fr/pages/viewpage.action?pageId=85723974 for how to use
    this script.
    """

    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--splitfile_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)

    # Optional arguments (label file for metrics computation + forward and tta paramters)
    parser.add_argument("--label_path", type=str, default=None)
    parser.add_argument("--tta", nargs='+', type=str, default=[])
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--in_channels", type=int)
    parser.add_argument("--crop_size", type=int)
    parser.add_argument("--crop_step", type=int)

    args = parser.parse_args()

    # Retrieving the full configuration
    # config = OmegaConf.load(args.config_path)

    # Loading the module used for training with the weights from the checkpoint.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt_path, map_location=device)

    module = Unet(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        pretrained=False,
        learning_rate=0.001,
        encoder='efficientnet-b0'
        initial_lr=None,
        final_lr=None,
        weight_decay=None,
        lr_milestones=None
    )
    # module = DummyModule(model=instantiate(config.model), config_loss=config.loss)
    module.load_state_dict(ckpt['state_dict'])
    module.eval()
    module.to(device)
    
    datasets = []
    with open(args.splitfile_path, newline='') as splitfile:
        reader=csv.reader(splitfile)
        next(reader)
        for row in reader:
            if int(row[8])==args.test_fold:
                window = Window(
                    col_off=int(row[4]),
                    row_off=int(row[5]),
                    width=int(row[6]),
                    height=int(row[7])
                )
                dataset = DigitanieDs(
                    image_path=row[2],
                    label_path=row[3],
                    fixed_crops=True,
                    tile=window,
                    crop_size=args.crop_size,
                    crop_step=args.crop_step,
                    img_aug='no'
                )
                datasets.append(dataset)
    val_set = ConcatDataset(datasets)
                
    # The collate function is needed to process the read windows from
    # the dataset __get_item__ method.
    dataloader = DataLoader(
        dataset=val_set,
        shuffle=False,
        collate_fn=CollateDefault(),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=worker_init_function
    )
 
    # The full matrix used to cumulate results from overlapping tiles or tta
    pred_sum = torch.zeros(size=(module.num_classes, height, width))
    metrics = {'accuracy': []}

    for batch in dataloader:

        inputs, _, windows = batch['image'], batch['mask'], batch['window']
        
        with torch.no_grad():
            outputs = module.forward(inputs.to(device)).cpu()

        split_pred = np.split(outputs, outputs.shape[0], axis=0)
        pred_list = [np.squeeze(e, axis=0) for e in split_pred]
        window_list = windows[:]
        
        for pred, window in zip(pred_list, window_list):
            pred_sum[:, window.row_off:window.row_off + window.width, window.col_off:window.col_off + window.height] += pred
    
    for t_name in args.tta:
        for batch in dataloader:
            inputs = batch['image']
            aug_inputs, _ = image_level_aug[t_name](p=1)(inputs)
            with torch.no_grad():
                outputs = module.forward(aug_inputs.to(device)).cpu()
            if t_name in anti_t_dict:
                outputs, _ = image_level_aug[anti_t_dict[t_name]](p=1)(outputs)
            split_pred = np.split(outputs, outputs.shape[0], axis=0)
            pred_list = [np.squeeze(e, axis=0) for e in split_pred]
            window_list = batch['window'][:]
            
            for pred, window in zip(pred_list, window_list):
                pred_sum[:, 
                        window.row_off:window.row_off + window.width, 
                        window.col_off:window.col_off + window.height] += pred

    preds = pred_sum.argmax(dim=0) + 1

    if args.label_path is not None:

        # test_labels = torch.zeros(size=(height, width), dtype=torch.int)
        windows = get_tiles(
            nols=width,
            nrows=height,
            size=128,
            step=128
        )
        for window in windows:        
            window_labels = rasterio.open(args.label_path).read(window=window, out_dtype=np.uint8)
            window_labels = torch.from_numpy(window_labels)
            window_preds = preds[window.row_off:window.row_off + window.width, window.col_off:window.col_off + window.height]
            accuracy = M.accuracy(torch.unsqueeze(window_preds, 0),
                                  torch.unsqueeze(window_labels, 0),
                                  ignore_index=0)
            metrics['accuracy'].append(accuracy)

        # IoU = M.iou(torch.unsqueeze(avg_probs, 0),
        #             torch.unsqueeze(test_labels, 0),
        #             reduction='none',
        #             num_classes=module.num_classes+1,
        #             ignore_index=0)
        # for i in range(module.num_classes):
        #     class_name = SemcityBdsdDs.labels_desc[i+1][2]
        #     metrics['Train_IoU_{}'.format(class_name)] = IoU[i]
        # metrics['IoU'] = IoU.mean()

        print(np.mean(metrics['accuracy']))

    if args.output_path is not None:

        preds_rgb = torch.zeros(size=(height, width, 3), dtype=torch.int)
        for window in get_tiles(
                nols=width,
                nrows=height,
                size=128,
                step=128
        ):
            pred = preds[window.row_off:window.row_off + window.width, window.col_off:window.col_off + window.height]
            for val, color in enumerate(dataset.DATASET_DESC['label_colors']):
                mask = pred == val
                preds_rgb[window.row_off:window.row_off + window.width, window.col_off:window.col_off + window.height, :][mask] = torch.tensor(color, dtype=torch.int)
        preds_rgb = preds_rgb.numpy().astype(np.uint8)

        pred_profile = rasterio.open(args.label_path).profile
        profile = {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': None,
            'width': pred_profile['width'],
            'height': pred_profile['height'],
            'count': 3,
            'crs': pred_profile['crs'],
            'transform': pred_profile['transform'],
            'tiled': False,
            'interleave': 'pixel'
        }
        with rasterio.open(args.output_path, 'w', **profile) as dst:
            dst.write(preds_rgb[:,:,0], indexes=1)
            dst.write(preds_rgb[:,:,1], indexes=2)
            dst.write(preds_rgb[:,:,2], indexes=3)


if __name__ == "__main__":

    main()
