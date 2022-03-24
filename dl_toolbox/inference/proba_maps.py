from torch.utils.data import DataLoader, ConcatDataset
from argparse import ArgumentParser
import torch
import imagesize
import numpy as np
import rasterio
from rasterio.windows import Window
from torch import nn
from dl_toolbox.torch_datasets import SemcityBdsdDs, DigitanieDs
from dl_toolbox.torch_collate import CustomCollate
from dl_toolbox.lightning_modules import Unet
from dl_toolbox.utils import worker_init_function, get_tiles
import torchmetrics.functional as  M
from dl_toolbox.augmentations import image_level_aug

anti_t_dict = {
    'hflip': 'hflip',
    'vflip': 'vflip',
    'd1flip': 'd1flip',
    'd2flip': 'd2flip',
    'rot90': 'rot270',
    'rot180': 'rot180',
    'rot270': 'rot90'
}

datasets = {
    'semcity': 
}


def compute_probas(
    image_path,
    dataset_type,
    tile,
    module,
    tta
)
    
    device = module.device

    col_off, row_off, width, height = tile
    window = Window(
        col_off=col_off,
        row_off=row_off,
        width=width,
        height=height
    )

    dataset = SemcityBdsdDs(
        image_path=args.image_path,
        fixed_crops=True,
        tile=window,
        crop_size=args.crop_size,
        crop_step=args.crop_step,
        img_aug='no'
    )

    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        collate_fn=CustomCollate(),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=worker_init_function
    )


    pred_sum = torch.zeros(size=(module.num_classes, height, width))

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

    probas = pred_sum.softmax(dim=0)

    return probas

def write_probas(probas, output_path, initial_profile, num_classes):

    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': None,
        'width': initial_profile['width'],
        'height': initial_profile['height'],
        'count': num_classes,
        'crs': initial_profile['crs'],
        'transform': initial_profile['transform'],
        'tiled': False,
        'interleave': 'pixel'
    }
    with rasterio.open(output_path, 'w', **profile) as dst:
        for i in range(num_classes):
            dst.write(probas[i, :, :], indexes=i+1)


def main():

    """"
    See https://confluence.cnes.fr/pages/viewpage.action?pageId=85723974 for how to use
    this script.
    """

    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--tile", nargs=4, type=int)

    # Optional arguments (label file for metrics computation + forward and tta paramters)
    parser.add_argument("--tta", nargs='+', type=str, default=[])
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--in_channels", type=int)
    parser.add_argument("--crop_size", type=int)
    parser.add_argument("--crop_step", type=int)

    args = parser.parse_args()


    # Loading the module used for training with the weights from the checkpoint.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt_path, map_location=device)

    module = Unet(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        pretrained=False,
        encoder='efficientnet-b5'
    )

    # module = DummyModule(model=instantiate(config.model), config_loss=config.loss)
    module.load_state_dict(ckpt['state_dict'])
    module.eval()
    module.to(device)
    
    probas = compute_probas(
        dataloader = dataloader,
        module=module,
        device=device,
        tta=args.tta
    )
        
    initial_profile = rasterio.open(args.image_path).profile
    write_probas(
        probas=probas,
        output_path=args.output_path,
        initial_profile=initial_profile,
        num_classes=args.num_classes
    )

if __name__ == "__main__":

    main()

