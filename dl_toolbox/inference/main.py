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

def main():

    """"
    See https://confluence.cnes.fr/pages/viewpage.action?pageId=85723974 for how to use
    this script.
    """

    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--output_probas", type=str, default=None)
    parser.add_argument("--tile", nargs=4, type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--tta", nargs='+', type=str, default=[])
    parser.add_argument("--output_preds", type=str, default=None)
    parser.add_argument("--stat_class", type=int, default=-1)
    parser.add_argument("--output_errors", type=str, default=None)
    parser.add_argument("--label_path", type=str, default=None)
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
        image_path=args.image_path,
        tile=args.tile,
        dataset_type=args.dataset,
        module=module,
        batch_size=args.batch_size,
        workers=args.workers,
        crop_size=args.crop_size,
        crop_step=args.crop_step,
        tta=args.tta
    )

    initial_profile = rasterio.open(args.image_path).profile
    preds = torch.squeeze(probas_to_preds(torch.unsqueeze(probas, dim=0)))
    
    if args.output_probas:    
        
        write_probas(
            probas=probas,
            output_path=args.output_probas,
            initial_profile=initial_profile,
            num_classes=args.num_classes
        )

    if args.output_preds:
        
        write_rgb_preds(
            rgb_preds=preds,
            output_path=args.output_preds,
            initial_profile=initial_profile
        )


    if args.label_path:
        
        metrics = compute_metrics(
            preds=preds,
            label_path=args.label_path,
            tile=args.tile
        )
        print(metrics)

        if args.stat_class >= 0:
            
            assert args.output_errors
            visualize_errors(
                preds=preds,
                label_path=args.label_path,
                tile=args.tile,
                output_path=args.output_errors,
                class_id=args.stat_class,
                initial_profile=initial_profile
            )




if __name__ == "__main__":

    main()

