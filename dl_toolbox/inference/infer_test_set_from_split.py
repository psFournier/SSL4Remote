from argparse import ArgumentParser
import torch
import numpy as np
import rasterio
from dl_toolbox.lightning_modules import Unet
import dl_toolbox.inference as dl_inf

def main():

    """"
    See https://confluence.cnes.fr/pages/viewpage.action?pageId=85723974 for how to use
    this script.
    """

    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--splitfile_path", type=str)
    parser.add_argument("--test_fold", type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--tta", nargs='+', type=str, default=[])
    parser.add_argument("--label_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--in_channels", type=int)
    parser.add_argument("--crop_size", type=int)
    parser.add_argument("--crop_step", type=int)
    parser.add_argument("--encoder", type=str)

    args = parser.parse_args()


    # Loading the module used for training with the weights from the checkpoint.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt_path, map_location=device)

    module = Unet(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        pretrained=False,
        encoder=args.encoder
    )

    # module = DummyModule(model=instantiate(config.model), config_loss=config.loss)
    module.load_state_dict(ckpt['state_dict'])
    module.eval()
    module.to(device)
    
    metrics = {
        'accuracy': []
    }
    with open(args.splitfile_path, newline='') as splitfile:
        reader = csv.reader(splitfile)
        next(reader)
        for row in reader:
            city, img_path, label_path, x0, y0, patch_width, patch_height, fold_id = *row
            if fold_id == args.test_fold:
                probas = dl_inf.compute_probas(
                    image_path=img_path,
                    tile=(x0, y0, patch_width, patch_height),
                    dataset_type=args.dataset,
                    module=module,
                    batch_size=args.batch_size,
                    workers=args.workers,
                    crop_size=args.crop_size,
                    crop_step=args.cropo_step,
                    tta=args.tta
                )
                preds = dl_inf.probas_to_preds(
                    torch.unsqueeze(probas, dim=0)
                ) + 1
                row_metrics = dl_inf.compute_metrics(
                    preds=torch.squeeze(preds),
                    label_path=label_path,
                    dataset_type=args.dataset,
                    tile=(x0, y0, patch_width, patch_height)
                )
                for k, v in row_metrics.items():
                    metrics[k].append(v)

    final_metrics = {k: np.mean(v) for k, v in metrics.items()}
    print(final_metrics)

if __name__ == "__main__":

    main()

