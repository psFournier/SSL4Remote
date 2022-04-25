from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os
import csv
import torch
import numpy as np
import rasterio
from dl_toolbox.lightning_modules import Unet
import dl_toolbox.inference as dl_inf
from dl_toolbox.callbacks import plot_confusion_matrix
from dl_toolbox.torch_datasets import DigitanieDs, SemcityBdsdDs

datasets = {
    'semcity': SemcityBdsdDs,
    'digitanie': DigitanieDs
}


def main():

    """"
    See https://confluence.cnes.fr/pages/viewpage.action?pageId=85723974 for how to use
    this script.
    """

    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--splitfile_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--test_fold", nargs='+', type=int)
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
    parser.add_argument("--train_with_void", action='store_true')
    parser.add_argument("--eval_with_void", action='store_true')


    args = parser.parse_args()


    # Loading the module used for training with the weights from the checkpoint.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt_path, map_location=device)

    module = Unet(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        pretrained=False,
        encoder=args.encoder,
        train_with_void=args.train_with_void
    )

    # module = DummyModule(model=instantiate(config.model), config_loss=config.loss)
    module.load_state_dict(ckpt['state_dict'])
    module.eval()
    module.to(device)
    
    metrics = {
        'accuracy': [],
        'f1': [],
        'iou': [],
        'f1_per_class': [],
        'accu_per_class': []
    }
    global_cm = np.zeros(shape=(args.num_classes, args.num_classes))
    with open(args.splitfile_path, newline='') as splitfile:
        reader = csv.reader(splitfile)
        next(reader)
        for row in reader:
            city,tile, img_path, label_path, x0, y0, patch_width, patch_height, fold_id = row
            if int(fold_id) in args.test_fold:
                probas = dl_inf.compute_probas(
                    image_path=os.path.join(args.data_path, img_path),
                    tile=(int(x0), int(y0), int(patch_width), int(patch_height)),
                    dataset_type=args.dataset,
                    module=module,
                    batch_size=args.batch_size,
                    workers=args.workers,
                    crop_size=args.crop_size,
                    crop_step=args.crop_step,
                    tta=args.tta,
                    mode='sigmoid'
                )
                preds = dl_inf.probas_to_preds(
                    torch.unsqueeze(probas, dim=0)
                ) + int(not args.train_with_void)
                row_cm = dl_inf.compute_cm(
                    preds=torch.squeeze(preds),
                    label_path=os.path.join(args.data_path, label_path),
                    dataset_type=args.dataset,
                    tile=(int(x0), int(y0), int(patch_width), int(patch_height)),
                    eval_with_void=args.eval_with_void,
                    num_classes=args.num_classes
                )
                global_cm += row_cm
    
    ignore_index = -1 if args.eval_with_void else 0
    metrics_per_class_df, average_metrics_df = dl_inf.cm2metrics(global_cm, ignore_index=ignore_index)
    print(metrics_per_class_df)
    print(average_metrics_df)
    norm_confmat = global_cm/(np.sum(global_cm,axis=1)[:,None]) 
    class_names = [l[1] for l in datasets[args.dataset].DATASET_DESC['labels']]
    figure = plot_confusion_matrix(norm_confmat, class_names=class_names)
    plt.savefig('/home/eh/fournip/cm.jpg')


if __name__ == "__main__":

    main()

