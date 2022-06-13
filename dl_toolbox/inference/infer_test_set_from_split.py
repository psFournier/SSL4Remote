from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os
import pandas as pd
import csv
import torch
import numpy as np
import rasterio
from dl_toolbox.lightning_modules import Unet
import dl_toolbox.inference as dl_inf
from dl_toolbox.callbacks import plot_confusion_matrix
from dl_toolbox.torch_datasets import DigitanieDs, SemcityBdsdDs
import pathlib

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
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--write_probas", action='store_true')
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

    with open(self.splitfile_path, newline='') as splitfile:
        train_sets, val_sets = build_split_from_csv(
            splitfile=splitfile,
            dataset_cls=self.dataset_cls,
            train_folds=self.train_folds,
            test_folds=self.test_folds,
            img_aug=self.img_aug,
            data_path=self.data_path,
            crop_size = self.crop_size,
            one_hot=True
        )

    for dataset in val_sets:

        print('Computing probas')
        probas = dl_inf.compute_probas(
            dataset=dataset,
            module=module,
            batch_size=args.batch_size,
            workers=args.workers,
            tta=args.tta,
            mode='sigmoid'
        )
        preds = np.argmax(np.expand_dims(probas, 0), 1)
        preds += int(not args.train_with_void)

        labels = dataset.read_label(label_path=dataset.label_path, window=dataset.tile)
        MERGE_SEMCITY = [[0,7], [3], [6], [2], [4], [1, 5]]
        labels = MergeLabels(MERGE_SEMCITY)(labels)
        row_cm = confusion_matrix(
            labels.flatten(),
            np.squeeze(preds).flatten(),
            labels = np.arange(6)
        )

        global_cm += row_cm

#    with open(args.splitfile_path, newline='') as splitfile:
#        reader = csv.reader(splitfile)
#        next(reader)
#        for row in reader:
#            city,tile_num, img_path, label_path, x0, y0, patch_width, patch_height, fold_id = row
#            if int(fold_id) in args.test_fold:
#                print(row)
#                image_path = os.path.join(args.data_path, img_path)
#                tile=(int(x0), int(y0), int(patch_width), int(patch_height))
#                probas = dl_inf.compute_probas(
#                    image_path=image_path,
#                    tile=tile,
#                    dataset_type=args.dataset,
#                    module=module,
#                    batch_size=args.batch_size,
#                    workers=args.workers,
#                    crop_size=args.crop_size,
#                    crop_step=args.crop_step,
#                    tta=args.tta,
#                    mode='sigmoid'
#                )
#
#                if args.write_probas:
#                    initial_profile = rasterio.open(image_path).profile
#                    pathlib.Path(args.output_path).mkdir(parents=True, exist_ok=True)
#                    output_probas = os.path.join(args.output_path, '_'.join([city, tile_num, x0, y0]) + '.tif')
#                    dl_inf.write_array(
#                        inputs=probas[[4],...],
#                        tile=tile,
#                        output_path=output_probas,
#                        profile=initial_profile
#                    )
#
#                preds = dl_inf.probas_to_preds(
#                    torch.unsqueeze(probas, dim=0)
#                ) + int(not args.train_with_void)

    
    #ignore_index = -1 if args.eval_with_void else 0
    print('Computing metrics')
    metrics_per_class_df, average_metrics_df = dl_inf.cm2metrics(global_cm, ignore_index=-1)
    metrics_per_class_df.rename(
        index=dict([(i, l) for i, l in enumerate(SemcityBdsd2Ds.labels.keys())]),
        inplace=True
    )
    with pd.ExcelWriter(os.path.join(args.output_path, 'metrics.xlsx')) as writer:
        metrics_per_class_df.to_excel(writer, sheet_name='metrics_per_class')
        average_metrics_df.to_excel(writer, sheet_name='average_metrics')
    print(metrics_per_class_df)
    print(average_metrics_df)
    norm_confmat = global_cm/(np.sum(global_cm,axis=1)[:,None]) 
    class_names = [l[1] for l in labels]
    figure = plot_confusion_matrix(norm_confmat, class_names=class_names)
    plt.savefig(os.path.join(args.output_path,'confmat.jpg'))


if __name__ == "__main__":

    main()

