from argparse import ArgumentParser
import torch
import numpy as np
import rasterio
from dl_toolbox.lightning_modules import Unet
import dl_toolbox.inference as dl_inf
from dl_toolbox.torch_datasets import DigitanieDs, SemcityBdsdDs
from sklearn.metrics import confusion_matrix as confusion_matrix

datasets = {
    'semcity': SemcityBdsdDs,
    'digi_toulouse': DigitanieToulouseDs,
    'digi_biarritz': DigitanieBiarritzDs
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

    module.load_state_dict(ckpt['state_dict'])
    module.eval()
    module.to(device)

    window = get_window(args.tile)
    dataset = datasets[args.dataset](
        image_path=args.image_path,
        fixed_crops=True,
        tile=window,
        crop_size=args.crop_size,
        crop_step=args.crop_step,
        img_aug='no'
    )
    
    print('Computing probas')
    probas = dl_inf.compute_probas(
        dataset=dataset,
        window=window,
        module=module,
        batch_size=args.batch_size,
        workers=args.workers,
        tta=args.tta,
        mode='sigmoid'
    )

    # Adding batch dimension 1 to probas before extracting predictions
    preds = dl_inf.probas_to_preds(torch.unsqueeze(probas, dim=0))
    # Adding 1 to prediction when the void class has been ignored
    preds += int(not args.train_with_void)
    
    initial_profile = rasterio.open(args.image_path).profile

    if args.output_probas:    

        dl_inf.write_array(
            inputs=probas,
            tile=args.tile,
            output_path=args.output_probas,
            profile=initial_profile
        )

    if args.output_preds:

        rgb = dataset.labels_to_rgb(preds)
        
        dl_inf.write_array(
            inputs=np.squeeze(rgb),
            tile=args.tile,
            output_path=args.output_preds,
            profile=initial_profile
        )


    if args.label_path:
        
        print('Computing metrics')
        labels = dataset.read_label(args.label_path, window=window)
        cm = confusion_matrix(
            labels.numpy().flatten(),
            torch.squeeze(preds).numpy().flatten(),
            labels = np.arange(args.num_classes)
        )

        #ignore_index = None if args.eval_with_void else 0
        metrics_per_class_df, average_metrics_df = dl_inf.cm2metrics(cm, ignore_index=-1)
        labels = datasets[args.dataset].DATASET_DESC['labels']
        metrics_per_class_df.rename(
            index=dict(labels),
            inplace=True
        )

        print(metrics_per_class_df)
        print(average_metrics_df)

        if args.stat_class >= 0:
            
            assert args.output_errors
            dl_inf.visualize_errors(
                preds=torch.squeeze(preds),
                label_path=args.label_path,
                dataset_type=args.dataset,
                tile=args.tile,
                output_path=args.output_errors,
                class_id=args.stat_class,
                initial_profile=initial_profile,
                eval_with_void=args.eval_with_void
            )
        
        else:

            assert args.output_errors
            dl_inf.visualize_errors(
                preds=torch.squeeze(preds),
                label_path=args.label_path,
                dataset_type=args.dataset,
                tile=args.tile,
                output_path=args.output_errors,
                initial_profile=initial_profile,
                eval_with_void=args.eval_with_void
            )
 



if __name__ == "__main__":

    main()

