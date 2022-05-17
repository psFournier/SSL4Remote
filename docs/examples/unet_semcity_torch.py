from argparse import ArgumentParser
from dl_toolbox.lightning_modules import Unet
from dl_toolbox.callbacks import SegmentationImagesVisualisation, ConfMatLogger
import torch
from rasterio.windows import Window
from dl_toolbox.torch_datasets import SemcityBdsdDs
import os
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from dl_toolbox.utils import worker_init_function
import segmentation_models_pytorch as smp
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
import time
import tabulate


def main():

    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--train_with_void", action='store_true')
    parser.add_argument("--eval_with_void", action='store_true')
    parser.add_argument("--in_channels", type=int)
    parser.add_argument("--pretrained", action='store_true')
    parser.add_argument("--encoder", type=str)
    parser.add_argument("--initial_lr", type=float)
    parser.add_argument("--final_lr", type=float)
    parser.add_argument("--lr_milestones", nargs=2, type=float)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--splitfile_path", type=str)
    parser.add_argument("--test_fold", type=int)
    parser.add_argument("--epoch_len", type=int, default=10000)
    parser.add_argument("--sup_batch_size", type=int, default=16)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--workers", default=6, type=int)
    parser.add_argument('--img_aug', type=str, default='no')
    parser.add_argument('--batch_aug', type=str, default='no')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    args_dict = vars(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Train dataset
    dataset1 = SemcityBdsdDs(
        image_path=os.path.join(args.data_path, 'BDSD_M_3_4_7_8.tif'),
        label_path=os.path.join(args.data_path, 'GT_3_4_7_8.tif'),
        fixed_crops=False,
        tile=Window(
            col_off=0,
            row_off=0,
            width=2000,
            height=2000
        ),
        crop_size=args.crop_size,
        crop_step=args.crop_size,
        img_aug=args.img_aug
    )
    dataset2 = SemcityBdsdDs(
        image_path=os.path.join(args.data_path, 'BDSD_M_3_4_7_8.tif'),
        label_path=os.path.join(args.data_path, 'GT_3_4_7_8.tif'),
        fixed_crops=False,
        tile=Window(
            col_off=2000,
            row_off=2000,
            width=2000,
            height=2000
        ),
        crop_size=args.crop_size,
        crop_step=args.crop_size,
        img_aug=args.img_aug
    )
    trainset = ConcatDataset([dataset1, dataset2])

    valset = SemcityBdsdDs(
        image_path=os.path.join(args.data_path, 'BDSD_M_3_4_7_8.tif'),
        label_path=os.path.join(args.data_path, 'GT_3_4_7_8.tif'),
        fixed_crops=True,
        tile=Window(
            col_off=4000,
            row_off=4000,
            width=2000,
            height=2000
        ),
        crop_size=args.crop_size,
        crop_step=args.crop_size,
        img_aug='no'
    )

    train_sampler = RandomSampler(
        data_source=trainset,
        replacement=True,
        num_samples=args.epoch_len
    )

    train_dataloader = DataLoader(
        dataset=trainset,
        batch_size=args.sup_batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers
    )

    val_dataloader = DataLoader(
        dataset=valset,
        shuffle=False,
        batch_size=args.sup_batch_size,
        num_workers=args.num_workers,
    )

    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights='imagenet' if args.pretrained else None,
        in_channels=args.in_channels,
        classes=args.num_classes if args.train_with_void else args.num_classes
                                                              - 1,
        decoder_use_batchnorm=True
    )

    model.to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

    optimizer = SGD(
        model,
        lr=args.initial_lr,
        momentum=0.9,
    )

    def lambda_lr(epoch):

        m = epoch / args.max_epochs
        if m < args.lr_milestones[0]:
            return 1
        elif m < args.lr_milestones[1]:
            return 1 + ((m - args.lr_milestones[0]) / (
                        args.lr_milestones[1] - args.lr_milestones[0])) * (
                               args.final_lr / args.initial_lr - 1)
        else:
            return args.final_lr / args.initial_lr

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda_lr
    )

    start_epoch = 0
    columns = ['ep', 'train_loss', 'val_loss', 'time']

    for epoch in range(start_epoch, args.epochs):

        time_ep = time.time()

        scheduler.step()

        loss_sum = 0.0

        model.train()

        for i, (input, target) in enumerate(train_dataloader):

            input = input['image'].to(device)
            target = input['mask'].to(device)

            # mask processing to do here
            #######################

            logits = model(input)

            # loss computation with masks to do here instead
            #######################
            loss = loss_fn(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.data[0] * input.size(0)

        train_res = {'loss': loss_sum / len(train_dataloader)}

        loss_sum = 0.0

        model.eval()

        for i, (input, target) in enumerate(val_dataloader):

            input = input.to(device)
            target = target.to(device)

            output = model(input)
            loss = loss_fn(output, target)

            loss_sum += loss.data[0] * input.size(0)

        val_res = {'loss': loss_sum / len(val_dataloader)}

        time_ep = time.time() - time_ep
        values = [epoch + 1, train_res['loss'], val_res['loss'], time_ep]
        table = tabulate.tabulate([values], columns, tablefmt='simple',
                                  floatfmt='8.4f')
        print(table)


if __name__ == "__main__":

    main()
