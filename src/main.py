# https://github.com/phborba/pytorch_segmentation_models_trainer/blob/main/pytorch_segmentation_models_trainer/main.py
import datetime
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, loggers

from pl_datamodules import IsprsVaiSemisup, IsprsVaiSup, MiniworldSup, \
    MiniworldSemisup
from pl_modules import MeanTeacher, SupervisedBaseline


def main():

    modules = {
        'supervised_baseline': SupervisedBaseline,
        'mean_teacher': MeanTeacher
    }

    datamodules = {
        'isprs_vai_semisup': IsprsVaiSemisup,
        'isprs_vai_sup': IsprsVaiSup,
        'miniworld_semisup': MiniworldSemisup,
        'miniworld_sup': MiniworldSup
    }

    parser = ArgumentParser()
    parser.add_argument("--datamodule", type=str, default='isprs_vai_sup')
    parser.add_argument("--module", type=str, default="supervised_baseline")

    args = parser.parse_known_args()[0]
    # Each class of interest contains a method to add its specific arguments
    # to the parser
    parser = datamodules[args.datamodule].add_model_specific_args(parser)
    parser = modules[args.module].add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--output_dir", type=str, default="./outputs")

    args = parser.parse_args()
    args_dict = vars(args)

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    tensorboard = loggers.TensorBoardLogger(
        save_dir=args.output_dir,
        name="tensorboard",
        version="%s_%s" % (args.exp_name, current_date)
    )

    # The lightning datamodule deals with instantiating the proper dataloaders.
    pl_datamodule = datamodules[args.datamodule](**args_dict)

    # The lightning module is where the training schema is implemented.
    pl_module = modules[args.module](**args_dict)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_sup_loss',
        mode='min',
        save_weights_only=True
    )
    # Using from_argparse_args enables to use any standard parameter of thea
    # lightning Trainer class without having to manually add them to the parser.
    # In particular, a parameter that does not explicitly appears here but is
    # important is multiple_trainloader_mode, which governs how the supervised and
    # unsupervised dataloaders interact.
    # See https://pytorch-lightning.readthedocs.io/en/latest/advanced/multiple_loaders.html
    # for information.
    trainer = Trainer.from_argparse_args(
        args,
        logger=tensorboard,
        callbacks=pl_module.callbacks + [
            # checkpoint_callback
        ],
        benchmark=True,
        min_epochs=500,
        max_epochs=1000,
        # num_sanity_val_steps=1,
        # log_every_n_steps=10,
        # flush_logs_every_n_steps=10
        # val_check_interval=1.0,
        # check_val_every_n_epoch=1
    )

    trainer.fit(model=pl_module, datamodule=pl_datamodule)


if __name__ == "__main__":

    main()
