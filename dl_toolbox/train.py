from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler
import sys

from lightning_modules import *
from lightning_datamodules import *
from callbacks import SegmentationImagesVisualisation, CustomSwa, ConfMatLogger

modules = {
        'sup': Unet,
}


datamodules = {
    'semcity_bdsd': {
        'sup': SemcityBdsdDm,
        'mean_teacher': SemcityBdsdDmSemisup
    },
    'miniworld_generalisation': {
        'sup': MiniworldDmV2,
        'mean_teacher': MiniworldDmV2Semisup
    },
    'miniworld_transfert': {
        'sup': MiniworldDmV3,
        'mean_teacher': MiniworldDmV3Semisup
    },
    'phr_pan': {
        'sup': PhrPanDm,
        'mean_teacher': PhrPanDmSemisup
    }
}

def main():

    # Reading parameters
    parser = ArgumentParser()

    parser.add_argument("--datamodule", type=str, default='semcity_bdsd')
    parser.add_argument("--module", type=str, default="sup")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--exp_name", type=str)

    # Datamodule and module classes add their own specific command line
    # arguments, so we retrieve them to go further with the parser.
    args = parser.parse_known_args()[0]
    parser = modules[args.module].add_model_specific_args(parser)
    parser = datamodules[args.datamodule][args.module].add_model_specific_args(parser)

    # The Trainer class also enables to add many arguments to customize the
    # training process (see Lightning Doc)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    args_dict = vars(args)

    # The lightning datamodule deals with instantiating the proper dataloaders.
    pl_datamodule = datamodules[args.datamodule][args.module](**args_dict)

    # The lightning module is where the training schema is implemented. Class
    # weights are a property of the dataset being processed, given by its class.
    # args_dict['class_weights'] = datamodules[args.datamodule][args.module].class_weights
    pl_module = modules[args.module](**args_dict)

    # Logs will be stored in the directory 'tensorboard' of the output
    # directory, and the individual log of each new run will be stored in a
    # subdirectory with the datetime as name. The parameters corresponding to
    # the run can be retrieved in Tensorboard.
    tensorboard = loggers.TensorBoardLogger(
        save_dir=args.output_dir,
        name=args.exp_name,
        default_hp_metric=False
    )

    # Callback that saves the weights of the last two epochs
    last_2_epoch_ckpt = ModelCheckpoint(
        monitor='epoch',
        mode='max',
        save_top_k=2,
        verbose=True
    )

    # Monitoring time spent in each call. Difficult to understand the data
    # loading part when multiple workers are at use.
    profiler = SimpleProfiler()

    # Using from_argparse_args enables to use any standard parameter of the
    # lightning Trainer class without having to manually add them to the parser.
    trainer = Trainer.from_argparse_args(
        args,
        logger=tensorboard,
        profiler=profiler,
        callbacks=[
            # last_2_epoch_ckpt,
            # CustomSwa(
            #     swa_epoch_start=0.8,
            #     swa_lrs=0.01,
            #     device=None,
            #     annealing_epochs=2
            # ),
            SegmentationImagesVisualisation(),
            ConfMatLogger(),
            LearningRateMonitor()
        ],
        log_every_n_steps=300,
        flush_logs_every_n_steps=1000,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        benchmark=True
    )

    trainer.fit(model=pl_module, datamodule=pl_datamodule)


if __name__ == "__main__":

    main()
