# This folder should disappear when using hydra:
# a single 'main' file would be enough given that hydra enables to instantiate
# whole classes (modules, datamodules in particular) from command line arguments.

import datetime
from argparse import ArgumentParser

from pytorch_lightning import Trainer, loggers

from pl_datamodules import IsprsVaiSemisup
from pl_modules import MeanTeacher


def main():

    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--output_dir", type=str, default="./outputs")

    # Each class of interest contains a method to add its specific arguments
    # to the parser
    parser = IsprsVaiSemisup.add_model_specific_args(parser)
    parser = MeanTeacher.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    args_dict = vars(args)

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    tensorboard = loggers.TensorBoardLogger(
        save_dir=args.output_dir,
        name="tensorboard",
        version="%s_%s" % (args.exp_name, current_date)
    )

    # The lightning datamodule deals with instantiating the proper dataloaders.
    pl_datamodule = IsprsVaiSemisup(**args_dict)

    # The lightning module is where the training schema is implemented.
    pl_module = MeanTeacher(**args_dict)

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
        callbacks=pl_module.callbacks,
    )

    trainer.fit(model=pl_module, datamodule=pl_datamodule)


if __name__ == "__main__":

    main()
