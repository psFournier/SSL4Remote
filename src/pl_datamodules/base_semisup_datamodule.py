from torch.utils.data import DataLoader, RandomSampler
from pl_datamodules import BaseSupervisedDatamodule


class BaseSemisupDatamodule(BaseSupervisedDatamodule):

    def __init__(self, unsup_train, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.unsup_train_set = None
        self.unsup_train = unsup_train

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument('--unsup_train', type=int, default=0)

        return parser

    def train_dataloader(self):

        """
        The semi supervised loader consists in two loaders for labeled and
        unlabeled data.
        """

        sup_train_dataloader = super().train_dataloader()

        unsup_train_sampler = RandomSampler(
            data_source=self.unsup_train_set,
            replacement=True,
            num_samples=self.epoch_len
        )

        unsup_train_dataloader = DataLoader(
            dataset=self.unsup_train_set,
            batch_size=self.batch_size,
            sampler=unsup_train_sampler,
            # collate_fn=self.collate_and_aug,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=self.wif
        )

        train_dataloaders = {
            "sup": sup_train_dataloader,
            "unsup": unsup_train_dataloader
        }

        return train_dataloaders