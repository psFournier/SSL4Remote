from torch.utils.data import DataLoader, RandomSampler
from lightning_datamodules import BaseSupervisedDatamodule
from utils import worker_init_function
from torch_collate import CustomCollate

class BaseSemisupDatamodule(BaseSupervisedDatamodule):

    def __init__(self, unsup_batch_size, unsup_crop_size, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.unsup_train_set = None
        self.unsup_batch_size = unsup_batch_size
        self.unsup_crop_size = unsup_crop_size

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument('--unsup_batch_size', type=int, default=16)
        parser.add_argument('--unsup_crop_size', type=int, default=160)

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
            batch_size=self.unsup_batch_size,
            sampler=unsup_train_sampler,
            collate_fn=CustomCollate(),
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_function
        )

        train_dataloaders = {
            "sup": sup_train_dataloader,
            "unsup": unsup_train_dataloader
        }

        return train_dataloaders
