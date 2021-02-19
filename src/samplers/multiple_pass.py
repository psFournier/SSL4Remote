from torch.utils.data import RandomSampler

class Multiple_pass(RandomSampler):

    def __init__(self, data_source, nb_pass_per_epoch):

        super(Multiple_pass, self).__init__(
            data_source=data_source,
            replacement=True,
            num_samples=nb_pass_per_epoch*len(data_source)
        )
