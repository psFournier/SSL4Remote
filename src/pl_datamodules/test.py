import numpy as np
from datasets import MiniworldParisLabeled, MiniworldArlingtonLabeled
from torch.utils.data import ConcatDataset, DataLoader


city_classes = [
    getattr('datasets', name) for name in [
        'MiniworldParisLabeled',
        'MiniworldArlingtonLabeled'
    ]
]
city_directories = [
    '/scratch_ai4geo/miniworld/'+name for name in [
        'paris',
        'arlington'
    ]
]

datasets = []
for city_class, directory in zip(city_classes, city_directories):

    shuffled_idxs = np.random.permutation(
        len(city_class.labeled_image_paths)
    )

    idxs = shuffled_idxs[:2]

    datasets.append(
        city_class(directory, idxs, 128)
    )

dataset = ConcatDataset(datasets)

sup_train_loader = DataLoader(
    dataset,
    shuffle=False,
    num_workers=0,
    batch_size=2
)