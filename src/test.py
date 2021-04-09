import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
from torch_datasets import MiniworldParisLabeled
from torch_datasets import MiniworldArlingtonLabeled

city_classes = [MiniworldParisLabeled, MiniworldArlingtonLabeled]
city_directories = [
    '/scratch_ai4geo/miniworld/'+name for name in [
        'paris',
        'Arlington'
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
print(len(dataset))

sup_train_loader = DataLoader(
    dataset,
    shuffle=False,
    num_workers=0,
    batch_size=2
)

for data in sup_train_loader:
    print(data)