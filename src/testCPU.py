import torch
import numpy as np
from torch import nn
import torch.autograd.profiler as profiler
from networks import Unet
import rasterio
from rasterio.windows import Window
import os
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, RandomSampler
from datasets import Isprs_labeled
from torch.optim import Adam
import torch.nn.functional as F


def get_crop_window(dataset, crop):
    cols = dataset.width
    rows = dataset.height
    cx = np.random.randint(0, cols - crop - 1)
    cy = np.random.randint(0, rows - crop - 1)
    w = Window(cx, cy, crop, crop)

    return w


data_path = '/home/pierre/Documents/ONERA/ai4geo'
# Surface model
# dsm_filepath = os.path.join(data_path, 'dsm',
#                             'dsm_09cm_matching_area{}.tif'.format(3))
# # True orthophoto
# top_filepath = os.path.join(data_path, 'top',
#                             'top_mosaic_09cm_area{}.tif'.format(3))
#
# with rasterio.open(dsm_filepath) as dsm_dataset:
#     w = get_crop_window(dsm_dataset, 128)
#     dsm = dsm_dataset.read(
#         window=w, out_dtype=np.float32
#     ).transpose(1, 2, 0) / 255
#
# with rasterio.open(top_filepath) as top_dataset:
#     w = get_crop_window(top_dataset, 128)
#     top = top_dataset.read(
#         window=w, out_dtype=np.float32
#     ).transpose(1, 2, 0) / 255
#
# transfo = ToTensorV2()
#
# input = transfo(image=np.concatenate((top, dsm), axis=2))['image']

sup_train_set = Isprs_labeled(data_path,
                              [1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28],
                              128,
                              ToTensorV2())

sup_train_sampler = RandomSampler(
    data_source=sup_train_set,
    replacement=True,
    num_samples=len(sup_train_set)
)
sup_train_dataloader = DataLoader(
    dataset=sup_train_set,
    batch_size=16,
    sampler=sup_train_sampler,
    num_workers=2,
    pin_memory=True
)



network = Unet(4,2)
optimizer = Adam(network.parameters(), lr=0.01)

network.train(True)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    for data in sup_train_dataloader:

        train_inputs, train_labels = data
        optimizer.zero_grad()
        outputs = network(train_inputs)
        supervised_loss = F.cross_entropy(outputs, train_labels)
        rotation_1, rotation_2 = np.random.choice(
            [0, 1, 2, 3],
            size=2,
            replace=False
        )
        augmented_1 = torch.rot90(train_inputs, k=rotation_1, dims=[2, 3])
        augmented_2 = torch.rot90(train_inputs, k=rotation_2, dims=[2, 3])
        outputs_1 = network(augmented_1)
        outputs_2 = network(augmented_2)
        unaugmented_1 = torch.rot90(outputs_1, k=-rotation_1, dims=[2, 3])
        unaugmented_2 = torch.rot90(outputs_2, k=-rotation_2, dims=[2, 3])

        unsupervised_loss = F.mse_loss(
            unaugmented_1,
            unaugmented_2
        )


        # For now, test supervised learning
        # total_loss = supervised_loss
        total_loss = supervised_loss + unsupervised_loss

        total_loss.backward()
        optimizer.step()

print(
    prof.key_averages(group_by_stack_n=5).table(
        sort_by='cpu_memory_usage'
    )
)
