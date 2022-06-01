#!/bin/bash

PYTHON=/d/pfournie/ai4geo/venv/bin/python3
SCRIPT=/d/pfournie/dl_toolbox/docs/examples/unet_semcity_torch.py

"${PYTHON}" "${SCRIPT}" \
--data_path /d/pfournie/ai4geo/data/SemcityTLS_DL \
--output_dir /d/pfournie/ai4geo/outputs \
--sup_batch_size 8 \
--workers 6 \
--max_epochs 10 \
--epoch_len 100 \
--crop_size 128 \
--num_classes 8 \
--img_aug d4_color-0 \
--encoder efficientnet-b1 \
--in_channels 3 \
--initial_lr 0.05 \
--final_lr 0.001 \
--lr_milestones 0.5 0.9 \
--pretrained \
--train_with_void
