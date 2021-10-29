#!/bin/bash

PYTHON=/d/pfournie/semi-supervised-learning/venv/bin/python
SCRIPT=/d/pfournie/semi-supervised-learning/dl_toolbox/examples/train.py

"${PYTHON}" "${SCRIPT}" \
--workers 10 \
--epoch_len 120000 \
--max_epochs 50 \
--lr_milestones 25 35 45 \
--encoder efficientnet-b0 \
--learning_rate 0.05 \
--img_aug no \
--batch_aug no \
--num_classes 2 \
--gpus 1 \
--module sup \
--datamodule miniworld_generalisation \
--cities christchurch \
--data_dir /scratch_ai4geo/miniworld_tif \
--output_dir /d/pfournie/semi-supervised-learning/outputs \
--exp_name airs