#!/bin/bash

HOME=/d/pfournie/semi-supervised-learning
PYTHON="$HOME"/venv/bin/python
SCRIPT="$HOME"/examples/miniworld/predict.py

"${PYTHON}" "${SCRIPT}" \
--ckpt_path "$HOME"/outputs/christchurch/christchurch_aug/version_0/epoch=400-step=125200.ckpt \
--gpus 1 \
--image_path /scratch_ai4geo/miniworld_tif/vienna/train/0_x.tif \
--label_path /scratch_ai4geo/miniworld_tif/vienna/train/0_y.tif \
--output_path /scratch_ai4geo/miniworld_tif/vienna/train/0_x_pred.tif \
--workers 6 \
--batch_size 16 \
--tile_size 128 \
--tile_step 128
