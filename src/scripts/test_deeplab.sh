#!/bin/bash

HOME=/d/pfournie/semi-supervised-learning
PYTHON="$HOME"/venv/bin/python
SCRIPT="$HOME"/src/test.py

"${PYTHON}" "${SCRIPT}" \
--ckpt_path "$HOME"/outputs/sup_mw_austin_31-5/version_0/checkpoints/epoch=418-step=131146.ckpt \
--data_dir /scratch_ai4geo/miniworld_tif \
--gpus 1 \
--image_path /scratch_ai4geo/miniworld_tif/austin/train/1_x.tif \
--label_path /scratch_ai4geo/miniworld_tif/austin/train/1_y.tif \
--with_swa \
--img_aug d4 hue \
--tta d4