#!/bin/bash

REPO=/home/pfournie/semi-supervised-learning
PYTHON="$REPO"/venv/bin/python
SCRIPT="$REPO"/dl_toolbox/predict.py

"${PYTHON}" "${SCRIPT}" \
--ckpt_path "$REPO"/outputs/test_semcity/version_0/checkpoints/epoch=5-step=5.ckpt \
--image_path /home/pfournie/ai4geo/data/SemcityTLS_DL/val/TLS_BDSD_M_04.tif \
--label_path /home/pfournie/ai4geo/data/SemcityTLS_DL/val/TLS_GT_04.tif \
--output_path "$REPO"/outputs/test_semcity/version_0/pred_04.tif \
--workers 0 \
--batch_size 16 \
--tile_size 128 128 \
--tile_step 128 128
