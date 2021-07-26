#!/bin/bash

HOME=/d/pfournie/semi-supervised-learning
PYTHON="$HOME"/venv/bin/python
SCRIPT="$HOME"/dl_toolbox/examples/miniworld/test.py

"${PYTHON}" "${SCRIPT}" \
--module sup \
--ckpt_path "$HOME"/outputs/sup_mw_austin_31-5/version_0/checkpoints/epoch=418-step=131146.ckpt \
--gpus 1 \
--workers 6 \
--data_dir /scratch_ai4geo/miniworld_tif \
--city vienna \
--tile_size 128 \
--batch_size 32