#!/bin/bash

PYTHON=/d/pfournie/Documents/ai4geo/SemiSupervised/SSL4Remote/venv/bin/python
SCRIPT=/d/pfournie/Documents/ai4geo/SemiSupervised/SSL4Remote/dl_toolbox/examples/miniworld/train.py

"${PYTHON}" "${SCRIPT}" \
--module sup \
--datamodule mw3 \
--data_dir /d/pfournie/Documents/ai4geo/data/miniworld_tif \
--output_dir /d/pfournie/Documents/ai4geo/SemiSupervised/SSL4Remote/outputs \
--workers 6 \
--max_epochs 500 \
--gpus 1 \
--train_cities austin vienna kitsap tyrol-w \
--val_cities chicago \
--exp_name test_mw3