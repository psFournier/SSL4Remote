#!/bin/bash
#PBS -q qgpgpu
#PBS -l select=1:ncpus=12:mem=92G:ngpus=1
#PBS -l walltime=12:00:00

module load python/3.7.2

PYTHON=/home/eh/fournip/SemiSupervised/SSL4Remote/venvpython37/bin/python
SCRIPT=/home/eh/fournip/SemiSupervised/SSL4Remote/src/test.py

"${PYTHON}" "${SCRIPT}" \
--ckpt_path /work/OT/ai4geo/users/fournip/outputs/sup_mw_austin_31-5/version_0/checkpoints/epoch=418-step=131146.ckpt \
--store_pred \
--gpus 1 \
--image_path /work/OT/ai4geo/users/fournip/miniworld_tif/austin/train/1_x.tif \
--label_path /work/OT/ai4geo/users/fournip/miniworld_tif/austin/train/1_y.tif \
--output_name test
