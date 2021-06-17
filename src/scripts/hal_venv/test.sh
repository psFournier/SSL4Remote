#!/bin/bash
#PBS -q qgpgpu
#PBS -l select=1:ncpus=6:mem=92G:ngpus=1
#PBS -l walltime=01:00:00

module load python/3.7.2

PYTHON=/home/eh/fournip/SemiSupervised/SSL4Remote/venvpython37/bin/python
SCRIPT=/home/eh/fournip/SemiSupervised/SSL4Remote/src/test.py

"${PYTHON}" "${SCRIPT}" \
--ckpt_path /work/OT/ai4geo/users/fournip/outputs/vienna_2/sup_mw_vienna_10-30_0-20_d4/version_0/checkpoints/epoch=399-step=125199.ckpt \
--store_pred \
--gpus 1 \
--image_path /work/OT/ai4geo/users/fournip/miniworld_tif/vienna/train/0_x.tif \
--label_path /work/OT/ai4geo/users/fournip/miniworld_tif/vienna/train/0_y.tif \
--output_name "${PBS_JOBNAME}" \
--workers 6 \
--batch_size 16 \
--crop_size 128 \
--crop_step 120
