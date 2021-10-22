#!/bin/bash
#PBS -q qgpgpu
#PBS -l select=1:ncpus=6:mem=92G:ngpus=1
#PBS -l walltime=12:00:00

module load python/3.7.2
cd "${TMPDIR}"

PYTHON=/home/eh/fournip/SemiSupervised/SSL4Remote/venvpython37/bin/python
SCRIPT=/home/eh/fournip/SemiSupervised/SSL4Remote/dl_toolbox/examples/train.py

"${PYTHON}" "${SCRIPT}" \
--output_dir /work/OT/ai4geo/users/fournip/outputs \
--workers 6 \
--max_epochs 300 \
--num_classes 2 \
--gpus 1 \
--module sup \
--datamodule miniworld_generalisation \
--cities christchurch \
--data_dir /work/OT/ai4geo/users/fournip/miniworld_tif \
--exp_name "${PBS_JOBNAME}"

module unload python/3.7.2