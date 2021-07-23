#!/bin/bash
#PBS -q qgpgpu
#PBS -l select=1:ncpus=6:mem=92G:ngpus=1
#PBS -l walltime=12:00:00

module load python/3.7.2
cd "${TMPDIR}"
#mkdir miniworld_tif
#CITY=austin
#cp -r /work/OT/ai4geo/users/fournip/miniworld_tif/"${CITY}" miniworld_tif/

PYTHON=/home/eh/fournip/SemiSupervised/SSL4Remote/venvpython37/bin/python
SCRIPT=/home/eh/fournip/SemiSupervised/SSL4Remote/dl_toolbox/examples/miniworld/train.py

"${PYTHON}" "${SCRIPT}" \
--data_dir "${TMPDIR}"/miniworld_tif \
--output_dir /work/OT/ai4geo/users/fournip/outputs \
--workers 6 \
--max_epochs 500 \
--gpus 1 \
--module sup \
--datamodule mw2 \
--cities christchurch \
--train_dataset_transforms_strat hard \
--exp_name "${PBS_JOBNAME}"

module unload python/3.7.2