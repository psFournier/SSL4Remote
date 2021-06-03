#!/bin/bash
#PBS -q qgpgpu
#PBS -l select=1:ncpus=12:mem=92G:ngpus=1
#PBS -l walltime=12:00:00

module load python/3.7.2
cd "${TMPDIR}"
mkdir miniworld_tif
CITY=austin
cp -r /work/OT/ai4geo/users/fournip/miniworld_tif/"${CITY}" miniworld_tif/

PYTHON=/home/eh/fournip/SemiSupervised/SSL4Remote/venvpython37/bin/python
SCRIPT=/home/eh/fournip/SemiSupervised/SSL4Remote/src/train.py

"${PYTHON}" "${SCRIPT}" \
--data_dir "${TMPDIR}"/miniworld_tif \
--output_dir /home/eh/fournip/SemiSupervised/SSL4Remote/outputs \
--workers 12 \
--max_epochs 500 \
--gpus 1 \
--module sup \
--datamodule mw \
--city "${CITY}" \
--train_val 31 5 \
--exp_name "${PBS_JOBNAME}"

module unload python/3.7.2