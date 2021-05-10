#!/bin/bash
#PBS -q qgpgpu
#PBS -l select=1:ncpus=12:mem=92G:ngpus=1
#PBS -l walltime=12:00:00

cd "${TMPDIR}"
mkdir miniworld
cp -r /work/OT/ai4geo/users/fournip/miniworld_tif/austin miniworld_tif/

PYTHON=/home/eh/fournip/SemiSupervised/SSL4Remote/venv/bin/python
SCRIPT=/home/eh/fournip/SemiSupervised/SSL4Remote/src/train.py

"${PYTHON}" "${SCRIPT}" \
--module supervised_baseline \
--datamodule miniworld_sup \
--data_dir "${TMPDIR}"/miniworld_tif \
--output_dir ${TMPDIR}/outputs \
--workers 12 \
--max_epochs 500 \
--gpus 1 \
--city austin \
--train_val 2 5 \
--tta_augment hsv contrast

# recopie des donnees de sortie à conserver
cp -r ${TMPDIR}/outputs /home/eh/fournip/SemiSupervised/SSL4Remote