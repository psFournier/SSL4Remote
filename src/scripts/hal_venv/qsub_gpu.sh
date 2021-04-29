#!/bin/bash
#PBS -q qgpgpu
#PBS -l select=1:ncpus=4:mem=92G:ngpus=1
#PBS -l walltime=1:00:00

ROOT=/home/eh/fournip/SemiSupervised/SSL4Remote
PYTHON=${ROOT}/venv/bin/python
SCRIPT=${ROOT}/src/train.py

cd "${TMPDIR}"
cp -r /work/OT/ai4geo/users/plyera/miniworld .
LOGDIR=${TMPDIR}/outputs

"${PYTHON}" "${SCRIPT}" \
--module supervised_baseline \
--datamodule miniworld_sup \
--exp_name austin \
--data_dir "${TMPDIR}"/miniworld \
--output_dir "${LOGDIR}" \
--workers 20 \
--augmentations no \
--encoder efficientnet-b0 \
--gpus 1 \
--max_epochs 5 \
--limit_train_batches 10 \
--limit_val_batches 5 \
--city austin

# recopie des donnees de sortie à conserver
cp -r "${LOGDIR}" "${ROOT}"
