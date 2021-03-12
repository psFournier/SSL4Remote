#!/bin/bash
#PBS -l select=1:ncpus=1:mem=4000mb
#PBS -l walltime=01:00:00

cd "${TMPDIR}"
cp -a /work/OT/ai4geo/DATA/REF/ISPRS_VAIHINGEN .
LOGDIR=${TMPDIR}/outputs

"${INTERPRETER}" "${PROGRAM}" --data_dir "${TMPDIR}" --unsup_loss_prop ${PARAM} --nb_pass_per_epoch 10 --output_dir "${LOGDIR}" --check_val_every_n_epoch 1 --max_epochs 20 --weights_summary full --multiple_trainloader_mode max_size_cycle --log_every_n_steps 10

# recopie des donnees de sortie à conserver
cp -r "${LOGDIR}" "${ROOT}"