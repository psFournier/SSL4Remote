#!/bin/bash
#PBS -q qgpgpu
#PBS -l select=1:ncpus=4:mem=16G:ngpus=1
#PBS -l walltime=1:00:00

cd "${TMPDIR}"
cp -a /work/OT/ai4geo/DATA/REF/${DATASET} .
LOGDIR=${TMPDIR}/outputs

"${INTERPRETER}" "${PROGRAM}" --exp_name "${NAME}" --data_dir "${TMPDIR}" --unsup_loss_prop "${PARAM}" --nb_pass_per_epoch 100 --output_dir "${LOGDIR}" --check_val_every_n_epoch 1 --max_epochs 20 --weights_summary full --multiple_trainloader_mode max_size_cycle --log_every_n_steps 10 --gpus 1

# recopie des donnees de sortie à conserver
cp -r "${LOGDIR}" "${ROOT}"
