#!/bin/bash

export ROOT=/home/eh/fournip/SemiSupervised/SSL4Remote
export DATADIR=/work/OT/ai4geo/DATA/DATASETS/ISPRS_VAIHINGEN
#export LOGDIR=$ROOT/outputs

export PROGRAM=${ROOT}/src/main_scripts/mean_teacher.py


for SEED in 1 2
do
  export NAME=MT_ISPRS_VAI_seed_${SEED}
  echo "Submitting: $NAME"
  qsub -V -N ${NAME} ${ROOT}/src/scripts/hal_venv/qsub_gpu.sh
  sleep 1
  echo "done."
done

