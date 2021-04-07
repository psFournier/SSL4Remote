#!/bin/bash

export ROOT=/home/pierre/PycharmProjects/RemoteSensing
export DATADIR=/home/pierre/Documents/ONERA/ai4geo/ISPRS_VAIHINGEN
export LOGDIR=$ROOT/outputs

export PROGRAM=${ROOT}/src/mean_teacher.py


for SEED in 1
do
  export NAME=MT_ISPRS_VAI_seed_${SEED}
  echo "Launching expe num $SEED"
  bash ${ROOT}/src/scripts/command.sh
  sleep 1
  echo "done."
done

