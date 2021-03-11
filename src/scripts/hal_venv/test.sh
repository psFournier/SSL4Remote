#!/bin/bash

export ROOT=/home/eh/fournip/SemiSupervised/SSL4Remote
export PROGRAM=${ROOT}/src/main.py
export INTERPRETER=${ROOT}/venv/bin/python

# lancement du programme CPU
for PARAM in 0. 0.01 0.1
do
  export PARAM
  for SEED in 1 2 3 4 5
  do
    export NAME=param_${PARAM}_seed_${SEED}
    echo "Submitting: $NAME"
    qsub -V -N ${NAME} ${ROOT}/src/scripts/hal_venv/qsub_cpu.sh
    sleep 1
    echo "done."
  done
done

