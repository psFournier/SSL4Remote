#!/bin/bash

PARAMS=$(seq 1 2) # Create an array of seed values from 1 to NSEEDS
export ROOT=/home/eh/fournip/SemiSupervised/SSL4Remote
export PROGRAM=${ROOT}/src/main.py
export INTERPRETER=${ROOT}/venv/bin/python

# lancement du programme CPU
for PARAM in ${PARAMS}
do
  export NAME=test_${PARAM}
  echo "Submitting: $NAME"
  qsub -V -N ${NAME} ${ROOT}/src/scripts/hal_venv/qsub_cpu.sh
  sleep 2
  echo "done."
done

