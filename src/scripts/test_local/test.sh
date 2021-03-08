#!/bin/bash

PARAMS=$(seq 1 2) # Create an array of seed values from 1 to NSEEDS
export ROOT=/home/pierre/PycharmProjects/RemoteSensing
export PROGRAM=${ROOT}/src/main.py
export LOGDIR=${ROOT}/outputs
export INTERPRETER=${ROOT}/venv/bin/python

# lancement du programme CPU
for PARAM in ${PARAMS}
do
  export NAME=test_${PARAM}
  echo "Submitting: $NAME"
  ${ROOT}/src/scripts/test_local/qsub_cpu.sh
  sleep 2
  echo "done"
done

