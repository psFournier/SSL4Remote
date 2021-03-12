#!/bin/bash

export ROOT=/home/pierre/PycharmProjects/RemoteSensing
export PROGRAM=${ROOT}/src/main.py
export INTERPRETER=${ROOT}/venv/bin/python

# lancement du programme CPU
for PARAM in 0.
do
  export PARAM
  for SEED in 1
  do
    export NAME=param_${PARAM}_seed_${SEED}
    echo "Submitting: $NAME"
    ${ROOT}/src/scripts/test_local/qsub_cpu.sh
    sleep 1
    echo "done."
  done
done

