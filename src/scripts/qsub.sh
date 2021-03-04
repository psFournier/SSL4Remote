#!/bin/bash

cd ~/SemiSupervised
qsub -v PARAMS=" " ~/SemiSupervised/src/pbs/pbs_train.sh
