#!/bin/bash

echo "#############################################################"
echo `python --version`
echo "#############################################################"
python ~/SemiSupervised/src/tests/validate_deps.py
echo "#############################################################"
echo "Params to train.sh : "
echo "$@"
echo "#############################################################"


cd ~/SemiSupervised/
python ./src/main.py $@