#!/bin/bash
#PBS -q qgpgpu
#PBS -l select=1:ncpus=4:mem=16G:ngpus=1
#PBS -l walltime=1:00:00

cd "${TMPDIR}"
export LOGDIR=${TMPDIR}/outputs

bash ${ROOT}/src/scripts/command.sh

# recopie des donnees de sortie à conserver
cp -r "${LOGDIR}" "${ROOT}"
