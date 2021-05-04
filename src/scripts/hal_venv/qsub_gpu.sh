#!/bin/bash
#PBS -q qgpgpu
#PBS -l select=1:ncpus=8:mem=92G:ngpus=1
#PBS -l walltime=1:00:00

cd "${TMPDIR}"
mkdir miniworld
cp -r /work/OT/ai4geo/users/plyera/miniworld/austin miniworld/

export ROOTDIR=/home/eh/fournip/SemiSupervised/SSL4Remote
export LOGDIR=${TMPDIR}/outputs
export DATADIR="${TMPDIR}"/miniworld

bash ${ROOTDIR}/src/scripts/train.sh

# recopie des donnees de sortie à conserver
cp -r "${LOGDIR}" "${ROOTDIR}"
