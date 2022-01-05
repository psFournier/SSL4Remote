#!/bin/bash
#PBS -N tmp_semisup
#PBS -l select=1:ncpus=1:mem=4000mb
#PBS -l walltime=01:00:00



# AI4Geo Jupyter container
export SINGULARITY_IMAGE_VERSION="latest" #current"
source /softs/projets/ai4geo/code/nbremote-hpc/resources/setup/nbremote-env.sh
source /softs/projets/ai4geo/code/nbremote-hpc/resources/setup/nbremote-aliases.sh
source /softs/projets/ai4geo/code/nbremote-hpc/resources/setup/nbremote.sh

SINGULARITY_CONTAINER_BASE="vreai4geo"

SING_CMD="${SINGULARITY_BIN} exec --nv --add-caps CAP_NET_BIND_SERVICE --bind ${SINGULARITY_BIND_OPTS} ${SINGULARITY_CONTAINER_PATH}/${SINGULARITY_CONTAINER_BASE}-${SINGULARITY_IMAGE_VERSION}.simg "


echo "##################################"
echo "$SING_CMD"
echo "##################################"
echo "params to pbs_train"
echo "${PARAMS}"
echo "##################################"

module load singularity
$SING_CMD ~/SemiSupervised/SSL4Remote/scripts/vre/train.sh ${PARAMS}

