#!/bin/bash

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/config.env

cd $SCRIPT_PATH

# ./install_dependencies.sh

export CONFIGURED=1

# export the base environment variable to singularity
export SINGULARITYENV_BASE=$BASE

./cgroup_setup.sh

download_datasets.sh "$@"

python run.py "$@"

