#!/bin/bash

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/config.env

cd $SCRIPT_PATH

# ./install_dependencies.sh

export CONFIGURED=1

./cgroup_setup.sh

./download_dataset.sh

python run.py "$@"

