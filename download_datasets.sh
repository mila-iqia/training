#!/bin/bash

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/config.env

export SINGULARITYENV_BASE=$BASE
python ${SCRIPT_PATH}/download_datasets.py "$@"
