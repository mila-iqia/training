#!/bin/bash

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/../../../config.env

export DATASET_DIR=$DATA_DIRECTORY/wmt16
export RESULTS_DIR=$OUTPUT_DIRECTORY/translator

mkdir -p $RESULTS_DIR

python $SCRIPT_PATH/train.py "$@"
