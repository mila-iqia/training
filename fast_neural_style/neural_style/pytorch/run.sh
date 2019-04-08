#!/bin/bash

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/../../../config.env


DEFAULT_ARGS="train --dataset $DATA_DIRECTORY/ImageNet/train --style-image $SCRIPT_PATH/../../images/style-images/candy.jpg  --save-model-dir $TEMP_DIRECTORY --no-checks --image-size 64 --style-size 64"

$EXEC python $SCRIPT_PATH/neural_style.py "$@" $DEFAULT_ARGS

