#!/usr/bin/env bash

# ~/datasets/ImageNetFake/train/ --arch resnet50 --lr 0.001 --opt-level O0 -b 64

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/../../../config.env

SCRIPT=${SCRIPT_PATH}/conv_distributed.py

echo $MASTER_PORT

$EXEC python -m torch.distributed.launch --nproc_per_node=$DEVICE_TOTAL $SCRIPT "$@"
