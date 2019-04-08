#!/bin/bash

ORIGINAL_ARGS="$@"

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/../../../config.env

$EXEC python $SCRIPT_PATH/main.py $ORIGINAL_ARGS $MORE_ARGS
