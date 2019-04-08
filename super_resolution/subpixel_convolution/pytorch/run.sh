#!/bin/bash

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/../../../config.env

$EXEC python $SCRIPT_PATH/main.py "$@"

