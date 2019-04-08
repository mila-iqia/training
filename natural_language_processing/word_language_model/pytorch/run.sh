#!/bin/bash

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/../../../config.env

OLD_WD=$(pwd)
EPOCHS=5
DATA=$SCRIPT_PATH/../data/wikitext-2

$EXEC python $SCRIPT_PATH/main.py --data $DATA "$@"

