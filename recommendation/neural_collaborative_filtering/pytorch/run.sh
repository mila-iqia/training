#!/bin/bash

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/../../../config.env

# skip if already converted
if [ ! -f $DATA_DIRECTORY/ml-20m/test-negative.csv ]; then
    $EXEC python $SCRIPT_PATH/convert.py $DATA_DIRECTORY/ml-20m/ratings.csv $DATA_DIRECTORY/ml-20m --negatives 999
fi

$EXEC python $SCRIPT_PATH/ncf.py "$@"
