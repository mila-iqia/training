#!/bin/bash

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/../config.env
set -e

if [ ! -f ${DATA_DIRECTORY}/ml-20m.cache ]; then
    touch ${DATA_DIRECTORY}/ml-20m.cache

    wget http://files.grouplens.org/datasets/movielens/ml-20m.zip -O ${TEMP_DIRECTORY}/ml-20m.zip
    unzip -u ${TEMP_DIRECTORY}/ml-20m.zip -d ${DATA_DIRECTORY}
    rm ${TEMP_DIRECTORY}/ml-20m.zip
else
    echo 'Skipping downloading'
fi

