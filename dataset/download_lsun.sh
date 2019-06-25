#!/bin/bash

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/../config.env
set -e

if [ ! -f ${DATA_DIRECTORY}/ImageNet.cache ]; then
    mkdir -p ${DATA_DIRECTORY}/lsun

    python2 "$(dirname "$0")"/download_lsun.py -o ${DATA_DIRECTORY}/lsun

    touch ${DATA_DIRECTORY}/lsun.cache
else
    echo 'DONE'
fi
