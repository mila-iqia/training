#!/bin/bash

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/../config.env
set -e

if [ ! -f ${DATA_DIRECTORY}/bsds500.cache ]; then
    mkdir -p ${DATA_DIRECTORY}/bsds500

    wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz -O ${TEMP_DIRECTORY}/BSR_bsds500.tgz

    tar -xzf ${TEMP_DIRECTORY}/BSR_bsds500.tgz -C ${DATA_DIRECTORY}/bsds500

    rm ${TEMP_DIRECTORY}/BSR_bsds500.tgz
    touch ${DATA_DIRECTORY}/bsds500.cache

else
    echo 'Skipping downloading'
fi





