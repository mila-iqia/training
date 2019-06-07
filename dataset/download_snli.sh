#!/bin/bash

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/../config.env
set -e

python "$(dirname "$0")"/download_snli.py


#
#if [ ! -f ${DATA_DIRECTORY}/snli.cache ]; then

#    wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip -O ${TEMP_DIRECTORY}/snli_1.0.zip

#    unzip -u ${TEMP_DIRECTORY}/snli_1.0.zip -d ${DATA_DIRECTORY}/snli

#    touch ${DATA_DIRECTORY}/snli.cache
#else
#    echo 'DONE'
#fi
