#!/bin/bash

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/config.env

if [ ! -f dependencies.cache ]; then
    # Install miniconda
    wget -nc https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -P $TEMP_DIRECTORY
    
    chmod +x $TEMP_DIRECTORY/Miniconda3-latest-Linux-x86_64.sh

    $TEMP_DIRECTORY/Miniconda3-latest-Linux-x86_64.sh -b

    source ~/.bashrc

    conda create -n mlperf python=3.6
    
    echo 'DONE'
    touch dependencies.cache
fi

apt-get install -y $(cat ${SCRIPT_PATH}/apt-packages)

# Install the perf library
conda activate mlperf
pip install -e common
pip install Cython
pip install --no-deps -r requirements.txt
