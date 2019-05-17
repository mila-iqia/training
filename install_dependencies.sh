#!/bin/bash

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/config.env

if [ ! -f dependencies.cache ]; then
    # Install miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $TEMP_DICRETORY

    ./$TEMP_DIRECTORY/Miniconda3-latest-Linux-x86_64.sh -b

    source ~/.bashrc

    conda create -n mlperf python=3.6
    source activate mlperf

    echo 'DONE'
    touch dependencies.cache
fi

sudo apt-get install -y $(cat apt-packages)

# Install the perf library
pip install -e common
pip install Cython
pip install --no-deps -r requirements.txt
