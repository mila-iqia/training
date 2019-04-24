#!/bin/bash

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/config.env

if [ ! -f dependencies.cache ]; then

    sudo apt-get install -y $(cat apt-packages)

    # Install miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $TEMP_DICRETORY

    ./$TEMP_DIRECTORY/Miniconda3-latest-Linux-x86_64.sh -b

    source ~/.bashrc

    conda create -n mlperf python=3.6
    source activate mlperf

    # Install the perf library
    cd common
    python setup.py install

    echo "Install pytorch and tensorflow now"

    pip install Cython
    pip install --no-deps -r requirements.txt
   
    echo 'DONE'
    touch dependencies.cache
fi
