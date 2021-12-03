#!/bin/bash

set -ex

SCRIPT_PATH=$(dirname "$0")
TEMP_DIRECTORY=/tmp

if [ ! -f conda_install.cache ]; then
    echo 'Install miniconda'
    wget -nc https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -P $TEMP_DIRECTORY

    chmod +x $TEMP_DIRECTORY/Miniconda3-latest-Linux-x86_64.sh 

    $TEMP_DIRECTORY/Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/anaconda3

    $HOME/anaconda3/bin/conda init bash

    $HOME/anaconda3/bin/conda create -n mlperf python=3.6 -y

    touch conda_install.cache
    echo 'DONE: install miniconda'
fi
