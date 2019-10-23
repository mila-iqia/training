#!/bin/bash


set -e

SCRIPT_PATH=$(dirname "$0")
# source ${SCRIPT_PATH}/config.env



TEMP_DIRECTORY=/tmp

if [ ! -f dependencies.cache ]; then
    # Install miniconda
    wget -nc https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -P $TEMP_DIRECTORY
    
    chmod +x $TEMP_DIRECTORY/Miniconda3-latest-Linux-x86_64.sh 

    $TEMP_DIRECTORY/Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/anaconda3

    $HOME/anaconda3/bin/conda init

    $HOME/anaconda3/bin/conda create -n mlperf python=3.6
    
    echo 'DONE'
    touch dependencies.cache
fi


sudo apt-get install -y $(cat ${SCRIPT_PATH}/apt_packages)

# Install the perf library
$HOME/anaconda3/bin/conda activate mlperf

pip install -e common
pip install Cython
pip install --no-deps -r requirements.txt

exec bash
