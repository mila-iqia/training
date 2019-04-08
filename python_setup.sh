#!/bin/bash

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/config.env

sudo apt-get install -y $(cat apt-packages)

# Install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $TEMP_DICRETORY

./$TEMP_DIRECTORY/Miniconda3-latest-Linux-x86_64.sh -b

# reload environment
source ~/.bashrc

conda create -n mlperf python=3.6

source activate mlperf
# Virtual env setup

# Install the perf library
cd common
python setup.py install

echo "Install pytorch and tensorflow now"

conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

pip install cython

pip install -r $SCRIPT_PATH/requirements.txt

./$SCRIPT_PATH/dependencies/install_warp_ctc.sh

echo 'DONE'
