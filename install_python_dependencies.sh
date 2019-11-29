#!/bin/bash

set -ex

echo 'Install Python packages'
pip install Cython==0.29.14
pip install numpy==1.17.4
pip install --no-deps -r requirements.txt
echo 'DONE: Install Python packages'
