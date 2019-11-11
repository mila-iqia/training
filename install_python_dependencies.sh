#!/bin/bash

set -ex

echo 'Install required Python dependencies'
conda activate mlperf
pip install --no-deps -r requirements.txt
echo 'DONE: Install required Python dependencies'
