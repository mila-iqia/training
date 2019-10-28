#!/bin/bash

conda activate mlperf

pip install -e common
pip install Cython
pip install numpy
pip install --no-deps -r requirements.txt

