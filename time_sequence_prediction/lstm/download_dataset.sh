#!/bin/bash

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/../../config.env
cd $SCRIPT_PATH

python generate_sine_wave.py
