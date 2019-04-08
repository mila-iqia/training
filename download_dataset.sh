#!/bin/bash

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/config.env

python ${SCRIPT_PATH}/downloads_datasets.py
