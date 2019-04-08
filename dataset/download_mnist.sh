#!/bin/bash

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/../config.env

python "$(dirname "$0")"/download_mnist.py
