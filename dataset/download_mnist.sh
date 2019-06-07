#!/bin/bash

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/../config.env
set -e

python "$(dirname "$0")"/download_mnist.py
