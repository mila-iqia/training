#!/bin/bash

set -ex

SCRIPT_PATH=$(dirname "$0")

echo 'Install required apt packages'
sudo apt-get update
sudo apt-get install -y $(cat ${SCRIPT_PATH}/apt_packages)
echo 'DONE: Install required apt packages'
