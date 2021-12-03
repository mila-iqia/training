#!/bin/bash

set -ex

SCRIPT_PATH=$(dirname "$0")

SUDO=""

if [[ "$EUID" -ne 0 ]]; then
    SUDO="sudo"
fi

echo 'Install required apt packages'
$SUDO apt-get update
$SUDO apt-get install -y $(cat ${SCRIPT_PATH}/apt_packages)
echo 'DONE: Install required apt packages'
