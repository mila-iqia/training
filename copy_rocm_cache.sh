#!/bin/bash

SCRIPT_PATH=$(dirname "$0")

# Copy the compilation cache
unzip ${SCRIPT_PATH}/common/miopen.zip -d .cache/

# Copy the performance database
cp ${SCRIPT_PATH}/common/gfx906_60.HIP.2_1_0.ufdb.txt ~/.config/miopen/

