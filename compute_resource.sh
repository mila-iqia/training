#!/usr/bin/env bash

# System Stats
export DEVICE_TOTAL=0
export DEVICE_TOTAL=$(python -c "import torch; print(torch.cuda.device_count())")
export CPU_TOTAL=$(nproc)
export RAM_TOTAL=$(cat /proc/meminfo | head -n 1 | grep -oP '[0-9]*')
export USE_CUDA=$(python -c "import torch; print(int(torch.cuda.is_available()))")
# ---------------

if [[ DEVICE_TOTAL == 0 ]]; then
    export RAM_CONSTRAINT=$(($RAM_TOTAL / $DEVICE_TOTAL))
    export CPU_COUNT=$(($CPU_TOTAL / $DEVICE_TOTAL))
else
    export RAM_CONSTRAINT=$(($RAM_TOTAL / 8))
    export CPU_COUNT=$(($CPU_TOTAL / 2))
fi
