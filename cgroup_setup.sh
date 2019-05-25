#!/usr/bin/env bash

SCRIPT_PATH=$(dirname "$0")
source $SCRIPT_PATH/compute_resource.sh

# Get the number of numa Nodes
NODE_COUNT=$(ls -d /sys/devices/system/node/node* | wc -l)
MEM_CONSTRAINT=$(ls -d  /sys/devices/system/node/node* | grep -Po '[0-9].*' | awk '{print $1}' | paste -s -d, -)

#if [[ $NODES == 1 ]]; then
#    MEM_CONSTRAINT=0
#else
#    MEM_CONSTRAINT=0-$(($NODES - 1))
#fi

# -----------------------------
echo $CPU_COUNT
CPU_CONSTRAINT="0-$(($CPU_COUNT - 1))"


if [[ ! -f /sys/fs//cgroup/memory/student/memory.limit_in_bytes ]]; then

    # make users able to manage cgroup
    # so we can run cgcreate as user and not sudo later on
    sudo chmod 777 -R /sys/fs/cgroup/*

    # Bound the resource for a student
    cgdelete memory:all 2> /dev/null
    cgdelete cpuset,memory:student 2> /dev/null

    cgcreate -a $USER:$USER -t $USER:$USER -g memory:all
    cgcreate -a $USER:$USER -t $USER:$USER -g cpuset,memory:student

    echo
    echo "Total Device $DEVICE_TOTAL"
    echo "Total Memory $RAM_CONSTRAINT / $RAM_TOTAL"
    echo "Total CPU    $CPU_CONSTRAINT / $CPU_TOTAL"
    echo "Total Numa   $MEM_CONSTRAINT"

    echo "cgroup config"
    echo "-------------"
    echo "cgcreate -g cpuset,memory:student"
    echo "cgset -r cpuset.cpus=$CPU_CONSTRAINT student"
    echo "cgset -r cpuset.mems=$MEM_CONSTRAINT student"
    echo "cgset -r memory.limit_in_bytes=${RAM_CONSTRAINT}k student"

    cgset -r cpuset.cpus=$CPU_CONSTRAINT student
    cgset -r cpuset.mems=$MEM_CONSTRAINT student
    cgset -r memory.limit_in_bytes=${RAM_CONSTRAINT}k student

    echo ---

    cgexec -g memory:all echo "all group is working"
    cgexec -g cpuset,memory:student echo "student group is working"

    echo ---

    # Takes a while on a 2To RAM machine
    #code="import numpy as np; a = np.ones(($RAM_CONSTRAINT * 1024, 2), dtype=np.uint8); b = a + a;"
    #echo "$code"
    #echo ----
    #python -c "$code" 2> /dev/null

    # Should work
    #echo $? == 0

    # Should get killed because using twice as much memory as allowed
    #cgexec -g cpuset,memory:student python -c "$code" 2> /dev/null
    #echo $? == 137
    #echo ----
fi
