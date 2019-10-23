#!/bin/bash

for i in {0..10}; do
    ./run.sh --uid ${i} "$@"

    mkdir -p $BASE/output${i}
    mv -f $BASE/output/* $BASE/output${i}
    sleep 10
done


