#!/bin/bash

for i in {0..9}; do
    ./run.sh --uid ${i} "$@"
    sleep 10
done
