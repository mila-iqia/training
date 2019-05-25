#!/bin/bash

for i in {0..10}; do
    ./run.sh
    mv $BASE/output $BASE/output${i}
done


