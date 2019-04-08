#!/bin/bash

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/../config.env

mkdir -p ${DATA_DIRECTORY}/coco


if [ ! -f ${DATA_DIRECTORY}/coco2017.cache ]; then
    touch ${DATA_DIRECTORY}/coco2017.cache

    wget http://images.cocodataset.org/zips/train2017.zip -O ${TEMP_DIRECTORY}/train2017.zip
    unzip -u ${TEMP_DIRECTORY}/train2017.zip -d ${DATA_DIRECTORY}/coco
    rm ${TEMP_DIRECTORY}/train2017.zip

    wget http://images.cocodataset.org/zips/val2017.zip -O ${TEMP_DIRECTORY}/val2017.zip
    unzip -u ${TEMP_DIRECTORY}/val2017.zip -d ${DATA_DIRECTORY}/coco
    rm ${TEMP_DIRECTORY}/val2017.zip

    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O ${TEMP_DIRECTORY}/annotations_trainval2017.zip
    unzip -u ${TEMP_DIRECTORY}/annotations_trainval2017.zip -d ${DATA_DIRECTORY}/coco
    rm ${TEMP_DIRECTORY}/annotations_trainval2017.zip


else
    echo 'Skipping downloading'
fi



