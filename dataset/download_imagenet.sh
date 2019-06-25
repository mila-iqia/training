#!/bin/bash

SCRIPT_PATH=$(dirname "$0")
source ${SCRIPT_PATH}/../config.env

set -e

if [ ! -f ${DATA_DIRECTORY}/ImageNet.cache ]; then
    touch ${DATA_DIRECTORY}/ImageNet.cache

    #mkdir -p ${DATA_DIRECTORY}/ImageNet/test
    #mkdir -p ${DATA_DIRECTORY}/ImageNet/train
    #mkdir -p ${DATA_DIRECTORY}/ImageNet/val

    if [ $FAKE_DATASET == 1 ]; then

        mkdir -p ${DATA_DIRECTORY}/ImageNet/train
        python ${SCRIPT_PATH}/fake_imagenet.py --output ${DATA_DIRECTORY}/ImageNet/train --size 512 --class-num 1000 --batch-size 256 --number 100 --repeat 4

        rm -rf ${DATA_DIRECTORY}/ImageNet/val
        rm -rf ${DATA_DIRECTORY}/ImageNet/test

        ln -s ${DATA_DIRECTORY}/ImageNet/train ${DATA_DIRECTORY}/ImageNet/test
        ln -s ${DATA_DIRECTORY}/ImageNet/train ${DATA_DIRECTORY}/ImageNet/val

        touch ${DATA_DIRECTORY}/ImageNet.cache

        #python ${SCRIPT_PATH}/fake_imagenet.py --output ${DATA_DIRECTORY}/ImageNet/val --size 512 --class-num 1000 --batch-size 256 --number 20 --repeat 4
        #python ${SCRIPT_PATH}/fake_imagenet.py --output ${DATA_DIRECTORY}/ImageNet/train --size 512 --class-num 1000 --batch-size 256 --number 20 --repeat 4
    else
        if [ ! $USE_VALIDATION ]; then
            mkdir -p ${DATA_DIRECTORY}/ImageNet/train

            wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar -O ${TEMP_DIRECTORY}/ILSVRC2012_img_train.tar
            tar -xvf ${TEMP_DIRECTORY}/ILSVRC2012_img_train.tar -C ${DATA_DIRECTORY}/ImageNet/train
            rm ${TEMP_DIRECTORY}/ILSVRC2012_img_train.tar
        fi

        wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar -O ${TEMP_DIRECTORY}/ILSVRC2012_img_val.tar
        tar -xvf ${TEMP_DIRECTORY}/ILSVRC2012_img_val.tar -C ${DATA_DIRECTORY}/ImageNet/val
        rm ${TEMP_DIRECTORY}/ILSVRC2012_img_val.tar

        #wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_test.tar -O ${TEMP_DIRECTORY}/ILSVRC2012_img_test.tar
        #tar -xvf ${TEMP_DIRECTORY}/ILSVRC2012_img_test.tar -C ${DATA_DIRECTORY}/ImageNet/test
        #rm ${TEMP_DIRECTORY}/ILSVRC2012_img_test.tar

        # Use the validation only
        if [ $USE_VALIDATION ]; then
            rm -rf ${DATA_DIRECTORY}/ImageNet/train
            mkdir -p ${DATA_DIRECTORY}/ImageNet/val
            ln -s ${DATA_DIRECTORY}/ImageNet/val ${DATA_DIRECTORY}/ImageNet/train
        fi
    fi
else
    echo 'Skipping downloading'
fi
