#!/bin/bash

export VGG16=$(python -c "from torchvision import models; models.vgg16(pretrained=True); print(1)")

./"$(dirname "$0")"/../dataset/download_coco.sh
