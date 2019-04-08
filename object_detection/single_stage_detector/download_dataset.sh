#!/bin/bash

# download pre-trained model
export RESNET34=$(python -c "from torchvision.models.resnet import resnet34; resnet34(pretrained=True); print(1)")
export RESNET18=$(python -c "from torchvision.models.resnet import resnet18; resnet18(pretrained=True); print(1)")

./"$(dirname "$0")"/../../dataset/download_coco.sh
