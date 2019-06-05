#!/bin/bash
# This is from https://www.sylabs.io/guides/3.2/user-guide/quick_start.html#quick-installation-steps

sudo apt-get update && sudo apt-get install -y \
    build-essential \
    libssl-dev \
    uuid-dev \
    libgpgme11-dev \
    squashfs-tools \
    wget \
    git

# Install Go
# -------------------------------------------
export VERSION=1.11.10 OS=linux ARCH=amd64
temp=$(mktemp -d)

cd $temp
#    https://dl.google.com/go/go1.11.10.linux-amd64.tar.gz
wget https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz
sudo tar -C /usr/local -xzf go$VERSION.$OS-$ARCH.tar.gz

rm -rf $temp

echo 'export GOPATH=${HOME}/go' >> ~/.bashrc
echo 'export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin' >> ~/.bashrc

# Singularity
# -------------------------------------------
mkdir -p $GOPATH/src/github.com/sylabs
cd $GOPATH/src/github.com/sylabs
git clone https://github.com/sylabs/singularity.git
cd singularity

cd $GOPATH/src/github.com/sylabs/singularity
./mconfig
make -C builddir
sudo make -C builddir install
