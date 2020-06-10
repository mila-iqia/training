ARG FROM_IMAGE_NAME=nvcr.io/nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
FROM ${FROM_IMAGE_NAME}

# Install dependencies for system configuration logger
RUN apt-get update && apt-get install -y --no-install-recommends --allow-unauthenticated \
        infiniband-diags \
	git \
	vim \
	cgroup-bin \
	cgroup-lite \
        pciutils && \
    rm -rf /var/lib/apt/lists/*

# Clone MILA benchmarks
WORKDIR /workspace/training
COPY . .

# Install dependencies
RUN ./install_dependencies.sh 
RUN ./install_conda.sh

SHELL ["conda", "run", "-n", "mlperf", "/bin/bash", "-c"]
ENV PATH=/root/anaconda3/envs/mlperf/bin:/root/anaconda3/bin:$PATH
RUN ./install_python_dependencies.sh
RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch -y

# Configure environment variables
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

