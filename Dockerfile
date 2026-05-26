FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel

# Optional cleanup of old sources
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

# Install gnupg (for apt-key)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg && rm -rf /var/lib/apt/lists/*

RUN apt-key del 7fa2af80 || true
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN sed -i \
    -e 's|http://archive.ubuntu.com/ubuntu|https://ftp.kaist.ac.kr/ubuntu|g' \
    -e 's|http://security.ubuntu.com/ubuntu|https://ftp.kaist.ac.kr/ubuntu|g' \
    /etc/apt/sources.list

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    libgl1-mesa-glx \
    libgtk2.0-dev \
    libopenexr-dev \
    build-essential \
    cmake \
    git \
    ninja-build \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Set environment variable for CUDA
ENV CUDA_HOME=/usr/local/cuda

# Upgrade pip
RUN pip install --upgrade pip

# Install Python packages (PyTorch 2.3.0 + Torch-TensorRT 2.3.0)
RUN pip install torch==2.8.0 torchvision==0.23.0 --extra-index-url https://download.pytorch.org/whl/cu126 && \
    pip install torch-tensorrt==2.8.0 && \
    pip install "numpy<2" PyEXR==0.3.9 opencv-python \
    parmap OpenEXR timm einops typing_extensions

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake && \
    rm -rf /var/lib/apt/lists/*
RUN pip install mitsuba
RUN apt-get update && apt-get install -y sudo
WORKDIR /tmp
RUN wget https://github.com/OpenImageDenoise/oidn/releases/download/v2.4.1/oidn-2.4.1.x86_64.linux.tar.gz
RUN tar -xzf oidn-2.4.1.x86_64.linux.tar.gz
env OIDN_DIR=/tmp/oidn-2.4.1.x86_64.linux

RUN echo 'user ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} user && \
    useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash user

USER user

# Run 
WORKDIR /workspace
