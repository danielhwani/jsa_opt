FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN sed -i \
    -e 's|http://archive.ubuntu.com/ubuntu|https://ftp.kaist.ac.kr/ubuntu|g' \
    -e 's|http://security.ubuntu.com/ubuntu|https://ftp.kaist.ac.kr/ubuntu|g' \
    /etc/apt/sources.list

# Install dependencies
RUN apt-get update -o Acquire::Retries=5 && apt-get install -y --no-install-recommends \
    python3-pip \
    libgl1-mesa-glx \
    libgtk2.0-dev \
    libopenexr-dev && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install PyEXR==0.3.9 opencv-python 
RUN pip3 install parmap
RUN pip3 install OpenEXR
RUN pip3 install timm==0.4.12
RUN pip3 install einops

RUN apt-get update && apt-get install -y \ 
build-essential \
git \
cmake 
RUN echo 'user ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} user && \
    useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash user

USER user

# Run 
WORKDIR /workspace
