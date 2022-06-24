# Updated Dockerfile for our Gradient base image
# Changes from Tom G's by Nick prefixed by "NMB"
#
# Last updated: Jun 24th 2022

# ==================================================================
# module list
# ------------------------------------------------------------------
# python         3.9    (apt) # NMB: Want Python 3.9.13
## python        3.6    (apt)
# jupyter        latest (pip)
# pytorch        latest (pip)
# tensorflow     latest (pip)
# jupyterlab     latest (pip)
## keras         latest (pip) # NMB: TensorFlow now has Keras
## opencv        4.5.1  (git) # NMB: Remove unless needed
## caffe         latest (git) # NMB: Same
# ==================================================================

# NMB: Update from CUDA 11.0 to 11.3 and Ubuntu from 18.04 to 20.04
# NMB: Based on the container from CW's candidate blank image at 
# https://github.com/Paperspace/notebook-runtimes/blob/main/nb-python/Dockerfile
# NMB: Change PIP_INSTALL from python to python3 (python command not found)

#FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
ENV LANG C.UTF-8
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python3 -m pip --no-cache-dir --default-timeout=1000 install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \

    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \

    apt-get update && \

# ==================================================================
# tools
# ------------------------------------------------------------------

    # NMB: Include new terminal commands:
    # aws csvkit emacs joe jq man nano ping sudo
    # NMB: Allow manpages to work was requested by fast.ai

    # NMB: Get the manpages back 1 of 2:
    # Comment out dpkg exclusion for manpages
    # Based on https://github.com/blitterated/docker_dev_env/wiki/Setup-man-pages-in-a-minimized-Ubuntu-container
    # Gets manpages without running "unminimize", which adds other
    # libraries too - the regular Ubuntu instead of the minimal one
    
    sed -e '\|/usr/share/man|s|^#*|#|g' -i /etc/dpkg/dpkg.cfg.d/excludes && \

    # APT installs

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        rsync \
        git \
        vim \
        libssl-dev \
        curl \
        openssh-client \
        unzip \
        unrar \
        zip \
        awscli \
        csvkit \
        emacs \
        joe \
        jq \
        dialog \
        man-db \
        manpages \
        manpages-dev \
        manpages-posix \
        manpages-posix-dev \
        nano \
        iputils-ping \
        sudo \
        && \

    # NMB: Get the manpages back 2 of 2:
    # Remove dpkg-divert entries
    
    rm -f /usr/bin/man && \
    dpkg-divert --quiet --remove --rename /usr/bin/man && \
    rm -f /usr/share/man/man1/sh.1.gz && \
    dpkg-divert --quiet --remove --rename /usr/share/man/man1/sh.1.gz && \

    # cmake

    $GIT_CLONE https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
    make -j"$(nproc)" install && \

# ==================================================================
# python
# ------------------------------------------------------------------

    # NMB: Want Python 3.9.13 so use apt method to get 3.9, which gets 3.9.13
    # NMB: Add ipywidgets and gradient following CW's candidate blank
    # image at https://github.com/Paperspace/notebook-runtimes/blob/main/nb-python/Dockerfile
    # NMB: Remove gradient-sdk from pip installs and just use gradient

    # APT default Python
    
    #DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
    #    python3 \
    #    python3-pip \
    #    && \

    # APT Python 3.9
    
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.9 \
        python3.9-dev \
        python3-distutils-extra \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.9 ~/get-pip.py && \
    ln -s /usr/bin/python3.9 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.9 /usr/local/bin/python && \

    # NMB: Add symlink so users don't hit python command not found
    # Ubuntu 20.04 doesn't have Python 2.x, aka. "python"
    # Cf. https://packages.debian.org/unstable/python-is-python3
    
    #ln -s /usr/bin/python3 /usr/bin/python && \

    # Other Python pip installs

    $PIP_INSTALL \
        setuptools \
        && \

    $PIP_INSTALL \
        numpy \
        scipy \
        pandas \
        cloudpickle \
        scikit-image>=0.14.2 \
        scikit-learn \
        matplotlib \
        ipython \
        ipykernel \
        ipywidgets \
        gradient \
        Cython \
        tqdm \
        && \

# ==================================================================
# boost
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        libboost-all-dev \
        && \

# ==================================================================
# node.js
# ------------------------------------------------------------------

    curl -sL https://deb.nodesource.com/setup_16.x | bash && \

    $APT_INSTALL \
        nodejs \
	&& \
	
# ==================================================================
# jupyter
# ------------------------------------------------------------------

    $PIP_INSTALL \
        jupyter \
        jupyter_contrib_nbextensions \
        && \

    jupyter contrib nbextension install --sys-prefix && \

# ==================================================================
# pytorch
# ------------------------------------------------------------------

# NMB: Based on selector in https://pytorch.org/get-started/locally
# using Stable (1.11.0), Linux, Pip, Python, CUDA 11.3
# The NumPy and pip prerequisites are met above

    $PIP_INSTALL \
        torch \
        torchvision \
        torchaudio \
        --extra-index-url https://download.pytorch.org/whl/cu113 \
        && \

#    $PIP_INSTALL \
#        future \
#        numpy \
#        protobuf \
#        enum34 \
#        pyyaml \
#        typing \
#        && \
#    $PIP_INSTALL \
#        --pre torch torchvision -f \
#        https://download.pytorch.org/whl/nightly/cu110/torch_nightly.html \
#        && \

# ==================================================================
# tensorflow
# ------------------------------------------------------------------

# NMB: Based on https://www.tensorflow.org/install and 
# https://www.tensorflow.org/install/pip, so is now not -gpu

    $PIP_INSTALL \
        tensorflow \
        && \

#    $PIP_INSTALL \
#        tensorflow-gpu \
#        && \

# ==================================================================
# jupyterlab
# ------------------------------------------------------------------

    $PIP_INSTALL \
        jupyterlab \
	jupyterlab-git \
        && \

# ==================================================================
# keras
# ------------------------------------------------------------------

# NMB: TensorFlow now comes with Keras so don't need separate install

#    $PIP_INSTALL \
#        h5py \
#        keras \
#        && \

# ==================================================================
# opencv
# ------------------------------------------------------------------

# NMB: Remove to save image size unless users need it on base

#    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
#        libatlas-base-dev \
#        libgflags-dev \
#        libgoogle-glog-dev \
#        libhdf5-serial-dev \
#        libleveldb-dev \
#        liblmdb-dev \
#        libprotobuf-dev \
#        libsnappy-dev \
#        protobuf-compiler \
#        && \

#    $GIT_CLONE --branch 4.5.1 https://github.com/opencv/opencv ~/opencv && \
#    mkdir -p ~/opencv/build && cd ~/opencv/build && \
#    cmake -D CMAKE_BUILD_TYPE=RELEASE \
#          -D CMAKE_INSTALL_PREFIX=/usr/local \
#          -D WITH_IPP=OFF \
#          -D WITH_CUDA=OFF \
#          -D WITH_OPENCL=OFF \
#          -D BUILD_TESTS=OFF \
#          -D BUILD_PERF_TESTS=OFF \
#          -D BUILD_DOCS=OFF \
#          -D BUILD_EXAMPLES=OFF \
#          .. && \
#    make -j"$(nproc)" install && \
#    ln -s /usr/local/include/opencv4/opencv2 /usr/local/include/opencv2 && \

# ==================================================================
# caffe
# ------------------------------------------------------------------

# NMB: Remove to save image size unless users need it on base

#    apt-get update && \
#    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
#        caffe-cuda \
#        && \

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

EXPOSE 8888 6006
