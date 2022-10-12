#!/bin/bash

# Paperspace Dockerfile for Gradient base image
# Paperspace image is located in Dockerhub registry: paperspace/gradient_base

# ==================================================================
# Module list
# ------------------------------------------------------------------
# python                3.9.12           (conda)
# torch                 1.12.1           (pip)
# torchvision           0.13.1           (pip)
# torchaudio            0.12.1           (pip)
# tensorflow            2.10.0           (pip)
# jax                   0.3.17           (pip)
# transformers          4.21.3           (pip)
# datasets              2.4.0            (pip)
# jupyterlab            3.4.6            (pip)
# numpy                 1.23.2           (pip)
# scipy                 1.9.1            (pip)
# pandas                1.4.4            (pip)
# cloudpickle           2.1.0            (pip)
# scikit-image          0.19.3           (pip)
# scikit-learn          1.1.2            (pip)
# matplotlib            3.5.3            (pip)
# ipython               8.5.0            (pip)
# ipykernel             6.15.2           (pip)
# ipywidgets            8.0.2            (pip)
# cython                0.29.32          (pip)
# tqdm                  4.64.1           (pip)
# gdown                 4.5.1            (pip)
# xgboost               1.6.2            (pip)
# pillow                9.2.0            (pip)
# seaborn               0.12.0           (pip)
# sqlalchemy            1.4.40           (pip)
# spacy                 3.4.1            (pip)
# nltk                  3.7              (pip)
# boto3                 1.24.66          (pip)
# tabulate              0.8.10           (pip)
# future                0.18.2           (pip)
# gradient              2.0.6            (pip)
# jsonify               0.5              (pip)
# opencv-python         4.6.0.66         (pip)
# pyyaml                5.4.1            (pip)
# sentence-transformers 2.2.2            (pip)
# nodejs                16.x latest      (apt)
# default-jre           latest           (apt)
# default-jdk           latest           (apt)


# ==================================================================
# Initial setup
# ------------------------------------------------------------------

    # Ubuntu 20.04 as base image
    FROM ubuntu:20.04

    # Set ENV variables
    ENV LANG C.UTF-8
    ENV SHELL=/bin/bash
    SHELL ["/bin/bash", "-c"]

    ENV APT_INSTALL="apt-get install -y --no-install-recommends"
    ENV PIP_INSTALL="python3 -m pip --no-cache-dir install --upgrade"
    ENV CONDA_INSTALL="conda install -y"
    ENV GIT_CLONE="git clone --depth 10"

    # Update apt
    RUN apt-get update


# ==================================================================
# Tools
# ------------------------------------------------------------------

    RUN DEBIAN_FRONTEND=noninteractive \
        $APT_INSTALL \
        gcc \
        make \
        pkg-config \
        apt-transport-https \
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
        ffmpeg \
        libsm6 \
        libxext6 \
        libboost-all-dev \
        software-properties-common


# ==================================================================
# Conda
# ------------------------------------------------------------------

    #Based on https://docs.anaconda.com/anaconda/install/linux/

    # RUN sudo $APT_INSTALL libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
    # RUN sudo wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
    # RUN sudo bash ~/Anaconda3-2022.05-Linux-x86_64.sh -b -p $HOME/anaconda3
    
    # RUN sudo chown -R $USER:$USER $HOME/anaconda3
    # RUN sudo chmod -R +x $HOME/anaconda3
    
    # RUN source $HOME/anaconda3/bin/activate
    # RUN conda init bash
    # RUN conda deactivate
    
    # RUN export PATH=$HOME/anaconda3/bin:${PATH}
    
    # RUN $PIP_INSTALL pip
    
    # RUN rm -f ~/Anaconda3-2022.05-Linux-x86_64.sh


# ==================================================================
# Conda
# ------------------------------------------------------------------

    RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
    RUN rm ~/miniconda.sh
    ENV PATH="opt/conda/condabin:${PATH}"
    ENV PATH="opt/conda/bin:${PATH}"
    RUN echo $PATH
    RUN $PIP_INSTALL pip


# ==================================================================
# Installing CUDA packages (CUDA Toolkit 11.6.2 & CUDNN 8.5.0)
# ------------------------------------------------------------------

    RUN $CONDA_INSTALL -c "nvidia/label/cuda-11.6.2" cuda

    RUN $APT_INSTALL gnupg2
    RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
    RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
    RUN apt-get update
    RUN apt-get install libcudnn8=8.5.0.*-1+cuda11.7
    RUN apt-get install libcudnn8-dev=8.5.0.*-1+cuda11.7


# ==================================================================
# PyTorch
# ------------------------------------------------------------------

    # Based on https://pytorch.org/get-started/locally/

    RUN $PIP_INSTALL torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
        

# ==================================================================
# JAX
# ------------------------------------------------------------------

    # Based on https://github.com/google/jax#pip-installation-gpu-cuda

    RUN $PIP_INSTALL "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


# ==================================================================
# TensorFlow
# ------------------------------------------------------------------

    # Based on https://www.tensorflow.org/install/pip

    RUN export LD_LIBRARY_PATH=${HOME}/anaconda3/lib
    RUN $PIP_INSTALL tensorflow==2.10.0


# ==================================================================
# Hugging Face
# ------------------------------------------------------------------
    
    # Based on https://huggingface.co/docs/transformers/installation
    # Based on https://huggingface.co/docs/datasets/installation

    RUN $PIP_INSTALL transformers==4.21.3 datasets==2.4.0


# ==================================================================
# JupyterLab
# ------------------------------------------------------------------

    # Based on https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html#pip

    RUN $PIP_INSTALL jupyterlab==3.4.6


# ==================================================================
# Additional Python Packages
# ------------------------------------------------------------------

    RUN $PIP_INSTALL \
        numpy==1.23.2 \
        scipy==1.9.1 \
        pandas==1.4.4 \
        cloudpickle==2.1.0 \
        scikit-image==0.19.3 \
        scikit-learn==1.1.2 \
        matplotlib==3.5.3 \
        ipython==8.5.0 \
        ipykernel==6.15.2 \
        ipywidgets==8.0.2 \
        cython==0.29.32 \
        tqdm==4.64.1 \
        gdown==4.5.1 \
        xgboost==1.6.2 \
        pillow==9.2.0 \
        seaborn==0.12.0 \
        sqlalchemy==1.4.40 \
        spacy==3.4.1 \
        nltk==3.7 \
        boto3==1.24.66 \
        tabulate==0.8.10 \
        future==0.18.2 \
        gradient==2.0.6 \
        jsonify==0.5 \
        opencv-python==4.6.0.66 \
        pyyaml==5.4.1 \
        sentence-transformers==2.2.2


# ==================================================================
# Installing JRE and JDK
# ------------------------------------------------------------------

    RUN $APT_INSTALL default-jre
    RUN $APT_INSTALL default-jdk


# ==================================================================
# CMake
# ------------------------------------------------------------------

    RUN $GIT_CLONE https://github.com/Kitware/CMake ~/cmake
    RUN cd ~/cmake && \
        ./bootstrap && \
        make -j"$(nproc)" install


# ==================================================================
# Node.js and Jupyter Notebook Extensions
# ------------------------------------------------------------------

    RUN curl -sL https://deb.nodesource.com/setup_16.x | bash
    RUN $APT_INSTALL nodejs
    RUN $PIP_INSTALL jupyter_contrib_nbextensions jupyterlab-git && \
        DEBIAN_FRONTEND=noninteractive jupyter contrib nbextension install --sys-prefix


# ==================================================================
# Config & Cleanup
# ------------------------------------------------------------------

    RUN echo "export PATH=${PATH}" >> ~/.bashrc
    RUN echo "export LD_LIBRARY_PATH=/opt/conda/lib" >> ~/.bashrc

    RUN echo "export PATH=${PATH}" >> ~/.profile
    RUN echo "export LD_LIBRARY_PATH=/opt/conda/lib" >> ~/.profile

    RUN echo "export PATH=${PATH}" >> ~/.bash_profile
    RUN echo "export LD_LIBRARY_PATH=/opt/conda/lib" >> ~/.bash_profile

    EXPOSE 8888 6006


