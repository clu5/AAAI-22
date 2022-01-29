FROM pytorch/pytorch:1.8.0-cuda10.1-cudnn7-runtime
LABEL maintainer="CLU5"
ARG SSH_KEY

# switch default shell execution from `sh` to `bash`
SHELL ["/bin/bash", "-c"]

RUN apt update --fix-missing && apt install -y \
    bzip2 \
    ca-certificates \
    curl\
    gcc \
    git \
    grep \
    htop \
    less \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    man \
    openssl \
    shellcheck \
    tar \
    tmux \
    unzip \
    vim \
    wget && \
    apt clean

# Minio S3 client
RUN wget --quiet https://dl.min.io/client/mc/release/linux-amd64/mc -O /bin/mc && chmod +x /bin/mc

# install python packages
COPY requirements.txt /home/requirements.txt
RUN python -m pip install --upgrade --no-cache-dir pip && \
    python -m pip install --no-cache-dir -r /home/requirements.txt


# vim things
RUN mkdir -p ~/.vim/undodir && \
    mkdir -p ~/.vim/pack/plug/start && cd ~/.vim/pack/plug/start && \
    git clone https://github.com/morhetz/gruvbox.git && \
    git clone https://github.com/scrooloose/nerdtree.git && \
    git clone https://github.com/nvie/vim-flake8.git
