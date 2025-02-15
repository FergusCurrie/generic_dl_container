# Use an official NVIDIA CUDA image as the base
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

ENV PROJECT_NAME=generic_dl_container
ENV WORKDIR=/workspaces/${PROJECT_NAME}
WORKDIR ${WORKDIR}


# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     git \
#     python3.10 \
#     python3.10-pip \
#     python3.10-dev \
#     python3.10-venv \
#     curl

# # Install Poetry
# RUN curl -sSL https://install.python-poetry.org | python3 -
# RUN pip install poetry


# Install Python 3.10
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    software-properties-common \
    curl \
    gnupg \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.10 python3.10-venv python3.10-dev \
    && apt-get install -y python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Verify installation
RUN python3 --version && pip3 --versio

# install poetry 
RUN pip install poetry

# copy poetry settings in
COPY poetry.lock pyproject.toml ${WORKDIR}

# install dependencies
RUN poetry install
