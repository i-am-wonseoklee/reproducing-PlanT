FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

COPY pyproject.toml /tmp/pyproject.toml
RUN pip install --no-cache-dir /tmp

WORKDIR /workspace
