# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# This Dockerfile can be used instead of the image property in devcontainer.json
# for more advanced customization

FROM ubuntu:24.04

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Configure apt and install packages
RUN apt-get update \
    && apt-get -y install \
    apt-utils \
    bash-completion \
    bat \
    build-essential \
    ca-certificates \
    curl \
    dialog \
    eza \
    fd-find \
    git \
    htop \
    httpie \
    iproute2 \
    jq \
    less \
    libprotobuf-dev \
    libssl-dev \
    locales \
    pipx \
    python3 \
    lsb-release \
    net-tools \
    openssh-client \
    pkg-config \
    postgresql-client \
    procps \
    protobuf-compiler \
    redis-tools \
    ripgrep \
    sudo \
    tofrodos \
    tree \
    unzip \
    wget \
    # Generate locale
    && locale-gen en_US.UTF-8 \
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

RUN echo 'ubuntu:ubuntu' | chpasswd
RUN echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu

# Set locale
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

USER ubuntu

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

# Install pre-commit via pipx
RUN pipx install pre-commit
ENV PATH="/home/ubuntu/.local/bin:${PATH}"

# Make git happy with codespaces mounted repos
RUN git config --global --add safe.directory /workspaces/ModelExpress

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog
