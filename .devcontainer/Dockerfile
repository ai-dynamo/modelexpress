# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# This Dockerfile can be used instead of the image property in devcontainer.json
# for more advanced customization

FROM ubuntu:24.04

ARG ARCH=amd64
ARG ARCH_ALT=x86_64

# NIXL configuration
ARG NIXL_UCX_REF=v1.19.0
ARG NIXL_REF=0.6.0

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Configure apt and install packages
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    apt-utils \
    bash-completion \
    bat \
    build-essential \
    ca-certificates \
    curl \
    dialog \
    eza \
    fd-find \
    htop \
    httpie \
    iproute2 \
    jq \
    less \
    libprotobuf-dev \
    libssl-dev \
    locales \
    lsb-release \
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
    # NIXL build dependencies
    autoconf \
    automake \
    cmake \
    git \
    git-lfs \
    libtool \
    meson \
    net-tools \
    ninja-build \
    pybind11-dev \
    # Generate locale
    && locale-gen en_US.UTF-8 \
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# These headers are missing with the hpcx installer, required
# by UCX to build and use RDMA devices. Reinstall to make sure to recreate
# symlink .so to .so.1 in case some packages are already found.
RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get -y install --reinstall --no-install-recommends \
    libibverbs-dev \
    rdma-core \
    ibverbs-utils \
    libibumad-dev \
    libnuma-dev \
    librdmacm-dev \
    ibverbs-providers \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

##################################
########## UCX Build #############
##################################

# TODO: Add CUDA and GDRCopy support using the following flags:
#   --with-cuda=/usr/local/cuda
#   --with-gdrcopy=/usr/local
#
# Build and install UCX
RUN rm -rf /opt/hpcx/ucx && \
    rm -rf /usr/local/ucx && \
    echo "Building UCX with reference $NIXL_UCX_REF" && \
    cd /usr/local/src && \
    git clone https://github.com/openucx/ucx.git && \
    cd ucx && git checkout $NIXL_UCX_REF && \
    CC=gcc && \
    CXX=g++ && \
    export CC=${CC} && \
    export CXX=${CXX} && \
    ./autogen.sh && \
    ./configure \
    --prefix=/usr/local/ucx \
    --enable-shared \
    --disable-static \
    --disable-doxygen-doc \
    --enable-optimizations \
    --enable-cma \
    --enable-devel-headers \
    --with-verbs \
    --with-efa \
    --with-dm \
    --enable-mt && \
    make -j$(nproc) && \
    make -j$(nproc) install-strip && \
    echo "/usr/local/ucx/lib" > /etc/ld.so.conf.d/ucx.conf && \
    echo "/usr/local/ucx/lib/ucx" >> /etc/ld.so.conf.d/ucx.conf && \
    ldconfig && \
    cd /usr/local/src && \
    rm -rf ucx

# UCX environment variables
ENV CPATH=/usr/include \
    PATH=/usr/bin:/usr/local/ucx/bin:$PATH \
    PKG_CONFIG_PATH=/usr/lib/pkgconfig

##################################
########## NIXL Setup ############
##################################

# NIXL environment setup
ENV NIXL_SRC_DIR=/opt/nixl \
    NIXL_PREFIX=/opt/nvidia/nvda_nixl \
    NIXL_LIB_DIR=/opt/nvidia/nvda_nixl/lib/${ARCH_ALT}-linux-gnu \
    NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/lib/${ARCH_ALT}-linux-gnu/plugins

# Build and install NIXL
RUN git clone --depth 1 --branch ${NIXL_REF} "https://github.com/ai-dynamo/nixl.git" ${NIXL_SRC_DIR} && \
    cd ${NIXL_SRC_DIR} && \
    if [ "$ARCH" = "arm64" ]; then \
    nixl_build_args="-Ddisable_gds_backend=true"; \
    else \
    nixl_build_args=""; \
    fi && \
    meson setup build/ --buildtype=release --prefix=$NIXL_PREFIX $nixl_build_args && \
    ninja -C build/ -j$(nproc) && ninja -C build/ install && \
    echo "$NIXL_LIB_DIR" > /etc/ld.so.conf.d/nixl.conf && \
    echo "$NIXL_PLUGIN_DIR" >> /etc/ld.so.conf.d/nixl.conf && \
    ldconfig


# Set locale
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog
