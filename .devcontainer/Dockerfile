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
    lsb-release \
    openssh-client \
    pkg-config \
    postgresql-client \
    procps \
    protobuf-compiler \
    redis-tools \
    ripgrep \
    sudo \
    tree \
    unzip \
    wget \
    # Generate locale
    && locale-gen en_US.UTF-8 \
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Set locale
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=1234
ARG USER_GID=$USER_UID

RUN groupadd --gid "$USER_GID" "$USERNAME" \
    && useradd --uid "$USER_UID" --gid "$USER_GID" -m "$USERNAME" \
    && echo "$USERNAME" ALL=\(root\) NOPASSWD:ALL > "/etc/sudoers.d/$USERNAME" \
    && chmod 0440 "/etc/sudoers.d/$USERNAME"

USER vscode
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y


# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog
