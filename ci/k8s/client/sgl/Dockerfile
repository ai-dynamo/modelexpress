# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# AMD64 SGLang image for CI P2P tests.
# Layers ModelExpress client on the upstream lmsysorg/sglang image.
#
# Build (from repo root):
#   docker buildx build --platform linux/amd64 \
#       -f ci/k8s/client/sgl/Dockerfile \
#       -t <IMAGE> --push --load .
#
# TODO: Switch to an SGLang release image once the ModelExpress loader PR is released.
ARG SGLANG_IMAGE=lmsysorg/sglang@sha256:6553f8288e848fd833a4d473eca7aa2c8f141f98c52f63a5f4684631de50f9f2
FROM ${SGLANG_IMAGE}

# Install ModelExpress Python client.
# Proto stubs (p2p_pb2.py, p2p_pb2_grpc.py) are checked-in source artifacts;
# regenerate with grpcio-tools when p2p.proto changes.
#
# Pip's default upgrade strategy (only-if-needed) keeps the base image's
# pre-installed nixl untouched as long as the constraint in modelexpress's
# pyproject.toml (`nixl[cu12]; sys_platform == 'linux'`) is unbounded.
# If lmsysorg/sglang ships a nixl that's incompatible with this image's UCX,
# see docs/NIXL_UCX_COMPATIBILITY.md for the resolution pattern.
COPY modelexpress_client/python /opt/modelexpress/client
WORKDIR /opt/modelexpress/client
RUN pip uninstall -y modelexpress 2>/dev/null || true && \
    rm -rf "$(python3 -c 'import site; print(site.getsitepackages()[0])')"/modelexpress* && \
    pip install .

# Verify the client and the SGLang loader hook imported successfully.
RUN python3 -c "from modelexpress.client import MxClient; print('MxClient OK')" && \
    python3 -c "from modelexpress.engines.sglang import MxModelLoader; print('SGLang MxModelLoader OK')"

WORKDIR /workspace
