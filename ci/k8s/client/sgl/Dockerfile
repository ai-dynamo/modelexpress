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
# lmsysorg/sglang:v0.5.13.post1 includes the upstream ModelExpress delegation
# hook from sgl-project/sglang#24723.
ARG SGLANG_IMAGE=lmsysorg/sglang:v0.5.13.post1
FROM ${SGLANG_IMAGE}

# Install ModelExpress Python client from this repository.
# Proto stubs (p2p_pb2.py, p2p_pb2_grpc.py) are checked-in source artifacts;
# regenerate with grpcio-tools when p2p.proto changes.
#
# The SGLang base image owns the CUDA/NIXL/Torch/protobuf stack. Install the MX
# package without dependency resolution so pip does not downgrade or replace
# engine-provided runtime packages.
COPY modelexpress_client/python /opt/modelexpress/client
WORKDIR /opt/modelexpress/client
RUN pip uninstall -y modelexpress 2>/dev/null || true && \
    rm -rf "$(python3 -c 'import site; print(site.getsitepackages()[0])')"/modelexpress* && \
    pip install --no-cache-dir --no-deps .

# Verify the client and the SGLang loader hook imported successfully.
RUN python3 -c "from modelexpress.client import MxClient; print('MxClient OK')" && \
    python3 -c "from modelexpress.engines.sglang import MxModelLoader; print('SGLang MxModelLoader OK')" && \
    python3 -c "import runai_model_streamer; print('ModelStreamer OK')"

WORKDIR /workspace
