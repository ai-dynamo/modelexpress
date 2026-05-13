#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd /tmp/modelexpress_client

pip install --no-cache-dir \
    "grpcio==1.66.2" \
    "grpcio-tools==1.66.2" \
    "protobuf>=5.27.0,<6.0.0"

python3 -m grpc_tools.protoc \
    -I/tmp/proto \
    --python_out=modelexpress \
    --grpc_python_out=modelexpress \
    /tmp/proto/p2p.proto
sed -i 's/^import p2p_pb2/from . import p2p_pb2/' modelexpress/p2p_pb2_grpc.py

pip install --no-cache-dir .

# TRT-LLM 1.3.x release images are CUDA 13, so use the matching NIXL CUDA
# plugin instead of the package default pulled in by the generic dependency.
pip uninstall -y nixl nixl-cu12 nixl-cu13
pip install --no-cache-dir --no-deps --force-reinstall \
    "nixl==0.10.1" \
    "nixl-cu13==0.10.1"
