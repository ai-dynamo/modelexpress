# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Dynamo vLLM runtime with ModelExpress P2P client.
#
# Base image includes dynamo runtime + vLLM. We layer the ModelExpress
# Python client on top for P2P RDMA weight transfer via NIXL.
#
# Alternatively, build from the dynamo repo with MX integrated:
#   cd path/to/dynamo
#   python container/render.py --framework vllm --target runtime \
#       --platform amd64 --cuda-version 12.9 --output-short-filename
#   docker build --platform linux/amd64 \
#       -f container/rendered.Dockerfile \
#       --build-arg ENABLE_MODELEXPRESS_P2P=true \
#       --build-arg MODELEXPRESS_REF=main \
#       -t <your-tag> .
FROM nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1

# Copy and install ModelExpress Python client
# Proto stubs (p2p_pb2.py, p2p_pb2_grpc.py) are checked-in source artifacts;
# regenerate them with grpcio-tools when p2p.proto changes.
COPY --chown=dynamo:dynamo modelexpress_client/python /opt/modelexpress/client
WORKDIR /opt/modelexpress/client
RUN uv pip install .

# Back to default workdir
WORKDIR /workspace
