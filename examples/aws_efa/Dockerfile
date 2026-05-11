# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Dynamo vLLM runtime with AWS EFA libfabric NIXL backend + ModelExpress client.
#
# The -efa-amd64 base image variant is published by the dynamo team with the
# AWS EFA installer, libfabric provider, and aws-ofi-nccl already in place.
# We layer the ModelExpress Python client on top for P2P RDMA weight transfer
# via NIXL over EFA.
#
# At runtime, set MX_NIXL_BACKEND=LIBFABRIC to select the libfabric backend
# (instead of the default UCX, which can silently fall back to TCP on EFA).
# See deploy/source.yaml and deploy/target.yaml for the required env vars.
FROM nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0-efa-amd64

# Copy and install ModelExpress Python client
COPY --chown=dynamo:dynamo modelexpress_client/python /opt/modelexpress/client
WORKDIR /opt/modelexpress/client
RUN uv pip install .

# Back to default workdir
WORKDIR /workspace
