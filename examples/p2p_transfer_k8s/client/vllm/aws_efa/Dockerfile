# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Dynamo vLLM runtime with AWS EFA libfabric NIXL backend + ModelExpress client.
#
# The base image must include Dynamo, vLLM 0.23.0, the AWS EFA installer,
# libfabric provider, and aws-ofi-nccl. We layer the ModelExpress Python client
# on top for P2P RDMA weight transfer via NIXL over EFA.
#
# At runtime, set MX_NIXL_BACKEND=LIBFABRIC to select the libfabric backend
# (instead of the default UCX, which can silently fall back to TCP on EFA).
# See vllm-aws-efa.yaml for the required env vars.
ARG DYNAMO_VLLM_EFA_RUNTIME_IMAGE
FROM ${DYNAMO_VLLM_EFA_RUNTIME_IMAGE}

RUN python3 -c \
    'import vllm; assert vllm.__version__.split("+", 1)[0] == "0.23.0", vllm.__version__'

# Copy and install ModelExpress Python client
COPY --chown=dynamo:dynamo modelexpress_client/python /opt/modelexpress/client
WORKDIR /opt/modelexpress/client
RUN uv pip install .

# Back to default workdir
WORKDIR /workspace
