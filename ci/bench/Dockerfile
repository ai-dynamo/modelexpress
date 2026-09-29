# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
ARG RUNTIME_BASE
FROM ${RUNTIME_BASE}
COPY modelexpress_client/python /opt/modelexpress/client
RUN python3 -m pip install --no-cache-dir /opt/modelexpress/client && \
    python3 -c 'import boto3, numpy, requests, safetensors, torch, vllm, modelexpress_rl'
