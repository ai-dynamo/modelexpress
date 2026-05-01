# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl jq \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir aiperf==0.7.0 tiktoken

ENV COLUMNS=200 \
    HF_HOME=/model-cache \
    PYTHONUNBUFFERED=1

WORKDIR /workspace
