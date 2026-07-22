# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SGLang v0.5.13.post1 includes the upstream ModelExpress delegation hook and
# the native RunAI ModelStreamer loader used by the ModelExpress strategy.
ARG SGLANG_IMAGE=lmsysorg/sglang:v0.5.13.post1
FROM ${SGLANG_IMAGE}

COPY modelexpress_client/python /opt/modelexpress/client
WORKDIR /opt/modelexpress/client
RUN pip uninstall -y modelexpress 2>/dev/null || true && \
    rm -rf "$(python3 -c 'import site; print(site.getsitepackages()[0])')"/modelexpress* && \
    pip install --no-cache-dir --no-deps .

RUN python3 -c "from modelexpress.engines.sglang import MxModelLoader" && \
    python3 -c "import runai_model_streamer"

WORKDIR /workspace
