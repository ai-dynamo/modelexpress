# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.5.1

# Preserve the runtime's SGLang, CUDA, Torch, NIXL, and protobuf versions.
USER root
COPY --chown=dynamo:dynamo modelexpress_client/python /opt/modelexpress/client
WORKDIR /opt/modelexpress/client
RUN python3 -m pip install --break-system-packages --no-cache-dir --no-deps --force-reinstall .
RUN python3 -c "from modelexpress.engines.sglang import MxModelLoader; from dynamo.sglang.args import parse_args"
COPY ci/k8s/client/launch_dynamo.py /opt/mx-ci/launch_dynamo.py

USER dynamo
WORKDIR /workspace
