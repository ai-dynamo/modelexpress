# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO: Update this to a TRT-LLM release image containing the complete upstream
# ModelExpress PR chain before treating this as the released integration path.
ARG TRTLLM_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22
FROM ${TRTLLM_IMAGE}

# Use the NIXL Python binding shipped with TensorRT-LLM so its Python and native
# integrations share one NIXL core.
ENV PYTHONPATH=/opt/nvidia/nvda_nixl/lib/python3/dist-packages
RUN printf '%s\n' \
      /opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu \
      /opt/nvidia/nvda_nixl/lib/aarch64-linux-gnu \
      /opt/nvidia/nvda_nixl/lib64 \
      > /etc/ld.so.conf.d/nixl.conf && \
    ldconfig

# TensorRT-LLM owns the dependency stack. Install only the ModelExpress package
# from this repository without resolving or replacing engine dependencies.
COPY modelexpress_client/python /opt/modelexpress/client
WORKDIR /opt/modelexpress/client
RUN python3 -m pip uninstall -y modelexpress 2>/dev/null || true
RUN python3 -m pip install --no-cache-dir --no-deps .
# --no-deps above excludes the [metrics] extra, so prometheus-client is
# installed explicitly; the collector otherwise catches the ImportError and
# disables itself. It has no dependencies of its own, so --no-deps keeps the
# TensorRT-LLM stack untouched.
RUN python3 -m pip install --no-cache-dir --no-deps "prometheus-client>=0.20.0"

WORKDIR /workspace
