# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SGLang v0.5.13.post1 includes the upstream ModelExpress delegation hook from
# sgl-project/sglang#24723.
ARG SGLANG_IMAGE=lmsysorg/sglang:v0.5.13.post1
FROM ${SGLANG_IMAGE}

ARG INSTALL_MOONCAKE=false
RUN if [ "${INSTALL_MOONCAKE}" = "true" ]; then \
        python3 -m pip install --no-cache-dir mooncake-transfer-engine; \
    fi

# The SGLang base image owns the CUDA/NIXL/Torch stack. Install only the
# ModelExpress package from this repository so dependency resolution does not
# downgrade or replace engine-provided runtime packages.
COPY modelexpress_client/python /opt/modelexpress/client
WORKDIR /opt/modelexpress/client
RUN python3 -m pip uninstall -y modelexpress 2>/dev/null || true && \
    rm -rf "$(python3 -c 'import site; print(site.getsitepackages()[0])')"/modelexpress* && \
    python3 -m pip install --no-cache-dir --no-deps .

WORKDIR /sgl-workspace
