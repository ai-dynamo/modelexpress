# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO: Switch to an SGLang release image once the ModelExpress loader PR is released.
ARG SGLANG_IMAGE=lmsysorg/sglang:nightly-dev-cu13-20260517-229cadec
FROM ${SGLANG_IMAGE}

# The SGLang base image owns the CUDA/NIXL/Torch stack. Install only the
# ModelExpress package from this repository so dependency resolution does not
# downgrade or replace engine-provided runtime packages.
COPY modelexpress_client/python /opt/modelexpress/client
WORKDIR /opt/modelexpress/client
RUN python3 -m pip uninstall -y modelexpress 2>/dev/null || true && \
    rm -rf "$(python3 -c 'import site; print(site.getsitepackages()[0])')"/modelexpress* && \
    python3 -m pip install --no-cache-dir --no-deps .

WORKDIR /sgl-workspace
