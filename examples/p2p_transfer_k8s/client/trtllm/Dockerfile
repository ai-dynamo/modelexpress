# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG TRTLLM_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc11@sha256:d91c80ba8baf763782b1078267ed6b1e06363bebff4961094bf6e5679d371d04
FROM ${TRTLLM_IMAGE}

USER root

ENV NIXL_PREFIX=/opt/nvidia/nvda_nixl \
    NIXL_LIB_DIR=/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu \
    NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu/plugins
ENV LD_LIBRARY_PATH=${NIXL_LIB_DIR}:${NIXL_PLUGIN_DIR}:/usr/local/tensorrt/targets/x86_64-linux-gnu/lib:${LD_LIBRARY_PATH}

COPY examples/p2p_transfer_k8s/client/trtllm/install_modelexpress_client.sh /tmp/trtllm_modelexpress/install_modelexpress_client.sh
COPY examples/p2p_transfer_k8s/client/trtllm/fix_nixl_runpath.py /tmp/trtllm_modelexpress/fix_nixl_runpath.py

COPY modelexpress_client/python /tmp/modelexpress_client
COPY modelexpress_common/proto /tmp/proto

RUN bash /tmp/trtllm_modelexpress/install_modelexpress_client.sh && \
    python3 /tmp/trtllm_modelexpress/fix_nixl_runpath.py

RUN python3 -c "import modelexpress; print('ModelExpress OK')" && \
    test -f "$NIXL_PLUGIN_DIR/libplugin_UCX.so" && \
    echo "TRT-LLM ModelExpress image OK" && \
    rm -rf /tmp/trtllm_modelexpress /tmp/modelexpress_client /tmp/proto
