# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM vllm/vllm-openai:v0.17.1

# Copy and install ModelExpress Python client
# Proto stubs (p2p_pb2.py, p2p_pb2_grpc.py) are checked-in source artifacts;
# regenerate them with grpcio-tools when p2p.proto changes.
COPY modelexpress_client/python /opt/modelexpress/client
WORKDIR /opt/modelexpress/client
RUN pip uninstall -y modelexpress 2>/dev/null || true && \
    rm -rf "$(python3 -c 'import site; print(site.getsitepackages()[0])')"/modelexpress* && \
    pip install .

# Back to default workdir
WORKDIR /workspace
