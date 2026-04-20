# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM vllm/vllm-openai:v0.17.1

# Install mx_peer_discovery (Python half) first so modelexpress can import it
COPY mx_peer_discovery_py /opt/mx_peer_discovery_py
RUN cd /opt/mx_peer_discovery_py && pip install --no-cache-dir .

# Install modelexpress client
COPY modelexpress_client/python /opt/modelexpress/client
RUN cd /opt/modelexpress/client && \
    pip uninstall -y modelexpress 2>/dev/null || true && \
    rm -rf "$(python3 -c 'import site; print(site.getsitepackages()[0])')"/modelexpress* && \
    pip install --no-cache-dir .

# Test script
COPY tests/peer_direct_k8s/peer_test.py /test/peer_test.py

WORKDIR /test
ENTRYPOINT []
CMD ["python3", "/test/peer_test.py"]
