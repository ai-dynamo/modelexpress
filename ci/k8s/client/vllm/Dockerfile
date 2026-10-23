# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM vllm/vllm-openai:v0.19.1

# Copy and install ModelExpress Python client
# Proto stubs (p2p_pb2.py, p2p_pb2_grpc.py) are checked-in source artifacts;
# regenerate them with grpcio-tools when p2p.proto changes.
COPY modelexpress_client/python /opt/modelexpress/client
WORKDIR /opt/modelexpress/client
RUN pip uninstall -y modelexpress 2>/dev/null || true && \
    rm -rf "$(python3 -c 'import site; print(site.getsitepackages()[0])')"/modelexpress* && \
    pip install . transformers==5.5.1

RUN python3 -c "from transformers import AutoConfig; assert AutoConfig.for_model('glm4_moe_lite').model_type == 'glm4_moe_lite'; from modelexpress.engines.vllm.loader import MxModelLoader"
RUN python3 -c "import torch; from modelexpress._nixl import load_nixl_api; api = load_nixl_api(); assert api is not None; assert api.__name__.split('.')[0] in ('nixl', 'nixl_cu' + torch.version.cuda.split('.')[0]); print('NIXL backend:', api.__name__, 'Torch CUDA:', torch.version.cuda)"

COPY ci/k8s/client/vllm/weight_diagnostics.py /opt/mx-ci-weight-diagnostics.py
RUN python3 /opt/mx-ci-weight-diagnostics.py

# Back to default workdir
WORKDIR /workspace
