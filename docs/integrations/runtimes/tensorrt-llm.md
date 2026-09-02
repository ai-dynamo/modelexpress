<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TensorRT-LLM

TensorRT-LLM integrates through its native `checkpoint_format="MX"` interface. The checked-in beta example covers `LlamaForCausalLM`, TP=4, a ModelExpress server, NIXL P2P, and native checkpoint fallback.

```yaml
checkpoint_format: MX
mx_config:
  server_url: modelexpress-server:8001
```

Start with the [example README](../../../examples/p2p_transfer_k8s/client/trtllm/README.md), [Dockerfile](../../../examples/p2p_transfer_k8s/client/trtllm/Dockerfile), and [Kubernetes manifest](../../../examples/p2p_transfer_k8s/client/trtllm/trtllm-single-node-p2p.yaml). The example currently pins `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22`; keep the TensorRT-LLM image, ModelExpress client, NIXL libraries, and native MX loader in one tested compatibility set.

The current adapter exposes P2P and the native loader. Server cache, InstantTensor, ModelStreamer, and GDS are not eligible because the adapter does not implement their hooks.

Wait for the first worker to become ready before scaling. Target logs should include `Eligible loaders: ['rdma', 'default']`, `Trying strategy: rdma`, and `RDMA transfer complete`. If only `default` is eligible, check the server address, source identity, NIXL availability, and RDMA resources.

TensorRT-LLM is not currently part of the active ModelExpress GPU CI matrix. Validate the upstream MX integration and target GPU/fabric combination before production use.
