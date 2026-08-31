<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM

vLLM uses the `modelexpress` load format. The loader can use P2P, server cache, InstantTensor, ModelStreamer, GDS, or vLLM's native loader according to the fixed strategy chain.

For vLLM `0.23.0` and newer, install the ModelExpress Python client and run:

```bash
export MX_SERVER_ADDRESS=modelexpress-server:8001
vllm serve my-org/my-model --load-format modelexpress
```

`mx` remains a backward-compatible alias. Older vLLM images use the ModelExpress plugin and `VLLM_PLUGINS=modelexpress`.

| Goal | Starting point |
|---|---|
| Single-node P2P | [`vllm-single-node.yaml`](../../../examples/p2p_transfer_k8s/client/vllm/vllm-single-node.yaml) |
| Multi-node P2P | [`vllm-multi-node.yaml`](../../../examples/p2p_transfer_k8s/client/vllm/vllm-multi-node.yaml) |
| S3, Azure, or local ModelStreamer | [vLLM storage examples](../../../examples/model_streamer_k8s/client/vllm/README.md) |
| Dynamo | [Dynamo integration](../orchestrators/dynamo.md) |

Build the example image from [`examples/p2p_transfer_k8s/client/vllm/Dockerfile`](../../../examples/p2p_transfer_k8s/client/vllm/Dockerfile), which currently starts from `vllm/vllm-openai:v0.23.0`. For P2P, configure source discovery and a bootstrap source. For direct storage loading, set `MX_MODEL_URI`; the server and RDMA resources are optional.

Verify the selected path through `Eligible loaders`, `Trying strategy`, and the path-specific completion message in worker logs.
