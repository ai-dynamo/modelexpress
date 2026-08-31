<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM

ModelExpress integrates with vLLM through the `modelexpress` load format. The loader can select P2P transfer, server cache, InstantTensor, ModelStreamer, GDS, or vLLM's native loader according to the fixed ModelExpress strategy chain.

## Choose a vLLM path

| Goal | Starting point |
|---|---|
| Scale replicas with P2P | [`vllm-single-node.yaml`](../../../examples/p2p_transfer_k8s/client/vllm/vllm-single-node.yaml) |
| Use P2P across multiple nodes | [`vllm-multi-node.yaml`](../../../examples/p2p_transfer_k8s/client/vllm/vllm-multi-node.yaml) |
| Stream from S3, Azure, or local storage | [`model_streamer_k8s/client/vllm`](../../../examples/model_streamer_k8s/client/vllm/README.md) |
| Run through Dynamo | [Dynamo integration](../orchestrators/dynamo.md) |

## Minimal command

For vLLM `0.23.0` and newer, the load format is recognized natively after the ModelExpress Python client is installed in the image:

```bash
export MX_SERVER_ADDRESS=modelexpress-server:8001
vllm serve my-org/my-model --load-format modelexpress
```

The `mx` load format is retained as a backward-compatible alias. For older vLLM images, set `VLLM_PLUGINS=modelexpress` and use the plugin registration supplied by the client image. The CI worker image currently starts from `vllm/vllm-openai:v0.17.1`; the public example image starts from `vllm/vllm-openai:v0.23.0`. See [Compatibility](../../COMPATIBILITY.md) before selecting an image.

## Image and deployment

Build from the repository root with [`examples/p2p_transfer_k8s/client/vllm/Dockerfile`](../../../examples/p2p_transfer_k8s/client/vllm/Dockerfile) for the current example image. The Dockerfile layers the Python client onto the runtime image; it should not replace the runtime's CUDA, Torch, or NIXL stack unless the integration explicitly requires it.

For P2P, set `MX_SERVER_ADDRESS`, provide a central metadata backend or use the `k8s-service` topology, and give the first replica a bootstrap source. For direct ModelStreamer, set `MX_MODEL_URI`; the server and P2P resources are optional.

## Verification

Look for `Eligible loaders`, `Trying strategy`, and a path-specific completion message in the worker logs. The active CI coverage checks vLLM P2P, TP, DP, EP, artifact transfer, rolling update, stale metadata, fleet scale, Dynamo aggregated/disaggregated serving, direct S3 streaming, and S3 fallback.
