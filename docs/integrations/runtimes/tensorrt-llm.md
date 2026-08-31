<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TensorRT-LLM

ModelExpress integrates with TensorRT-LLM through TensorRT-LLM's native `checkpoint_format="MX"` interface. The current repository contains a beta/example path for the `LlamaForCausalLM` family; it is not part of the active ModelExpress GPU CI matrix.

## What the current adapter supports

The TensorRT-LLM adapter exposes P2P transfer from a compatible ready source and the native Hugging Face checkpoint loader as fallback. The general ModelExpress strategy chain also contains server cache, InstantTensor, ModelStreamer, and GDS, but those strategies are not eligible for the current TensorRT-LLM adapter because it does not implement their adapter hooks.

## Qualified example

The checked-in example uses one TensorRT-LLM worker per pod with TP=4, NIXL over InfiniBand/RoCE, and a PVC containing the bootstrap checkpoint:

- [TensorRT-LLM example README](../../../examples/p2p_transfer_k8s/client/trtllm/README.md)
- [TensorRT-LLM Dockerfile](../../../examples/p2p_transfer_k8s/client/trtllm/Dockerfile)
- [TensorRT-LLM Kubernetes manifest](../../../examples/p2p_transfer_k8s/client/trtllm/trtllm-single-node-p2p.yaml)

The example currently defaults to `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22` and marks the upstream MX integration as a prerequisite. Treat that image and the example as a qualification snapshot, not as a stable release guarantee. Keep the TensorRT-LLM image, ModelExpress client, NIXL libraries, and native MX loader from one tested compatibility set.

## Configuration shape

The manifest passes this configuration to `trtllm-serve`:

```yaml
checkpoint_format: MX
mx_config:
  server_url: modelexpress-server:8001
```

The first ready worker loads from the configured checkpoint and publishes its post-transform tensors. Later compatible workers query the ModelExpress server and receive their rank-matched tensors over NIXL. If no source is ready, the worker uses TensorRT-LLM's native checkpoint path.

## Verification

After applying the manifest, wait for the first worker to become ready before scaling. On a target worker, inspect logs for:

```text
Eligible loaders: ['rdma', 'default']
Trying strategy: rdma
RDMA transfer complete
TRT-LLM MxModelLoader.load_model() COMPLETE
```

The exact timing prefix may vary by image. If the eligible list contains only `default`, check the server address, source identity, NIXL availability, and RDMA resources before debugging TensorRT-LLM model code.

## Compatibility boundary

TensorRT-LLM is not currently covered by ModelExpress CI for P2P, ModelStreamer, GDS, or Dynamo. Validate the full upstream TensorRT-LLM MX change set and the target GPU/fabric combination before production use. See [Compatibility](../../COMPATIBILITY.md) for the current support matrix.
