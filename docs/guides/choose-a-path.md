<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Choose a ModelExpress path

Start with what you want to do:

- **Start inference replicas faster:** one worker loads a checkpoint, then compatible workers copy its ready GPU weights. Follow the [vLLM Kubernetes quickstart](../integrations/runtimes/vllm.md), or choose [SGLang](../integrations/runtimes/sglang.md) or [TensorRT-LLM](../integrations/runtimes/tensorrt-llm.md).
- **Update rollout workers during RL training:** publish a weight version, install it on the required rollout workers, then resume generation. Start with [RL weight updates](rl.md).

Inference startup and an RL weight update are different operations. Selecting the `modelexpress` loader starts an inference worker; updating a running rollout worker also requires the training framework's version and refit lifecycle.

## Inference: load once, then copy from a peer

The first worker loads weights from storage and publishes its availability. Later compatible workers discover that source and receive post-processed tensors directly over NIXL. Keep the source running while targets start. The ModelExpress server coordinates discovery; these P2P weight bytes do not pass through it.

For a first deployment, use the central server with Redis in the [vLLM quickstart](../integrations/runtimes/vllm.md). After that works, choose a [Kubernetes metadata backend](../../examples/p2p_transfer_k8s/server/README.md) or an orchestrator integration such as [Dynamo](../integrations/orchestrators/dynamo.md) or [llm-d](../integrations/orchestrators/llm-d.md). The serverless [`k8s-service` backend](../../examples/k8s_service_sources/README.md) is an advanced option for fixed, interchangeable model revisions.

Source and target must use compatible runtime, model revision, dtype, and parallelism settings. Targets still need model configuration and tokenizer files. P2P does not require a shared model filesystem, but it does not make a worker automatically offline-capable: package non-weight files locally or use the [server-backed no-shared-storage setup](../DEPLOYMENT.md#server-backed-model-cache-no-shared-storage).

A healthy target alone does not prove P2P succeeded: inference loading can fall back to storage. Check its logs for `RDMA transfer complete` and exercise its inference API, as shown in the quickstart.

## Inference: read weights directly from storage

Use ModelStreamer when workers should read safetensors from S3, GCS, Azure Blob Storage, or an absolute local path. Set `MX_MODEL_URI` on the **inference worker**, with credentials available to that worker. This path is storage → worker; it does not need a ModelExpress server, Redis, a shared PVC, or RDMA. Add a server address and a working NIXL transport if the loaded worker should also become a P2P source.

Start with the [vLLM](../../examples/model_streamer_k8s/client/vllm/README.md) or [SGLang](../../examples/model_streamer_k8s/client/sglang/README.md) storage examples. The runtime still needs configuration and tokenizer files. For SGLang, keep `--model-path` on the model identity or configuration path and supply the storage URI through `MX_MODEL_URI`; an object-storage `--model-path` selects SGLang's native loader instead.

Verify `Trying strategy: model_streamer` followed by `Model streamer weight loading complete`. If the logs show another loader, check that the worker image includes the expected MX version and runtime integration. Setting a URI alone does not prove the storage path ran.

## Defaults and optional configuration

`MX_LOAD_STRATEGY_CHAIN=INFERENCE` is the default. Its fixed order is P2P, server cache, InstantTensor, ModelStreamer, GDS, then the runtime's native loader; unavailable strategies are skipped and recoverable failures can fall through. With an object-storage `MX_MODEL_URI`, current MX skips InstantTensor automatically. A local path can still be eligible for InstantTensor first.

`MX_LOAD_STRATEGY_CHAIN=RL` selects the separate RL startup policy. When an exact desired version UID is configured, startup must load that version through the supported RL sources or fail; it must not silently use an unrelated checkpoint. See the [RL guide](rl.md) for integration requirements.

Without RDMA, storage loading remains available. P2P eligibility checks package, device, and discovery support; they do not prove the network is usable. A failed transport may trigger fallback, so inspect completion logs before claiming a P2P result. See [Configuration](../CONFIGURATION.md#loading-strategy-selection) and [Troubleshooting](../TROUBLESHOOTING.md#loader-selection) for details.
