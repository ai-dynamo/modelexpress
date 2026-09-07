<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Choose a ModelExpress path

Choose based on where the weights start and what the target workers can reach.

| Scenario | Use | Main requirement |
|---|---|---|
| One replica can load the model; later replicas should start quickly | P2P | NIXL, a supported fabric, and source discovery |
| Replicas cannot share a filesystem | P2P or server-backed cache | Targets still need model configuration and non-weight files |
| Every worker should read from object storage | ModelStreamer | `MX_MODEL_URI`, credentials, and safetensors |
| Weights are fixed and you do not want a central MX server | `k8s-service` | Stable revisions and rank-aware Kubernetes Services |
| You only need cache management | Standalone server and CLI | Redis or Kubernetes metadata backend |
| An orchestrator owns worker lifecycle | [Dynamo](../integrations/orchestrators/dynamo.md) or [llm-d](../integrations/orchestrators/llm-d.md) | Orchestrator operator, runtime image, and MX configuration |

## P2P without shared storage

P2P does not require target replicas to mount the source model filesystem. The first compatible worker loads the model, publishes metadata, and later workers receive post-processed tensors directly from GPU memory over NIXL. The ModelExpress server coordinates discovery; weight bytes do not pass through it.

Targets still need the runtime and ModelExpress client, model configuration and tokenizer files, and connectivity to the metadata endpoint. If targets cannot obtain non-weight repository files, use the [server-backed no-shared-storage path](../DEPLOYMENT.md#server-backed-model-cache-no-shared-storage) or package those files in the image.

For the central topology, deploy a [Redis or Kubernetes-backed server](../../examples/p2p_transfer_k8s/server/README.md), build a runtime image, apply the matching [P2P example](../../examples/p2p_transfer_k8s/README.md), wait for the first replica to become ready, then scale. Use the [`k8s-service` examples](../../examples/k8s_service_sources/README.md) only when source pods hold stable, interchangeable revisions.

## Load from object storage or a local path

Set `MX_MODEL_URI` to `s3://`, `gs://`, `az://`, or an absolute local path. ModelStreamer reads safetensors directly in the worker, so direct storage loading does not require a ModelExpress server, Redis, a PVC, or RDMA. Add `MX_SERVER_ADDRESS` and fabric resources only if the loaded worker should become a P2P source.

For vLLM, the storage URI can be the model argument. For SGLang, keep `--model-path` on the model identity or configuration path and pass the storage URI only through `MX_MODEL_URI`; using an object-storage URI as `--model-path` selects SGLang's native loader instead.

Start with the checked-in [vLLM](../../examples/model_streamer_k8s/client/vllm/README.md) or [SGLang](../../examples/model_streamer_k8s/client/sglang/README.md) manifests. Credentials use the storage SDK's normal environment, workload identity, or secret chain. Leave `MX_MS_DISTRIBUTED=1` to divide ModelStreamer reads across CUDA tensor-parallel ranks.

## Configure loader behavior

ModelExpress uses a fixed strategy order: P2P, server cache, InstantTensor, ModelStreamer, GDS, then the runtime's native loader. You configure whether a path is eligible and how it behaves; you do not supply an arbitrary order.

| Goal | Setting |
|---|---|
| Connect to a central P2P coordinator | `MX_SERVER_ADDRESS` and `MX_METADATA_BACKEND=redis` or `kubernetes` |
| Use decentralized source discovery | `MX_METADATA_BACKEND=k8s-service` |
| Fetch repository files through the server | `MODEL_EXPRESS_NO_SHARED_STORAGE=1` plus a server address |
| Stream from storage | `MX_MODEL_URI` |
| Disable InstantTensor | `MX_INSTANT_TENSOR=0` |
| Inspect the decision | `MODEL_EXPRESS_LOG_LEVEL=DEBUG` |

Eligibility also depends on the runtime adapter, installed packages, device, model format, and metadata reachability. See [Configuration](../CONFIGURATION.md#loading-strategy-selection) for defaults and [Troubleshooting](../TROUBLESHOOTING.md#loader-selection) for the relevant log messages.

## Without RDMA or GDS

ModelExpress remains usable. P2P and GDS are skipped, while server cache, InstantTensor, ModelStreamer, or the native runtime loader can still run when eligible.
