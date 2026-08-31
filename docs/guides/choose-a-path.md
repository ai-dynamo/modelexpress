<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Choose a ModelExpress path

This guide answers the deployment questions that usually come before a configuration reference: where the model bytes come from, whether replicas need shared storage, and which parts of the loading behavior are configurable.

## Quick decision table

| Your situation | Recommended path | What you need |
|---|---|---|
| A first replica can read the model and later replicas should start quickly | [P2P weight transfer](../integrations/loading-paths/p2p.md) | Compatible runtime images, NIXL, a supported fabric, and metadata coordination |
| Replicas cannot share a filesystem but can reach each other over RDMA | P2P with a central `redis` or `kubernetes` metadata backend | One source must bootstrap from Hugging Face, object storage, or a local path; targets need not read the checkpoint bytes |
| Every worker should read the checkpoint from object storage | [ModelStreamer](../integrations/loading-paths/model-streamer.md) | `MX_MODEL_URI`, storage credentials, and a ModelExpress-enabled runtime image |
| Workers cannot reach Hugging Face and do not have a shared cache | Server-backed model cache | An MX server with upstream access, `MODEL_EXPRESS_NO_SHARED_STORAGE=1`, and a worker-local writable cache |
| Weights are fixed for pod lifetime and you want no central MX server | `k8s-service` backend | Kubernetes Services, rank-aware manifests, and stable model revisions |
| You are deploying through an orchestrator | [Dynamo](../integrations/orchestrators/dynamo.md) or [llm-d](../integrations/orchestrators/llm-d.md) | The orchestrator's operator, image, and deployment contract |

## I need replicas without shared model storage

Yes. ModelExpress P2P does not require the target replicas to mount the source replica's model filesystem. The first compatible replica loads the model from whatever source it can access, publishes tensor metadata, and later replicas pull the post-processed weights directly from GPU memory over NIXL. The MX server stores coordination metadata; it does not proxy the weight bytes.

The central-coordinator topology uses Redis or Kubernetes CRDs. The decentralized `k8s-service` topology also avoids a central server, but it is intended for stable-weight inference and requires a Kubernetes Service layout that matches the worker ranks. Choose the central topology for mixed revisions, per-worker addressability, or live-update workflows.

### What the target still needs

P2P transfers weight tensors, not the entire model repository. Each target still needs an image with the runtime and ModelExpress client, enough model configuration for the runtime to construct the model, and network access to the ModelExpress metadata endpoint or the `k8s-service` source. If the target cannot resolve model metadata from Hugging Face or a shared cache, use the [server-backed model cache](../DEPLOYMENT.md#server-backed-model-cache-no-shared-storage) or package the non-weight files with the deployment.

### Central-coordinator setup

1. Install the ModelExpress CRDs when using the Kubernetes backend: `kubectl apply -f examples/crds.yaml`.
2. Deploy the server with either [`modelexpress-server-kubernetes.yaml`](../../examples/p2p_transfer_k8s/server/kubernetes_backend/modelexpress-server-kubernetes.yaml) or [`modelexpress-server-redis.yaml`](../../examples/p2p_transfer_k8s/server/redis_backend/modelexpress-server-redis.yaml).
3. Build a worker image from [`examples/p2p_transfer_k8s/client/vllm/Dockerfile`](../../examples/p2p_transfer_k8s/client/vllm/Dockerfile), or use the equivalent SGLang or TensorRT-LLM image recipe.
4. Set `MX_SERVER_ADDRESS` to the server Service address and give the first replica a model source.
5. Apply [`vllm-single-node.yaml`](../../examples/p2p_transfer_k8s/client/vllm/vllm-single-node.yaml), wait for the first replica to become ready, and scale the Deployment.

The checked-in manifest uses a PVC for the bootstrap path, but the P2P contract itself does not require the PVC to be shared by replicas. For an object-storage bootstrap, set `MX_MODEL_URI` as described in [Can I load from S3, GCS, Azure Blob, or a local path?](#can-i-load-from-s3-gcs-azure-blob-or-a-local-path) and add the storage credentials to the worker.

### No server, Kubernetes Service routing

Use the [`k8s-service` examples](../../examples/k8s_service_sources/README.md) when all pods behind a source Service hold the same fixed revision for their entire lifetime. The source pool is exposed through rank-aware Service ports or selectors, and the client calls the worker directly. There is no Redis, ModelExpress server, or CRD in this topology.

## Can I load from S3, GCS, Azure Blob, or a local path?

Yes. Set `MX_MODEL_URI` to a ModelStreamer URI. The ModelStreamer strategy supports `s3://`, `gs://`, `az://`, and absolute local paths. A direct ModelStreamer deployment does not require a ModelExpress server, Redis, or P2P resources. vLLM's checked-in recipes pass the URI as `--model`; SGLang keeps `--model-path` on the model identity or local configuration path and passes the storage URI through `MX_MODEL_URI`.

### S3

```bash
export MX_MODEL_URI=s3://my-bucket/models/my-model
export AWS_DEFAULT_REGION=us-west-2
vllm serve s3://my-bucket/models/my-model --load-format modelexpress
```

The process needs AWS SDK credentials through the environment, workload identity, or the runtime's normal credential chain. For Kubernetes, start with the vLLM [`s3` manifest](../../examples/model_streamer_k8s/client/vllm/vllm-single-node-streamer-s3.yaml) or the SGLang [`s3` manifest](../../examples/model_streamer_k8s/client/sglang/sglang-single-node-streamer-s3.yaml). The current CI workflow exercises both direct S3 streaming and S3 failure fallback for vLLM and SGLang.

### GCS and Azure Blob

Use `gs://bucket/prefix` with Google Application Default Credentials or `az://container/prefix` with Azure `DefaultAzureCredential`. The [`model_streamer_k8s` examples](../../examples/model_streamer_k8s/README.md) contain Azure, S3, and local/PVC recipes. The ModelExpress server is not involved in direct streaming, so credentials stay in the worker environment.

### Local disk or PVC

Set `MX_MODEL_URI` to an absolute path or to a Hugging Face model ID that resolves through the runtime's local cache. Use the [`local vLLM manifest`](../../examples/model_streamer_k8s/client/vllm/vllm-single-node-streamer-local.yaml) or [`local SGLang manifest`](../../examples/model_streamer_k8s/client/sglang/sglang-single-node-streamer-local.yaml). For tensor parallelism greater than one, leave `MX_MS_DISTRIBUTED=1` so ModelStreamer divides reads across ranks; at TP=1 the setting has no effect.

## Can I configure the loader?

You can configure which paths are eligible and how they behave, but the Python client does not expose an arbitrary user-defined loader order. The current chain is fixed in [`LoadStrategyChain`](../../modelexpress_client/python/modelexpress/load_strategy/__init__.py): P2P RDMA, server cache, InstantTensor, ModelStreamer, GDS, then the runtime's native loader.

The most common controls are:

| Goal | Setting |
|---|---|
| Enable a P2P metadata path | `MX_SERVER_ADDRESS` plus `MX_METADATA_BACKEND=redis` or `kubernetes`, or `MX_METADATA_BACKEND=k8s-service` |
| Disable embedded/full P2P metadata exchange | `MX_P2P_METADATA=0` with a central coordinator |
| Prefer ModelStreamer for a storage bootstrap | Set `MX_MODEL_URI` |
| Disable InstantTensor | `MX_INSTANT_TENSOR=0` |
| Enable server-backed repository files for offline workers | `MODEL_EXPRESS_NO_SHARED_STORAGE=1` and `MODEL_EXPRESS_URL` or `MX_SERVER_ADDRESS` |
| Enable GDS | Use a GDS-capable accelerator and a runtime adapter that supports the GDS path |
| Make the native loader the fallback | Leave the optional paths ineligible or let earlier strategies fail; the native loader remains the final fallback when the adapter supports it |

Eligibility is also determined by the runtime adapter, installed packages, device capabilities, model format, and metadata reachability. A setting does not force a path that the environment cannot support. Enable `MODEL_EXPRESS_LOG_LEVEL=DEBUG` and look for `Eligible loaders` and `Trying strategy` to see the decision for a specific worker.

## I need to deploy through Dynamo

The Dynamo integration uses the same ModelExpress vLLM loader and adds Dynamo's `DynamoGraphDeployment` lifecycle around it. Start with the [Dynamo integration guide](../integrations/orchestrators/dynamo.md), then choose the aggregated or disaggregated manifest in [`examples/dynamo_p2p_transfer_k8s`](../../examples/dynamo_p2p_transfer_k8s/README.md). vLLM aggregated and disaggregated Dynamo P2P paths are active in the current ModelExpress CI workflow.

## I need to deploy through llm-d

llm-d has a runnable [ModelExpress P2P guide](https://github.com/llm-d/llm-d/tree/main/guides/modelexpress-p2p) that composes MX with its Optimized Baseline. Use the [llm-d integration page](../integrations/orchestrators/llm-d.md) for the boundary between the two projects, version alignment, and what ModelExpress itself validates.

## I only need model-cache management

Use the standalone server and CLI path when you want to prewarm or inspect a model cache without an inference runtime. Start with [Deployment](../DEPLOYMENT.md) and [CLI](../CLI.md). The server's registry backend is required at startup: choose Redis with `REDIS_URL`, or Kubernetes with `POD_NAMESPACE`/`MX_METADATA_NAMESPACE` and the CRDs installed.

## I do not have RDMA or GDS

ModelExpress remains usable without those accelerators. P2P and GDS become ineligible, and the chain can use server cache, InstantTensor when available, ModelStreamer when `MX_MODEL_URI` is set, or the runtime's native loader. For the supported path in your environment, check [Compatibility](../COMPATIBILITY.md) rather than assuming that a successful package install implies hardware support.
