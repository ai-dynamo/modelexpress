# ModelExpress Client Deployment

Inference engine instances with ModelExpress P2P weight transfer support.

## Client Images

| Engine | Dockerfile |
|--------|------------|
| **vLLM** | [`vllm/Dockerfile`](vllm/Dockerfile) |
| **SGLang** | [`sglang/Dockerfile`](sglang/Dockerfile) |

For ModelStreamer-only examples that load from Azure Blob Storage, S3, or a local PVC without an MX server, see [`../../model_streamer_k8s/`](../../model_streamer_k8s/).

## vLLM Deployments

The vLLM image is pinned to 0.23.0. These manifests use vLLM's native `modelexpress` load format and do not require `VLLM_PLUGINS`.

| Topology | Manifest | Model | Configuration |
|----------|----------|-------|---------------|
| **Single-node** | [`vllm/vllm-single-node.yaml`](vllm/vllm-single-node.yaml) | DeepSeek-V4-Pro | TP=8, P2P weights + JIT caches |
| **Multi-node** | [`vllm/vllm-multi-node.yaml`](vllm/vllm-multi-node.yaml) | DeepSeek-V4-Pro | TP=4, PP=2, P2P weights + JIT caches |

## SGLang Deployments

| Topology | Manifest | Model | Configuration |
|----------|----------|-------|---------------|
| **Single-node NIXL** | [`sglang/sglang-single-node-p2p.yaml`](sglang/sglang-single-node-p2p.yaml) | Kimi-K2.5-NVFP4 | TP=8, 1 node (8 GPUs) |
| **Single-node Mooncake TransferEngine** | [`sglang/sglang-single-node-transfer-engine.yaml`](sglang/sglang-single-node-transfer-engine.yaml) | Kimi-K2.5-NVFP4 | TP=8, 1 node (8 GPUs) |

The SGLang manifests use the same `remote_instance` command for the first and
later replicas. `modelexpress-config` selects `nixl` or `transfer_engine`; the
ModelExpress server endpoint is provided through `MX_SERVER_ADDRESS`.

## How It Works

On startup, the engine-specific ModelExpress loader auto-detects the best loading strategy:

1. **RDMA** -- If a ready source exists for this model/rank, receive weights via NIXL
2. **GDS** -- If GPUDirect Storage is available, load directly from file to GPU
3. **Disk** -- Standard engine-native weight loading as final fallback

After loading, every worker publishes its metadata so future instances can discover it as an RDMA source.

When the server uses the Kubernetes metadata backend, the example manifests
also inject `POD_NAME`, `POD_UID`, and `POD_NAMESPACE` through the Downward API.
ModelExpress uses a complete same-namespace identity to make the Pod the owner
of each `ModelMetadata` CR it publishes, including artifact metadata. This
ownership behavior is independent of the inference engine.
Deleting the Pod then removes its metadata through Kubernetes garbage
collection. If any identity field is unavailable, publication continues without
an owner reference so older clients and non-Kubernetes deployments remain
compatible.

## Prerequisites

- ModelExpress server deployed (see [`../server/`](../server/))
- HuggingFace token secret: `kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=<token>`
- PVC with model weights (see [`../model-download.yaml`](../model-download.yaml))

## Verify

The SGLang deployment defines a readiness probe on `/health`, so it reports
Ready only after warmup completes. Wait for that, then confirm the
OpenAI-compatible API serves the expected model:

```bash
kubectl rollout status deployment/mx-sglang --timeout=20m
kubectl exec deployment/mx-sglang -c sglang -- \
  curl -sS http://localhost:8000/v1/models
```

The vLLM examples do not define a readiness probe, so `rollout status` returns
as soon as the container starts. Poll the same `curl` (swapping in the vLLM
deployment and container names) until it returns the model list.
