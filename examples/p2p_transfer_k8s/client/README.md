# ModelExpress Client Deployment

Inference engine instances with ModelExpress P2P weight transfer support.

## Client Images

| Engine | Dockerfile |
|--------|------------|
| **vLLM** | [`vllm/Dockerfile`](vllm/Dockerfile) |
| **SGLang** | [`sglang/Dockerfile`](sglang/Dockerfile) |

For ModelStreamer-only examples that load from Azure Blob Storage, S3, or a local PVC without an MX server, see [`../../model_streamer_k8s/`](../../model_streamer_k8s/).

## vLLM Deployments

| Topology | Manifest | Model | Configuration |
|----------|----------|-------|---------------|
| **Single-node** | [`vllm/vllm-single-node.yaml`](vllm/vllm-single-node.yaml) | Llama-3.1-405B-Instruct | TP=8, 1 node (8 GPUs) |
| **Multi-node** | [`vllm/vllm-multi-node.yaml`](vllm/vllm-multi-node.yaml) | Llama-3.1-405B-Instruct | TP=4, PP=2, 2 nodes (8 GPUs) |

## SGLang Deployments

| Topology | Manifest | Model | Configuration |
|----------|----------|-------|---------------|
| **Single-node** | [`sglang/sglang-single-node-p2p.yaml`](sglang/sglang-single-node-p2p.yaml) | Kimi-K2.5-NVFP4 | TP=8, 1 node (8 GPUs) |

## How It Works

On startup, the engine-specific ModelExpress loader auto-detects the best loading strategy:

1. **RDMA** -- If a ready source exists for this model/rank, receive weights via NIXL
2. **GDS** -- If GPUDirect Storage is available, load directly from file to GPU
3. **Disk** -- Standard engine-native weight loading as final fallback

After loading, every worker publishes its metadata so future instances can discover it as an RDMA source.

## Prerequisites

- ModelExpress server deployed (see [`../server/`](../server/))
- HuggingFace token secret: `kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=<token>`
- PVC with model weights (see [`../model-download.yaml`](../model-download.yaml))
