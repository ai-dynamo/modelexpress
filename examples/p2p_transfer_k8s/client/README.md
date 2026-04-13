# ModelExpress Client Deployment

vLLM instances with ModelExpress P2P weight transfer support.

## vLLM Deployments

| Topology | Manifest | Model | Configuration |
|----------|----------|-------|---------------|
| **Single-node** | [`vllm/vllm-single-node.yaml`](vllm/vllm-single-node.yaml) | Llama-3.1-405B-Instruct | TP=8, 1 node (8 GPUs) |
| **Multi-node** | [`vllm/vllm-multi-node.yaml`](vllm/vllm-multi-node.yaml) | Llama-3.1-405B-Instruct | TP=4, PP=2, 2 nodes (8 GPUs) |

## How It Works

On startup, the `mx` loader (`--load-format mx`) auto-detects the best loading strategy:

1. **RDMA** -- If a ready source exists for this model/rank, receive weights via NIXL
2. **GDS** -- If GPUDirect Storage is available, load directly from file to GPU
3. **Disk** -- Standard vLLM weight loading as final fallback

After loading, every worker publishes its metadata so future instances can discover it as an RDMA source.

## Prerequisites

- ModelExpress server deployed (see [`../server/`](../server/))
- HuggingFace token secret: `kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=<token>`
- PVC with model weights (see [`../model-download.yaml`](../model-download.yaml))
