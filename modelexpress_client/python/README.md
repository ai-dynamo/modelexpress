# ModelExpress Python Client

Python client for ModelExpress -- high-performance GPU-to-GPU model weight transfers using NVIDIA NIXL over RDMA/InfiniBand.

Instead of each vLLM instance loading model weights from storage, one "source" instance loads the model and transfers weights directly to "target" instances via GPUDirect RDMA, bypassing the CPU entirely.

## Installation

```bash
# From PyPI (coming soon)
pip install modelexpress

# Editable install from source
pip install -e .

# With dev dependencies (pytest, grpcio-tools)
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.10
- NVIDIA GPUs with RDMA/InfiniBand support
- [NIXL](https://github.com/ai-dynamo/nixl) (NVIDIA Interconnect eXchange Library)
- A running [ModelExpress server](https://github.com/ai-dynamo/modelexpress/tree/main/modelexpress_server) (Rust gRPC service backed by Redis or Kubernetes CRDs)

## Quick Start with vLLM

ModelExpress integrates with vLLM via custom model loaders. Set `MX_REGISTER_LOADERS=1` to auto-register them, or call `register_modelexpress_loaders()` in your code.

```bash
export MODEL_EXPRESS_URL="modelexpress-server:8001"

vllm serve deepseek-ai/DeepSeek-V3 \
    --load-format mx \
    --tensor-parallel-size 8 \
    --worker-cls modelexpress.vllm_worker.ModelExpressWorker
```

Starting the vLLM engine with `mx` loader on the source worker will load the weights from disk, register tensors with NIXL, start a per-worker gRPC server for tensor manifests, and publish lightweight metadata to the MX server.
On the target worker, it will discover the source via the MX server, fetch the tensor manifest directly from the source worker, and stream weights over RDMA from GPU to GPU.

## Programmatic Usage

### MxClient

`MxClient` is a lightweight gRPC client for communicating with the ModelExpress server:

```python
from modelexpress import MxClient

client = MxClient(server_url="modelexpress-server:8001")

# List available sources for a model identity
sources = client.list_sources(identity)
for src in sources:
    print(f"Source {src.mx_source_id} worker {src.worker_rank}")

# Get metadata for a specific worker
response = client.get_metadata(mx_source_id="abc123...", worker_id="uuid-...")
if response.found:
    worker = response.worker
    print(f"Worker rank {worker.worker_rank}, endpoint: {worker.metadata_endpoint}")

client.close()
```

### Registering Loaders Manually

```python
from modelexpress import register_modelexpress_loaders

register_modelexpress_loaders()
# Now vLLM recognizes --load-format mx-source and mx-target
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_EXPRESS_URL` | `localhost:8001` | ModelExpress gRPC server address |
| `MX_SERVER_ADDRESS` | `localhost:8001` | Backward-compatible alias for `MODEL_EXPRESS_URL` |
| `MX_REGISTER_LOADERS` | `1` | Auto-register `mx` loader with vLLM |
| `MX_METADATA_PORT` | `5555` | Base port for NIXL P2P metadata exchange (per-worker: base + rank) |
| `MX_WORKER_ADDRESS` | (auto-detect) | Worker IP/hostname for NIXL listen thread and gRPC endpoint |
| `MX_WORKER_GRPC_PORT` | `6555` | Base port for per-worker gRPC server (per-worker: base + rank) |

### UCX/NIXL Tuning

| Variable | Recommended | Description |
|----------|-------------|-------------|
| `UCX_TLS` | `rc_x,rc,dc_x,dc,cuda_copy` | Transport layers for InfiniBand |
| `UCX_RNDV_SCHEME` | `get_zcopy` | Zero-copy RDMA reads |
| `UCX_RNDV_THRESH` | `0` | Force rendezvous for all transfers |
| `NIXL_LOG_LEVEL` | `INFO` | NIXL logging level |

## Package Structure

| Module | Description |
|--------|-------------|
| `modelexpress.client` | `MxClient` -- gRPC client for the ModelExpress server |
| `modelexpress.vllm_loader` | `MxModelLoader` -- vLLM integration |
| `modelexpress.nixl_transfer` | `NixlTransferManager` -- NIXL agent lifecycle and RDMA transfers |
| `modelexpress.types` | `TensorDescriptor`, `WorkerMetadata` -- core data types |
| `modelexpress.vllm_worker` | vLLM worker extensions |

## How It Works

1. **Source** loads weights from disk, runs `process_weights_after_loading()`, registers the fully-processed tensors with NIXL, starts a per-worker gRPC server for tensor manifests, and publishes lightweight metadata to the ModelExpress server.
2. **Target** creates dummy weights, runs `process_weights_after_loading()` on dummies to establish the correct tensor layout, then discovers a ready source via the MX server.
3. **Target** fetches the tensor manifest directly from the source worker's gRPC server, fetches the NIXL agent blob via native P2P exchange, and pulls processed weights via RDMA.

This post-processing transfer strategy is critical for FP8 models (e.g., DeepSeek-V3) where `weight_scale_inv` tensors are renamed and transformed during processing. Transferring after processing avoids ~50GB/worker of wasted GPU memory from pinned pre-processing tensors.

## License

Apache-2.0
