# ModelExpress Python Client

Python client for ModelExpress -- high-performance GPU-to-GPU model weight transfers using NVIDIA NIXL over RDMA/InfiniBand.

Instead of each vLLM instance loading model weights from storage, one "source" instance loads the model and transfers weights directly to "target" instances via GPUDirect RDMA, bypassing the CPU entirely.

## Installation

```bash
pip install modelexpress
```

### Development

```bash
# Editable install from source
pip install -e .

# With dev dependencies (pytest, grpcio-tools)
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.10
- NVIDIA GPUs with RDMA/InfiniBand support
- [NIXL](https://github.com/ai-dynamo/nixl) (NVIDIA Interconnect eXchange Library)
- A running [ModelExpress server](https://github.com/ai-dynamo/modelexpress/tree/main/modelexpress_server) (Rust gRPC service backed by Redis)

## Quick Start with vLLM

ModelExpress integrates with vLLM via custom model loaders. Set `MX_REGISTER_LOADERS=1` to auto-register them, or call `register_modelexpress_loaders()` in your code.

### Source Instance (loads weights from disk, serves via RDMA)

```bash
export MODEL_EXPRESS_URL="modelexpress-server:8001"
export MODEL_NAME="deepseek-ai/DeepSeek-V3"
export MX_EXPECTED_WORKERS=8

vllm serve deepseek-ai/DeepSeek-V3 \
    --load-format mx-source \
    --tensor-parallel-size 8 \
    --worker-cls modelexpress.vllm_worker.ModelExpressWorker
```

### Target Instance (receives weights via RDMA)

```bash
export MODEL_EXPRESS_URL="modelexpress-server:8001"
export MODEL_NAME="deepseek-ai/DeepSeek-V3"
export MX_EXPECTED_WORKERS=8

vllm serve deepseek-ai/DeepSeek-V3 \
    --load-format mx-target \
    --tensor-parallel-size 8 \
    --worker-cls modelexpress.vllm_worker.ModelExpressWorker
```

The target waits for the source to publish its NIXL ready flag, then pulls weights over RDMA. For DeepSeek-V3 (681 GB across 8 GPUs), this takes ~15 seconds vs ~25 minutes from NVMe storage.

## Programmatic Usage

### MxClient

`MxClient` is a lightweight gRPC client for communicating with the ModelExpress server:

```python
from modelexpress import MxClient

client = MxClient(server_url="modelexpress-server:8001")

# Query for a source model
response = client.get_metadata("deepseek-ai/DeepSeek-V3")
if response.found:
    for worker in response.workers:
        print(f"Worker rank {worker.worker_rank}: {len(worker.tensors)} tensors")

# Wait for source readiness (blocks until ready or timeout)
success, session_id, metadata_hash = client.wait_for_ready(
    model_name="deepseek-ai/DeepSeek-V3",
    worker_id=0,
    timeout_seconds=7200,
)

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
| `MODEL_NAME` | _(required)_ | Model name used for source/target coordination |
| `MX_REGISTER_LOADERS` | `1` | Auto-register `mx-source`/`mx-target` loaders with vLLM |
| `MX_EXPECTED_WORKERS` | Auto-detected from TP size | Number of GPU workers to coordinate |
| `MX_SYNC_PUBLISH` | `0` | Source: wait for all workers before publishing metadata |
| `MX_SYNC_START` | `1` | Target: wait for all source workers before transferring |
| `MX_CONTIGUOUS_REG` | `0` | Enable contiguous region registration (experimental) |

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
| `modelexpress.vllm_loader` | `MxSourceModelLoader`, `MxTargetModelLoader` -- vLLM integration |
| `modelexpress.nixl_transfer` | `NixlTransferManager` -- NIXL agent lifecycle and RDMA transfers |
| `modelexpress.types` | `TensorDescriptor`, `WorkerMetadata` -- core data types |
| `modelexpress.vllm_worker` | vLLM worker extensions |

## How It Works

1. **Source** loads weights from disk, registers raw tensors with NIXL *before* FP8 processing, and publishes metadata to the ModelExpress server.
2. **Target** creates dummy weights, waits for the source ready flag, then pulls raw tensors via RDMA read.
3. Both source and target run `process_weights_after_loading()` independently, producing identical FP8-transformed weights.

This pre-processing transfer strategy is critical for FP8 models (e.g., DeepSeek-V3) where `weight_scale_inv` tensors are renamed and transformed during processing.

## License

Apache-2.0
