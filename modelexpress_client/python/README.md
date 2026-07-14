# ModelExpress Python Client

Python client for ModelExpress -- high-performance GPU-to-GPU model weight transfers using NVIDIA NIXL over RDMA/InfiniBand.

Instead of each inference engine instance loading model weights from storage,
one instance loads the model and transfers weights directly to later instances
via GPUDirect RDMA, bypassing the CPU entirely.

## Installation

```bash
# From PyPI (coming soon)
pip install modelexpress

# Editable install from source
pip install -e .

# With dev dependencies (pytest, grpcio-tools)
pip install -e ".[dev]"
```

NIXL is expected to be supplied by the runtime environment (TRT-LLM,
SGLang, Dynamo, and NemoRL runtime images all ship `nixl-cu12` or
`nixl-cu13`). For a bare-environment install, run `pip install nixl-cu12`
or `pip install nixl-cu13` separately, matching your host CUDA toolkit.

### Requirements

- Python >= 3.10
- NVIDIA GPUs with RDMA/InfiniBand support
- [NIXL](https://github.com/ai-dynamo/nixl) (NVIDIA Interconnect eXchange Library)
- A running [ModelExpress server](https://github.com/ai-dynamo/modelexpress/tree/main/modelexpress_server) (Rust gRPC service backed by Redis)

## Quick Start with vLLM

vLLM 0.23.0 and newer recognize `--load-format modelexpress` natively. Install the ModelExpress Python package in the vLLM image; no `VLLM_PLUGINS` setting or manual loader registration is required. For older vLLM versions, set `VLLM_PLUGINS=modelexpress` or call `register_modelexpress_loaders()` manually.

```bash
export MX_SERVER_ADDRESS="modelexpress-server:8001"

vllm serve deepseek-ai/DeepSeek-V4-Pro \
    --load-format modelexpress \
    --tensor-parallel-size 8 \
    --trust-remote-code
```

Starting the vLLM engine with the `modelexpress` load format on the source worker will load the weights from disk and register/publish the NIXL and tensor metadata to the MX server. The `mx` load format is kept as a backward-compatible alias.
On the target worker, it retrieves metadata from the MX server and streams weights over RDMA from GPU to GPU. Set `MX_ARTIFACT_TRANSFER=1` to also reuse compatible vLLM JIT caches from a ready source.

## Quick Start with SGLang

SGLang integrates through its `remote_instance` loader with the `modelexpress`
backend. Use an SGLang image that includes upstream sgl-project/sglang#24723,
such as the known-good release image `lmsysorg/sglang:v0.5.13.post1`, and
install the ModelExpress package into that image.

```bash
export MX_SERVER_ADDRESS="modelexpress-server:8001"

python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --tp 8 \
    --load-format remote_instance \
    --remote-instance-weight-loader-backend modelexpress \
    --modelexpress-config '{"transport": "nixl"}'
```

## Programmatic Usage

### MxClient

`MxClient` is a lightweight gRPC client for communicating with the ModelExpress server:

```python
from modelexpress import MxClient

client = MxClient(server_url="modelexpress-server:8001")

# Query for a source model
response = client.get_metadata("deepseek-ai/DeepSeek-V4-Pro")
if response.found:
    for worker in response.workers:
        print(f"Worker rank {worker.worker_rank}: {len(worker.tensors)} tensors")

# Wait for source readiness (blocks until ready or timeout)
success, session_id, metadata_hash = client.wait_for_ready(
    model_name="deepseek-ai/DeepSeek-V4-Pro",
    worker_id=0,
    timeout_seconds=7200,
)

client.close()
```

### Registering Loaders Manually

Manual registration is only needed for integrations that construct vLLM loaders outside vLLM 0.23.0's native load-format path.

```python
from modelexpress import register_modelexpress_loaders

register_modelexpress_loaders()
# Now vLLM recognizes --load-format modelexpress and mx
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MX_SERVER_ADDRESS` | `localhost:8001` | ModelExpress gRPC server address (recommended) |
| `MODEL_EXPRESS_URL` | `localhost:8001` | Deprecated, pending removal in a future release. Still read by all client paths and takes precedence when both are set; keep setting it during the transition. |
| `MX_EXPECTED_WORKERS` | Auto-detected from TP size | Number of GPU workers to coordinate |
| `MX_SYNC_PUBLISH` | `0` | Source: wait for all workers before publishing metadata |
| `MX_SYNC_START` | `1` | Target: wait for all source workers before transferring |
| `MX_POOL_REG` | `0` | Allocation-level NIXL registration (registers cudaMalloc blocks instead of individual tensors) |
| `MX_P2P_METADATA` | `1` | Serve tensor and artifact manifests directly from source workers; set to `0` to route full tensor metadata through the central server |
| `MX_ARTIFACT_TRANSFER` | `0` | Transfer compatible vLLM TorchInductor, Triton, DeepGEMM, TileLang, CuTe DSL, and FlashInfer JIT caches, including persistent autotune files when supported by vLLM |
| `MX_ARTIFACT_BUNDLE_ROOT` | `$TMPDIR/modelexpress-artifacts` | Staging root for tarred cache artifact bundles |
| `MX_ARTIFACT_COMPILE_CONFIG_DIGEST` | empty | Optional compile-configuration compatibility digest for cache discovery |
| `MX_ARTIFACT_READY_URL` | `http://127.0.0.1:8000/health` | Readiness endpoint checked before a source publishes JIT cache artifacts |
| `MX_ARTIFACT_READY_TIMEOUT_SECS` | `1800` | Maximum time to wait for readiness and successful artifact publication |

### UCX/NIXL Tuning

| Variable | Recommended | Description |
|----------|-------------|-------------|
| `UCX_RNDV_SCHEME` | `get_zcopy` | Zero-copy RDMA reads |
| `UCX_RNDV_THRESH` | `0` | Force rendezvous for all transfers |
| `NIXL_LOG_LEVEL` | `INFO` | NIXL logging level |

## Package Structure

| Module | Description |
|--------|-------------|
| `modelexpress.client` | `MxClient` -- gRPC client for the ModelExpress server |
| `modelexpress.metadata` | Metadata clients, source identity, publishing, and worker manifest serving |
| `modelexpress.engines.vllm.loader` | `MxModelLoader` -- vLLM integration |
| `modelexpress.engines.sglang.loader` | `MxModelLoader` -- SGLang `remote_instance` integration |
| `modelexpress.vllm_loader` | Compatibility shim for the vLLM loader |
| `modelexpress.nixl_transfer` | `NixlTransferManager` -- NIXL agent lifecycle and RDMA transfers |
| `modelexpress.types` | `TensorDescriptor`, `WorkerMetadata` -- core data types |
| `modelexpress.vllm_worker` | Compatibility worker extension for older manual-registration workflows |

## How It Works

1. **Source** loads weights from disk, registers raw tensors with NIXL *before* FP8 processing, and publishes metadata to the ModelExpress server.
2. **Target** creates dummy weights, waits for the source ready flag, then pulls raw tensors via RDMA read.
3. Both source and target run `process_weights_after_loading()` independently, producing identical FP8-transformed weights.
4. When artifact transfer is enabled, a healthy source publishes its pod-scoped JIT caches and later pods install compatible caches before model initialization.

This pre-processing transfer strategy is critical for FP8 models (e.g., DeepSeek-V4-Pro) where tensors are renamed and transformed during processing.

## License

Apache-2.0
