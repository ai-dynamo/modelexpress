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

# With test dependencies
pip install -e ".[dev]"

# Additionally install the pinned protobuf code generator when changing protobuf APIs
pip install -e ".[codegen]"
```

NIXL is expected to be supplied by the runtime environment (TRT-LLM,
SGLang, Dynamo, and NemoRL runtime images all ship `nixl-cu12` or
`nixl-cu13`). For a bare-environment install, run `pip install nixl-cu12`
or `pip install nixl-cu13` separately, matching your host CUDA toolkit.

The collective refit path in `modelexpress_rl.collective` needs NCCL M2N,
which ships separately from `nccl4py` as the `nccl-extensions` distribution
under the same `nccl` namespace. Run `pip install "nccl-extensions[cu12]"`
or `pip install "nccl-extensions[cu13]"`, matching your host CUDA toolkit.
The CUDA major has to be named: a bare `pip install nccl-extensions`
resolves and then fails to import, because `cuda-bindings` only arrives
through one of those two extras. Nothing else in the client imports it, so
an installation that never uses collective refit does not need it.

Reshard needs NCCL 2.30.7 or newer, and an image that ships its own older
libnccl will win the load over the one pip installed. That failure surfaces
inside the reshard call rather than at import, so check which library the
process resolves rather than which one is on disk. The NGC PyTorch 26.06
image ships 2.30.5 and needs the newer library preloaded.

### Requirements

- Python >= 3.10
- protobuf >= 5.27.2 and < 7
- NVIDIA GPUs with RDMA/InfiniBand support
- [NIXL](https://github.com/ai-dynamo/nixl) (NVIDIA Interconnect eXchange Library)
- [NCCL M2N](https://pypi.org/project/nccl-extensions/) (`nccl-extensions`, collective refit only)
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

## Quick Start with TensorRT-LLM

TensorRT-LLM integrates through its native `checkpoint_format="MX"` interface.
Install ModelExpress in a qualified TensorRT-LLM image, then construct the
PyTorch backend with the ModelExpress server configuration:

```python
from tensorrt_llm.llmapi import LLM

llm = LLM(
    model="/model",
    checkpoint_format="MX",
    mx_config={
        "server_url": "modelexpress-server:8001",
    },
    tensor_parallel_size=4,
    backend="pytorch",
)
```

The first replica falls back to the Hugging Face checkpoint and publishes its
post-transform weights; later compatible replicas receive them through
ModelExpress. The current qualified scope is the `LlamaForCausalLM` family.
See the
[TensorRT-LLM P2P example](../../examples/p2p_transfer_k8s/client/trtllm/)
for the qualified-image requirement and production-style Kubernetes
deployment.

## Programmatic Usage

### RL trainer and generator APIs

See the [RL package quickstart and support guide](../../docs/rl/README.md) for trainer publication, generator initialization, and import examples.

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

Shared client settings are registered in `modelexpress.envs`. RL settings are listed in the [RL configuration reference](../../docs/rl/CONFIGURATION.md).

| Variable | Default | Description |
|----------|---------|-------------|
| `MX_SERVER_ADDRESS` | `localhost:8001` | ModelExpress gRPC server address (recommended) |
| `MODEL_EXPRESS_URL` | `localhost:8001` | Deprecated in favor of `MX_SERVER_ADDRESS`. Still read by all client paths and still takes precedence when both are set, because the TRT-LLM live-transfer integration reads only this name. It is removed once that path reads `MX_SERVER_ADDRESS`; until then set both to the same value. |
| `MX_DISABLE_PATCHES` | `0` | Emergency escape hatch that skips all runtime compatibility patches. Set to `1`, `true`, `yes`, or `on` if a patch is incompatible with the installed engine. |
| `MX_EXPECTED_WORKERS` | Auto-detected from TP size | Number of GPU workers to coordinate |
| `MX_SYNC_PUBLISH` | `0` | Source: wait for all workers before publishing metadata |
| `MX_SYNC_START` | `1` | Target: wait for all source workers before transferring |
| `MX_POOL_REG` | `0` | Allocation-level NIXL registration (registers cudaMalloc blocks instead of individual tensors) |
| `MX_P2P_METADATA` | `1` | Serve tensor and artifact manifests directly from source workers; set to `0` to route full tensor metadata through the central server |
| `MX_ARTIFACT_TRANSFER` | `0` | Transfer compatible vLLM TorchInductor, Triton, DeepGEMM, TileLang, CuTe DSL, and FlashInfer JIT caches, including persistent autotune files when supported by vLLM |
| `MX_ARTIFACT_BUNDLE_ROOT` | `$TMPDIR/modelexpress-artifacts` | Staging root for tarred cache artifact bundles |
| `MX_ARTIFACT_COMPILE_CONFIG_DIGEST` | empty | Optional compile-configuration compatibility digest for cache discovery |
| `MX_ARTIFACT_READY_URL` | Framework default | Readiness endpoint checked before a source publishes weights or JIT cache artifacts (`http://127.0.0.1:8000/health` for vLLM; `http://127.0.0.1:30000/health` for SGLang). On the non-head nodes of a multi-node engine, a loopback host is rewritten onto the head's address (the engine's own distributed-init address, else `LWS_LEADER_ADDRESS`), preserving the configured port and path. A non-loopback host is used verbatim |
| `MX_ARTIFACT_READY_TIMEOUT_SECS` | `1800` | Maximum time to wait for readiness and successful artifact publication |
| `MX_HEARTBEAT_INTERVAL_SECS` | `30` | Seconds between READY status heartbeats for published sources, including reshard rendezvous sources; keep below the server heartbeat timeout |
### Canonical S3 Transfer Tuning

Objects below the configured thresholds use one PUT or GET. Larger uploads use
multipart parts, and larger downloads use ranged GETs through one persistent
`s3transfer.TransferManager` per `S3Client`. The receiver's file-level pool and
the manager's global request concurrency use the same worker setting, so all
whole-object and ranged data GETs share one 16-request budget. HEAD requests use
the manager's separate submission executor. Downloads target a seekable
`BytesIO`, so the complete downloaded object remains resident; the I/O settings
below bound queued chunks, not the final object size.

| Variable | Default | Description |
|----------|---------|-------------|
| `MX_S3_MULTIPART_THRESHOLD_BYTES` | `104857600` (100 MiB) | Minimum object size for multipart upload |
| `MX_S3_UPLOAD_PART_BYTES` | `16777216` (16 MiB) | Multipart upload part size |
| `MX_S3_UPLOAD_WORKERS` | `8` | Maximum concurrent multipart part uploads |
| `MX_S3_DOWNLOAD_RANGE_THRESHOLD_BYTES` | `104857600` (100 MiB) | Minimum object size for parallel ranged download |
| `MX_S3_DOWNLOAD_RANGE_BYTES` | `8388608` (8 MiB) | Byte-range size for parallel downloads |
| `MX_S3_DOWNLOAD_WORKERS` | `16` | Receiver file-worker limit and shared whole/ranged data GET concurrency budget |
| `MX_S3_DOWNLOAD_IO_CHUNK_BYTES` | `1048576` (1 MiB) | TransferManager I/O queue chunk size |
| `MX_S3_DOWNLOAD_MAX_IN_MEMORY_CHUNKS` | `16` | Sets `max_io_queue_size` and the non-seekable-output chunk limit. For the seekable `BytesIO` target, the default bounds queued I/O to about 16 MiB but does not cap the full downloaded object |
| `MX_S3_MAX_POOL_CONNECTIONS` | `32` | Botocore HTTP connection-pool size |
| `MX_S3_MAX_ATTEMPTS` | `5` | Botocore total request attempts and TransferManager post-200 streaming-download attempts |
| `MX_S3_TCP_KEEPALIVE` | `true` | Enable TCP keepalive for S3 connections |

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
| `modelexpress.engines.trtllm.loader` | `MxModelLoader` -- TensorRT-LLM shared-strategy integration |
| `modelexpress.vllm_loader` | Compatibility shim for the vLLM loader |
| `modelexpress.nixl_transfer` | `NixlTransferManager` -- NIXL agent lifecycle and RDMA transfers |
| `modelexpress.types` | `TensorDescriptor`, `WorkerMetadata` -- core data types |
| `modelexpress.vllm_worker` | Compatibility worker extension for older manual-registration workflows |

The optional RL APIs are installed with this distribution under the `modelexpress_rl` namespace. See the [RL package guide](../../docs/rl/README.md).

## How It Works

1. **Source** loads weights from disk, registers raw tensors with NIXL *before* FP8 processing, and publishes metadata to the ModelExpress server.
2. **Target** creates dummy weights, waits for the source ready flag, then pulls raw tensors via RDMA read.
3. Both source and target run `process_weights_after_loading()` independently, producing identical FP8-transformed weights.
4. When artifact transfer is enabled, a healthy source publishes its pod-scoped JIT caches and later pods install compatible caches before model initialization.

This pre-processing transfer strategy is critical for FP8 models (e.g., DeepSeek-V4-Pro) where tensors are renamed and transformed during processing.

## License

Apache-2.0
