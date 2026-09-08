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

### Requirements

- Python >= 3.10
- protobuf >= 5.27.2 and < 7
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

### RL trainer publication

An RL framework creates a weight version through the external Refit API. Each
trainer actor then invokes its rank-local client to stage and publish one shard.
Worker registration, manifest serving, and internal shard CRUD remain hidden
behind the client.

When creating a `WeightVersion`, the orchestrator may supply its UID or let MX
generate one. A caller-supplied UID already assigned to another request returns
`ALREADY_EXISTS`; an identical request retried with the same idempotency key
returns the existing version.

```python
from modelexpress_rl import (
    MegatronTrainerContext,
    ModelExpressTrainerClient,
    ModelExpressTrainerConfig,
    WeightVersionRef,
)

trainer = ModelExpressTrainerClient.initialize(
    ModelExpressTrainerConfig(engine_context=MegatronTrainerContext())
)
trainer.bind_tensors(megatron_tensor_specs)
trainer.publish_version(version=WeightVersionRef(version.uid))
```

The deployment supplies `MODEL_NAME`,
`MX_TRAINER_STAGING_MODE`, `MX_WEIGHT_PAYLOAD_FORMAT`, `MX_WORKER_HOST`, and the
normal ModelExpress server configuration. The Megatron adapter derives its
source slot from logical tensor names and shard geometry. DP replicas of the same
partition therefore publish redundant workers for one slot, while distinct TP
partitions remain separate required slots. The NIXL metadata endpoint is derived
from `MX_WORKER_HOST` and the client-owned NIXL manager's listen port. `LOCAL_RANK`
selects the device unless `device_id` is passed to `initialize()`.

Canonical S3 staging consumes Hugging Face tensor buckets produced by the
training framework. Framework-native bucket settings remain the default.
The public trainer API accepts
`ModelExpressTrainerConfig(object_storage=ObjectStorageConfig(...))`; its
`storage_type` selects the provider. Weight versions use the corresponding
typed `ObjectStorageSource` envelope. Generator clients likewise accept
`ModelExpressGeneratorConfig(object_storage=ObjectStorageGeneratorConfig(...))`.
The current trainer and generator clients support only `ObjectStorageType.S3`.
Integrations may use `MX_REFIT_DELTA_BUCKET_BYTES` as an explicit override, or
its 512 MiB default when they have no native setting. CPU workers are configured
by `MX_REFIT_DELTA_WORKERS` (default `min(32, CPU count)`), while
`MX_S3_UPLOAD_WORKERS` controls concurrent full-checkpoint batch uploads.
`MX_REFIT_CHECKSUM_FORMAT` selects the checksum algorithm and defaults to
`adler32`.
The framework integration reads the bucket-size setting while constructing the
stream; ModelExpress processes each supplied bucket without splitting or merging
it.
Before training begins, the framework calls `prepare_delta_base()` with one
bucket stream. ModelExpress submits each framework bucket directly for
concurrent rank-local seed-checkpoint reads. Real delta staging therefore
performs no seed-checkpoint reads. A `FULL_HF_CHECKPOINT` version serializes
the current buckets as native HF safetensor shards, omits `base_version_id`, and
replaces the retained snapshot so the next `XOR_DELTA` uses it as its exact
base. A bounded worker pool updates the rank-local snapshot with immutable CPU
tensors. Each publishing rank groups its snapshot into concurrently uploaded
safetensors objects with at most `MX_REFIT_FULL_CHECKPOINT_BATCH_BYTES` tensor
bytes (4 GiB by default); an oversized tensor occupies its own object. The
objects are sent directly to S3 without a trainer-side temporary checkpoint.
Framework integrations own the optional full-checkpoint period; it is disabled
by default.
Generators use the ModelExpress S3 client to download full-checkpoint batches
concurrently. Each worker validates one downloaded batch and copies its tensors
into their existing local mmap destinations, without materializing a second full
checkpoint. ModelStreamer integration remains a future optimization.
The local checkpoint state changes from `READY` to `UPDATING` before mutation
and returns to `READY` only after success. An interrupted update must be reseeded
from `seed_checkpoint_path` during initialization.
The framework supplies each version's exact `object_storage.uri` under the
configured `uri_prefix`. That URI names the global safetensors index; its
objects are stored beside it. A delta index records the target
`WeightVersion.uid` and its `base_version_id` as `metadata.version` and
`metadata.base_version`. After upload, the orchestrator changes the version from
`STAGING` to `READY`. S3 versions remain READY for rollout recovery; their
immutable objects are governed by the bucket's external lifecycle policy.

The client owns the NIXL manager and trainer-side manifest service. `server_url`
selects the central ModelExpress control-plane service and defaults to the
normal ModelExpress server configuration. A Megatron worker may initialize the
client before its distributed process group is ready; the explicitly selected
`engine_context` is constructed lazily on the first tensor operation. Deployment
environment variables do not select Python implementations.

Initialization fixes the staging mode. NIXL also fixes its payload format;
canonical S3 publication follows each target `WeightVersion`. On NIXL,
`publish()` hides manifest publication and the internal
`CreateWeightVersionShard` RPC. The current Megatron adapter registers and
exposes its live buffers through
`IN_PLACE`, so callers must keep those tensors immutable while the version is
published. The required lifecycle is synchronous: create and publish the
version, update every generator, retire and release the version, and only then
resume training or begin the next optimizer step.

Version creation and expected-source-slot declaration remain
framework-orchestrator responsibilities. Each trainer adapter derives its own
source slot from the engine's native topology; the orchestrator declares the
expected slots using the same adapter-defined convention. `initialize()`
constructs the adapter selected by `engine_context` internally.
Megatron and FSDP implementations are available. Megatron-specific APIs live under
`modelexpress_rl`;
`modelexpress.refit.reshard` remains the shared, engine-neutral transfer core.

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
| `MODEL_EXPRESS_URL` | `localhost:8001` | Deprecated in favor of `MX_SERVER_ADDRESS`. Still read by all client paths and still takes precedence when both are set, because the TRT-LLM live-transfer integration reads only this name. It is removed once that path reads `MX_SERVER_ADDRESS`; until then set both to the same value. |
| `MX_DISABLE_PATCHES` | `0` | Emergency escape hatch that skips all runtime compatibility patches. Set to `1`, `true`, `yes`, or `on` if a patch is incompatible with the installed engine. |
| `MX_EXPECTED_WORKERS` | Auto-detected from TP size | Number of GPU workers to coordinate |
| `MX_SYNC_PUBLISH` | `0` | Source: wait for all workers before publishing metadata |
| `MX_SYNC_START` | `1` | Target: wait for all source workers before transferring |
| `MX_POOL_REG` | `0` | Allocation-level NIXL registration (registers cudaMalloc blocks instead of individual tensors) |
| `MX_P2P_METADATA` | `1` | Serve tensor and artifact manifests directly from source workers; set to `0` to route full tensor metadata through the central server |
| `MX_REFIT_METADATA_PORT` | `7555` | Base NIXL metadata-listener port for RL generator refit; each rank adds its local device ID. Kept separate from `MX_METADATA_PORT`, which may remain owned by the boot-time loader |
| `MX_ARTIFACT_TRANSFER` | `0` | Transfer compatible vLLM TorchInductor, Triton, DeepGEMM, TileLang, CuTe DSL, and FlashInfer JIT caches, including persistent autotune files when supported by vLLM |
| `MX_ARTIFACT_BUNDLE_ROOT` | `$TMPDIR/modelexpress-artifacts` | Staging root for tarred cache artifact bundles |
| `MX_ARTIFACT_COMPILE_CONFIG_DIGEST` | empty | Optional compile-configuration compatibility digest for cache discovery |
| `MX_ARTIFACT_READY_URL` | Framework default | Readiness endpoint checked before a source publishes weights or JIT cache artifacts (`http://127.0.0.1:8000/health` for vLLM; `http://127.0.0.1:30000/health` for SGLang). On the non-head nodes of a multi-node engine, a loopback host is rewritten onto the head's address (the engine's own distributed-init address, else `LWS_LEADER_ADDRESS`), preserving the configured port and path. A non-loopback host is used verbatim |
| `MX_ARTIFACT_READY_TIMEOUT_SECS` | `1800` | Maximum time to wait for readiness and successful artifact publication |
| `MX_HEARTBEAT_INTERVAL_SECS` | `30` | Seconds between READY status heartbeats for published sources, including reshard rendezvous sources; keep below the server heartbeat timeout |
| `MX_RESHARD_MAX_SEGMENTS_PER_COPY` | `64` | Maximum exact descriptors for one no-gather refit copy before a compatible dim-0-sharded source is pulled once into contiguous staging and sliced locally |
| `MX_RESHARD_FUSED_WIRE` | `1` | Issue a refit's exact-segment, full-pull, and convert reads as one transport batch instead of draining each phase in turn. Set to `0` to restore the phased reads for an A/B comparison |
| `MX_RESHARD_BATCH_INSTALL` | `1` | Re-slice a refit's full-pulled sources with one batched `torch._foreach_copy_` instead of one `copy_()` per captured view. Issues the same copies; a per-view loop costs thousands of kernel launches whose overhead can rival the RDMA. Set to `0` to restore the per-view loop for an A/B comparison |
| `MX_RESHARD_CACHE_DESCRIPTORS` | `1` | Build NIXL read descriptors once per stable transfer plan and reuse them across refits. Set to `0` to rebuild the descriptor lists on every step for an A/B comparison |
| `MX_RESHARD_REQUIRE_FULL_COVERAGE` | `0` | Fail a refit that installs less than `MX_RESHARD_COVERAGE_FLOOR` of the engine's parameter bytes. Off by default because partial and subset refit are intended; set to `1` for benchmark runs, where an incomplete refit produces timings that are the wrong magnitude |
| `MX_RESHARD_COVERAGE_FLOOR` | `0.995` | Fraction of engine parameter bytes a gated refit must install. Not `1.0`: a few engine parameters, such as rotary `inv_freq`, are legitimately not refit material. Values outside `[0.0, 1.0]` are rejected |
| `MX_RESHARD_HANDSHAKE_TIMEOUT_S` | `900` | Budget for the whole P2P metadata handshake, across every trainer peer and every retry. Bounds the handshake independently of the refit timeout, so one unreachable publisher cannot consume the entire refit |
| `MX_RESHARD_HANDSHAKE_ATTEMPT_S` | `20` | Ceiling on a single peer dial. A reachable peer answers in well under a second, so a short attempt frees the budget to try a different peer rather than block on one |
| `MX_RESHARD_HANDSHAKE_BACKOFF_S` | `2` | Pause after a full pass over the pending peers makes no progress, so a transient stall is waited out rather than hammered |
| `MX_REFIT_STAGE_RECORD` | `1` | Emit one `refit-stage-v2` JSON record per refit, giving a benchmark harness the per-stage timings without parsing logs. Set to `0` to silence it |
| `MX_RESHARD_MAX_GBPS` | `0` | Per-rank fabric ceiling in Gbps. A measured wire rate above it means the timing is wrong rather than the transfer being fast, so the refit is rejected. `0` disables the check, since only the operator knows the real per-rank limit |
| `MX_RESHARD_MIN_GBPS` | `0` | Per-rank floor in Gbps. Below it, the refit emits a `refit-slow-throughput-v1` JSON warning naming the rate, the bound and the shortfall. Applies to both receivers — the Megatron slice-reshard receiver and the staged receiver the FSDP trainers pull over, including its peer-to-peer pull — so one setting covers a job whichever path it refits on. Warns rather than aborting, unlike the ceiling: an impossible rate means the payload never moved, but a slow one is still correct, so enforcement belongs in a CI gate reading the record rather than in a running job. `0` disables it. Worth setting for any throughput run — a 20x collapse has been observed with byte counts exact, descriptor counts exact, coverage 100%, fallback 0 and no error anywhere, and without a lower bound there is nothing in the telemetry that dissents. When choosing a value, note that it is compared per rank against the rate one receiver sees *while its siblings are also receiving*, which is well below the rate a single receiver reaches alone; sizing it against the solo number puts the floor above the healthy concurrent rate and it will fire on good runs. Aim between the two — roughly the geometric mean of the collapsed rate and the healthy concurrent rate leaves both verdicts off a tight margin |
| `MX_RESHARD_PUBLISH_DIGEST` | `0` | Have each trainer publish a position-sensitive digest of every shard it advertises, so a receiver can later confirm it installed the bytes the publisher held. Off by default: the reduction costs a pass over every published tensor, which is large next to a ~1.5 s wire, so turn it on when qualifying a build rather than when measuring throughput |

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
| [`modelexpress.refit`](modelexpress/refit/README.md) | Experimental RL weight-refit timing, receiver-driven resharding, and engine adapter contracts |
| `modelexpress.engines.vllm.loader` | `MxModelLoader` -- vLLM integration |
| `modelexpress.refit.reshard` | Engine-agnostic loader-geometry capture and bounded no-gather transfer planning |
| `modelexpress.engines.sglang.loader` | `MxModelLoader` -- SGLang `remote_instance` integration |
| `modelexpress.engines.trtllm.loader` | `MxModelLoader` -- TensorRT-LLM shared-strategy integration |
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
