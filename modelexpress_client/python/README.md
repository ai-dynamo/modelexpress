# ModelExpress Python client

ModelExpress helps inference workers load model weights and RL rollout workers install new trainer versions. This package supplies the runtime integrations, Python clients, and weight-transfer code; your inference or training runtime owns GPU execution.

- **Inference:** start with the [vLLM Kubernetes quickstart](../../docs/integrations/runtimes/vllm.md), then use [SGLang](../../docs/integrations/runtimes/sglang.md), [TensorRT-LLM](../../docs/integrations/runtimes/tensorrt-llm.md), or [Dynamo](../../docs/integrations/orchestrators/dynamo.md) if that is your serving stack.
- **RL:** start with [RL weight updates](../../docs/guides/rl.md) for trainer-to-rollout refit and version handling.
- **Direct storage loading:** use the [ModelStreamer examples](../../examples/model_streamer_k8s/README.md) to load safetensors without an MX server.

## Installation

Install into your runtime environment using Python 3.10 or newer. From the ModelExpress repository root:

```bash
pip install ./modelexpress_client/python

# For client development and tests
pip install -e './modelexpress_client/python[dev]'

# Only when regenerating protobuf APIs
pip install -e './modelexpress_client/python[codegen]'
```

The package does not install vLLM, SGLang, TensorRT-LLM, or NIXL. Use a [supported runtime image](../../docs/COMPATIBILITY.md) with a compatible CUDA/NIXL stack. Prefer the NIXL package supplied by that image; for a bare environment, install `nixl-cu12` or `nixl-cu13` to match the runtime's CUDA stack.

P2P requires compatible GPU workers, a working NIXL transport, and source discovery through a ModelExpress server or the `k8s-service` backend. Direct ModelStreamer storage loading does not require NIXL, RDMA, or a server. Ordinary Python control-plane API calls do not require a GPU.

## How inference loading works

1. A source worker loads a checkpoint through an eligible storage loader and publishes its availability.
2. A compatible target discovers the source and copies its ready GPU tensors through NIXL. For vLLM, these are **post-processed** tensors, including the layouts produced by quantization processing; the target reconstructs the corresponding layout before receiving them.
3. The runtime finishes startup and serves requests. Verify the target's transfer completion log and an inference response; automatic fallback can make a healthy worker look like a successful P2P run.

The central server carries P2P discovery metadata, while weight bytes move between workers. Sources must stay available during transfer. Targets still need configuration and tokenizer files. Optional `MX_ARTIFACT_TRANSFER=1` also reuses compatible vLLM JIT caches from a healthy source. See [Choose a path](../../docs/guides/choose-a-path.md) for storage and offline deployment choices.

## Programmatic usage

### MxClient

`MxClient` is a lightweight gRPC client for communicating with the ModelExpress server. This example lists ready sources; runtime integrations perform compatibility checks and transfers:

```python
from modelexpress import MxClient, p2p_pb2

client = MxClient(server_url="modelexpress-server:8001")
try:
    response = client.list_sources(status_filter=p2p_pb2.SOURCE_STATUS_READY)
    for source in response.instances:
        print(source.model_name, source.worker_rank, source.worker_id)
finally:
    client.close()
```

### Registering Loaders Manually

Manual registration is only needed for integrations that construct vLLM loaders outside vLLM 0.23.0's native load-format path.

```python
from modelexpress import register_modelexpress_loaders

register_modelexpress_loaders()
# Now vLLM recognizes --load-format modelexpress and mx
```

### RL trainer publication

For an end-to-end training and rollout workflow, start with [RL weight updates](../../docs/guides/rl.md). The trainer API below is an integration reference: the framework must create `version`, supply `megatron_tensor_specs`, coordinate all publishing ranks, and control when rollout workers resume.

<details>
<summary>Trainer API and lifecycle reference</summary>

An RL framework creates a weight version through the external Refit API. Each trainer actor then invokes its rank-local client to stage and publish one shard. Worker registration, manifest serving, and internal shard CRUD remain hidden behind the client.

When creating a `WeightVersion`, the orchestrator may supply its UID or let MX generate one. A caller-supplied UID already assigned to another request returns `ALREADY_EXISTS`; an identical request retried with the same idempotency key returns the existing version.

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

The deployment supplies `MODEL_NAME`, `MX_TRAINER_STAGING_MODE`, `MX_WEIGHT_PAYLOAD_FORMAT`, `MX_WORKER_HOST`, and the normal ModelExpress server configuration. The Megatron adapter derives its source slot from logical tensor names and shard geometry. DP replicas of the same partition therefore publish redundant workers for one slot, while distinct TP partitions remain separate required slots. The NIXL metadata endpoint is derived from `MX_WORKER_HOST` and the client-owned NIXL manager's listen port. `LOCAL_RANK` selects the device unless `device_id` is passed to `initialize()`.

Canonical S3 staging consumes Hugging Face tensor buckets produced by the training framework. Framework-native bucket settings remain the default. The public trainer API accepts `ModelExpressTrainerConfig(object_storage=ObjectStorageConfig(...))`; its `storage_type` selects the provider. Weight versions use the corresponding typed `ObjectStorageSource` envelope. Generator clients likewise accept `ModelExpressGeneratorConfig(object_storage=ObjectStorageGeneratorConfig(...))`. The current trainer and generator clients support only `ObjectStorageType.S3`. Integrations may use `MX_REFIT_DELTA_BUCKET_BYTES` as an explicit override, or its 512 MiB default when they have no native setting. CPU workers are configured by `MX_REFIT_DELTA_WORKERS` (default `min(32, CPU count)`), while `MX_S3_UPLOAD_WORKERS` controls concurrent full-checkpoint batch uploads. `MX_REFIT_CHECKSUM_FORMAT` selects the checksum algorithm and defaults to `adler32`. The framework integration reads the bucket-size setting while constructing the stream; ModelExpress processes each supplied bucket without splitting or merging it.

Before training begins, the framework calls `prepare_delta_base()` with one bucket stream. ModelExpress submits each framework bucket directly for concurrent rank-local seed-checkpoint reads. Real delta staging therefore performs no seed-checkpoint reads. A `FULL_HF_CHECKPOINT` version serializes the current buckets as native HF safetensor shards, omits `base_version_id`, and replaces the retained snapshot so the next `XOR_DELTA` uses it as its exact base. A bounded worker pool updates the rank-local snapshot with immutable CPU tensors. Each publishing rank groups its snapshot into concurrently uploaded safetensors objects with at most `MX_REFIT_FULL_CHECKPOINT_BATCH_BYTES` tensor bytes (4 GiB by default); an oversized tensor occupies its own object. The objects are sent directly to S3 without a trainer-side temporary checkpoint. Framework integrations own the optional full-checkpoint period; it is disabled by default.

Generators use the ModelExpress S3 client to download full-checkpoint batches concurrently. Each worker validates one downloaded batch and copies its tensors into their existing local mmap destinations, without materializing a second full checkpoint. ModelStreamer integration remains a future optimization. The local checkpoint state changes from `READY` to `UPDATING` before mutation and returns to `READY` only after success. An interrupted update must be reseeded from `seed_checkpoint_path` during initialization.

The framework supplies each version's exact `object_storage.uri` under the configured `uri_prefix`. That URI names the global safetensors index; its objects are stored beside it. A delta index records the target `WeightVersion.uid` and its `base_version_id` as `metadata.version` and `metadata.base_version`. After upload, the orchestrator changes the version from `STAGING` to `READY`. S3 versions remain READY for rollout recovery; their immutable objects are governed by the bucket's external lifecycle policy.

The client owns the NIXL manager and trainer-side manifest service. `server_url` selects the central ModelExpress control-plane service and defaults to the normal ModelExpress server configuration. A Megatron worker may initialize the client before its distributed process group is ready; the explicitly selected `engine_context` is constructed lazily on the first tensor operation. Deployment environment variables do not select Python implementations.

Initialization fixes the staging mode. NIXL also fixes its payload format; canonical S3 publication follows each target `WeightVersion`. On NIXL, `publish()` hides manifest publication and the internal `CreateWeightVersionShard` RPC. The current Megatron adapter registers and exposes its live buffers through `IN_PLACE`, so callers must keep those tensors immutable while the version is published. The required lifecycle is synchronous: create and publish the version, update every generator, retire and release the version, and only then resume training or begin the next optimizer step.

Version creation and expected-source-slot declaration remain framework-orchestrator responsibilities. Each trainer adapter derives its own source slot from the engine's native topology; the orchestrator declares the expected slots using the same adapter-defined convention. `initialize()` constructs the adapter selected by `engine_context` internally. Megatron and FSDP implementations are available. Megatron-specific APIs live under `modelexpress_rl`; `modelexpress.refit.reshard` remains the shared, engine-neutral transfer core.

</details>

## Environment variables

For loading policy, defaults, and deployment settings, use the [configuration reference](../../docs/CONFIGURATION.md). `MX_LOAD_STRATEGY_CHAIN=INFERENCE` is the default; `RL` selects the separate RL startup policy and does not by itself perform an update on a running worker.

<details>
<summary>Advanced client, refit, and transport settings</summary>

| Variable | Default | Description |
|----------|---------|-------------|
| `MX_SERVER_ADDRESS` | `localhost:8001` | ModelExpress gRPC server address (recommended) |
| `MODEL_EXPRESS_URL` | `localhost:8001` | Legacy server address; takes precedence over `MX_SERVER_ADDRESS` when both are set. Prefer `MX_SERVER_ADDRESS`; if an older integration requires both, set them to the same endpoint. |
| `MX_DISABLE_PATCHES` | `0` | Emergency escape hatch that skips all runtime compatibility patches. Set to `1`, `true`, `yes`, or `on` if a patch is incompatible with the installed engine. |
| `MX_POOL_REG` | `0` | Allocation-level NIXL registration (registers cudaMalloc blocks instead of individual tensors) |
| `MX_P2P_METADATA` | `1` | Serve tensor and artifact manifests directly from source workers; set to `0` to route full tensor metadata through the central server |
| `MX_LOAD_STRATEGY_CHAIN` | `INFERENCE` | Select the engine-neutral initial-load policy. With `RL`, a configured desired UID permits only desired-version P2P and S3 replay; without one, loading falls back through `MX_MODEL_URI` and then the engine default. vLLM speculative draft models are rejected in `RL` mode. |
| `MX_REFIT_DESIRED_VERSION_UID` | (unset) | Exact version required when `MX_LOAD_STRATEGY_CHAIN=RL`; startup fails rather than using a version-agnostic fallback when neither P2P nor S3 can load it |
| `MX_GENERATOR_SOURCE_ORDER` | Auto-detected | Ordered RL weight sources. Desired-version cold start defaults to `GENERATOR,OBJECT_STORAGE`; `OBJECT_STORAGE` disables P2P for both cold start and active refit. `TRAINER` applies only to active refit and is rejected for desired-version cold start. |
| `MX_REFIT_CHECKPOINT_DIR` | (unset) | Host-local cache for RL S3 full checkpoints, deltas, and materialized checkpoints; the S3 strategy skips when it is unset |
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
| `MX_RESHARD_MIN_GBPS` | `0` | Per-rank throughput floor; emits a `refit-slow-throughput-v1` warning below the threshold. `0` disables it. Choose a floor based on concurrent multi-rank performance, not a single-rank peak; CI or benchmark tooling decides whether warnings fail a run. |
| `MX_RESHARD_PUBLISH_DIGEST` | `0` | Have each trainer publish a position-sensitive digest of every shard it advertises, so a receiver can later confirm it installed the bytes the publisher held. Off by default: the reduction costs a pass over every published tensor, which is large next to a ~1.5 s wire, so turn it on when qualifying a build rather than when measuring throughput |

### Canonical S3 Transfer Tuning

Objects below the configured thresholds use one PUT or GET. Larger uploads use multipart parts, and larger downloads use ranged GETs through one persistent `s3transfer.TransferManager` per `S3Client`. The receiver's file-level pool and the manager's global request concurrency use the same worker setting, so all whole-object and ranged data GETs share one 16-request budget. HEAD requests use the manager's separate submission executor. Downloads target a seekable `BytesIO`, so the complete downloaded object remains resident; the I/O settings below bound queued chunks, not the final object size.

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

</details>

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

## License

Apache-2.0
