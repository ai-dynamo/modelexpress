<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# RL configuration reference

This page covers settings registered by the optional RL package. Shared ModelExpress client settings remain in [base configuration](../CONFIGURATION.md); S3 transfer settings shared with normal client operations remain in the [Python client README](../../modelexpress_client/python/README.md#canonical-s3-transfer-tuning). RL variables are read by `modelexpress_rl` APIs and registered in the source package `modelexpress_rl/envs.py`.

## RL environment settings

RL-specific settings are registered in `modelexpress_rl/envs.py` and imported through the installed `modelexpress_rl` namespace. See [base configuration](../CONFIGURATION.md) for shared settings.

| Variable | Default | Description |
|----------|---------|-------------|
| `MX_LOAD_STRATEGY_CHAIN` | `INFERENCE` | Select the engine-neutral initial-load policy. With `RL`, a configured desired UID permits only desired-version P2P and S3 replay; without one, loading falls back through `MX_MODEL_URI` and then the engine default. vLLM speculative draft models are rejected in `RL` mode. |
| `MX_REFIT_DESIRED_VERSION_UID` | (unset) | Exact version required when `MX_LOAD_STRATEGY_CHAIN=RL`; startup fails rather than using a version-agnostic fallback when neither P2P nor S3 can load it |
| `MX_GENERATOR_SOURCE_ORDER` | Auto-detected | Ordered RL weight sources. Desired-version cold start defaults to `GENERATOR,OBJECT_STORAGE`; `OBJECT_STORAGE` disables P2P for both cold start and active refit. `TRAINER` applies only to active refit and is rejected for desired-version cold start. |
| `MX_MAX_REPLAY_CHAIN_LENGTH` | `64` | Positive cold-start object-storage replay limit, including the full checkpoint root; align with the running engine's `max_replay_chain_length` when accepting longer chains |
| `MX_REFIT_CHECKPOINT_DIR` | (unset) | Host-local cache for RL S3 full checkpoints, deltas, and materialized checkpoints; the S3 strategy skips when it is unset |
| `MX_REFIT_CHECKPOINT_MAX_SIZE_GB` | `2000` | Positive decimal GB cache quota used by default in vLLM weight-transfer initialization. An explicit `refit_checkpoint_max_size_gb`, including `null`, overrides this setting. |
| `MX_REFIT_METADATA_PORT` | `7555` | Base NIXL metadata-listener port for RL generator refit; each rank adds its local device ID. Kept separate from `MX_METADATA_PORT`, which may remain owned by the boot-time loader |
| `MX_RESHARD_MAX_SEGMENTS_PER_COPY` | `64` | Maximum exact descriptors for one no-gather refit copy before a compatible dim-0-sharded source is pulled once into contiguous staging and sliced locally |
| `MX_REFIT_STAGE_RECORD` | `1` | Emit one `refit-stage-v2` JSON record per refit, giving a benchmark harness the per-stage timings without parsing logs. Set to `0` to silence it |
| `MX_REFIT_PACK_MODULES` | `1` | Legacy setting with no effect on ordered source replay. Trainer transfers batch consecutive captured writes up to the staging limit |
| `MX_RESHARD_MIN_GBPS` | `0` | Per-rank floor in Gbps. Below it, the refit emits a `refit-slow-throughput-v1` JSON warning naming the rate, the bound and the shortfall. Applies to staged trainer transfers and generator-peer pulls. Warns rather than aborting a correct transfer, so enforcement belongs in a CI gate reading the record rather than in a running job. `0` disables it. Worth setting for any throughput run — a 20x collapse has been observed with byte counts exact, descriptor counts exact, coverage 100%, fallback 0 and no error anywhere, and without a lower bound there is nothing in the telemetry that dissents. When choosing a value, note that it is compared per rank against the rate one receiver sees *while its siblings are also receiving*, which is well below the rate a single receiver reaches alone; sizing it against the solo number puts the floor above the healthy concurrent rate and it will fire on good runs. Aim between the two — roughly the geometric mean of the collapsed rate and the healthy concurrent rate leaves both verdicts off a tight margin |
| `MX_RESHARD_PUBLISH_DIGEST` | `0` | Have each trainer publish a position-sensitive digest of every shard it advertises, so a receiver can later confirm it installed the bytes the publisher held. Off by default: the reduction costs a pass over every published tensor, which is large next to a ~1.5 s wire, so turn it on when qualifying a build rather than when measuring throughput |
| `MX_MODEL_NAME_OVERRIDE` | (unset) | Override vLLM's MX model identity without changing its model-loading path. Set before worker startup and use the same name for RL trainers and WeightVersions. Unset or empty preserves vLLM's configured model path/ID, including rewritten S3 cache paths. |
| `MX_REFIT_DELTA_BUCKET_BYTES` | `536870912` (512 MiB) | Optional framework integration override for canonical delta tensor-bucket size; native framework bucket settings take precedence. |
| `MX_REFIT_DELTA_WORKERS` | `min(32, CPU count)` | CPU worker pool for XOR delta computation and application. |
| `MX_S3_DOWNLOAD_BACKEND` | `transfer_manager` | Canonical S3 artifact downloader: `transfer_manager` or opt-in `preallocated`. See [transfer tuning](../../modelexpress_client/python/README.md#canonical-s3-transfer-tuning) for compatibility and memory requirements. |
| `MX_REFIT_DOWNLOAD_WORKERS` | `MX_S3_DOWNLOAD_WORKERS`, then `16` | Concurrent shard downloads for S3 and Azure; also sets the per-client S3 data GET concurrency for either backend. |
| `MX_S3_DOWNLOAD_WORKERS` | `16` | Legacy fallback when `MX_REFIT_DOWNLOAD_WORKERS` is unset. |
| `MX_REFIT_CHECKSUM_FORMAT` | `adler32` | Checksum format written by canonical S3 trainers. |
| `MX_REFIT_FULL_CHECKPOINT_BATCH_BYTES` | `4294967296` (4 GiB) | Maximum tensor bytes grouped into one full-checkpoint safetensors object; oversized tensors occupy a separate object. |
| `MX_REFIT_DELTA_SURGICAL` | `false` | Experimental vLLM delta install: reload only the destination modules touched by the delta chain instead of the whole prepared checkpoint. Falls back to a full reload when lineage or the live version is unknown, or when a fed module is incomplete. See the [S3 delta refit guide](S3_DELTA_WEIGHT_REFIT.md). |
| `MX_REFIT_DEBUG_VALIDATE_PLAN` | `0` | Recheck cached whole-model and owning-module transfer coverage. |
| `MX_REFIT_DEBUG_VALIDATE_GENERATOR_LAYOUT` | `0` | Validate that the generator layout has not changed. |
| `MX_REFIT_PREFETCH_REPLICA_METADATA` | `0` | Fetch stable metadata for all replicas in needed logical-shard groups on first preparation. |
| `MX_REFIT_DEBUG_VALIDATE_REPLICAS` | `0` | Check that replicas in each logical-shard group have equivalent canonical coverage. |
| `MX_REFIT_FULL_STREAMING` | `false` | Experimental vLLM full-HF refit: use ModelStreamer once and tee tensors to a bounded background disk writer. Requires distributed ModelStreamer for TP greater than one. Requires `runai-model-streamer>=0.17.0`. Checksummed artifacts and delta replay retain canonical preparation. For S3-compatible storage, set matching `AWS_ENDPOINT_URL` and `AWS_DEFAULT_REGION` on all generator workers; `object_storage_endpoint_url` / `object_storage_region_name` configure only the index reader, not streamed shard reads. See the architecture section on streamed full refit. |
| `MX_REFIT_STREAM_WINDOW_LAYERS` | `2` | Decoder layers per ranged ModelStreamer request during streamed full refit. Bounds GPU memory held by partially loaded layers; `0` streams the whole checkpoint in one request. |

The selected download worker count must be a positive integer. An invalid
`MX_REFIT_DOWNLOAD_WORKERS` value fails validation without falling back.
`MX_S3_MAX_POOL_CONNECTIONS` separately controls the Botocore HTTP connection pool.
Azure's per-blob concurrency and connection pool remain controlled by
`MX_AZURE_DOWNLOAD_CONCURRENCY` and `MX_AZURE_MAX_POOL_CONNECTIONS`.

## Collective Refit (NCCL M2N) Environment Variables

Client-side policy for the NCCL M2N collective refit path
(`modelexpress_rl.collective`), which is a sibling of the NIXL pull path
rather than a mode of it. Every value is validated when it is read: a
non-numeric, zero or negative setting raises rather than falling back to the
default, so a typo fails loudly at startup instead of silently restoring
stock behavior. Design: [NCCL_M2N_REFIT.md](../NCCL_M2N_REFIT.md).

| Variable | Default | Description |
|----------|---------|-------------|
| `MX_NCCL_REFIT_GROUP_TIMEOUT_S` | `600.0` | Deadline for group formation, from join until the group reports `READY` with every participant admitted. |
| `MX_NCCL_REFIT_POLL_INTERVAL_S` | `0.25` | Backoff floor for `GetCollectiveGroup` polling while waiting for formation and for lane bootstrap ids to be published. |
| `MX_NCCL_REFIT_COMM_INIT_TIMEOUT_S` | `300.0` | Deadline for bringing up one lane's NCCL communicator once its bootstrap id is published. Communicators are created non-blocking, and this bounds the poll to `ncclSuccess`. Setting `NCCL_COMM_BLOCKING` to anything other than `0` is rejected at startup, because blocking initialization would defeat this deadline. |
| `MX_NCCL_REFIT_TRANSFER_TIMEOUT_S` | `600.0` | Deadline for the reshard itself, armed per weight version and checked between submissions, not inside them. A native `nccl.m2n.reshard` call, or the `nccl.m2n.group()` exit that issues a grouped submission (its group end), that blocks the host cannot be interrupted by it: the deadline cannot fire while it blocks, and the client only logs that the call is still blocking. The launcher must therefore bound each trainer and generator process with its own process-level timeout (the reference `examples/rl/m2n_collective_refit/run_e2e.sh` does so only when `PROC_TIMEOUT_S` is set; see [NCCL_M2N_REFIT.md](../NCCL_M2N_REFIT.md) section 8.2). `READY` only means the group formed, so this bounds what happens after it; on expiry the client aborts its lanes, because peers that disagree about which collectives completed cannot be recovered on the same communicator. The abort is local: the group epoch moves when that failure is reported against the operation `start_weight_update` bound (every round call reports on failure, best effort), or when membership changes. Until then the group stays at its epoch and a re-join cannot bootstrap fresh communicators. |
| `MX_NCCL_REFIT_ABORT_TIMEOUT_S` | `10.0` | Shared budget for aborting every lane of a group, data and control communicators included. An overrun raises `CommunicatorWedgedError`, and a wedged abort can only be recovered by exiting the process (see [NCCL_M2N_REFIT.md](../NCCL_M2N_REFIT.md) section 8.2). |
| `MX_NCCL_REFIT_NUM_STREAMS` | `2` | Size of the CUDA stream pool an integration builds when it does not pass its own `streams`. `RefitClientTrainer` and `RefitClientGenerator` do not read it: they use the `streams` the caller passes, or the current stream. |
| `MX_NCCL_REFIT_BLOCKING_DATA_PLANE` | `1` | On by default; set `0` to run data traffic on the non-blocking communicator instead. When on, each lane is still initialized as a non-blocking communicator, bounded by `MX_NCCL_REFIT_COMM_INIT_TIMEOUT_S`, and the reshard and broadcast traffic then runs on a blocking communicator split off it, also under that deadline. On the M2N NVLink path a non-blocking communicator costs roughly 0.4 ms of host time per reshard call, so this cuts a single-node refit of a few hundred parameters several-fold; over the network the per-call cost is already small and this changes little. Abort tears down both communicators. With it off, a rank whose peer dies in the middle of a refit can block past `MX_NCCL_REFIT_TRANSFER_TIMEOUT_S` instead of raising. Read once per client and part of the plan digest, so a cohort that disagrees on it stays `FORMING`. |
| `MX_NCCL_REFIT_GROUPED_SUBMISSION` | `0` | Opt in to recording each layer group as one `nccl.m2n.group()` instead of one `nccl.m2n.reshard` call per entry. Requires GIN: nccl-extensions 0.1.0 fails a group of two or more reshards at group end on every rank without it. Read once per client and part of the plan digest, so a cohort that disagrees on it stays `FORMING`. |
| `MX_NCCL_REFIT_PIPELINE_DEPTH` | `2` | How many layer groups `RefitClientGenerator.update_weights` keeps in flight before installing the oldest. At 2, group N is installed while group N+1 is on the wire; every group is installed before `finish_weight_update` returns. `1` installs each group before `update_weights` returns. Each extra level keeps one more group's outstanding receive buffers alive (at most `MX_NCCL_REFIT_RECEIVE_WINDOW_BYTES` of `pre` scratch per group on the default per-call path). |
| `MX_NCCL_REFIT_RECEIVE_WINDOW_BYTES` | `1073741824` | Generator-side bound on the `pre` scratch receive buffers one layer group keeps alive on the per-call path. Calls are closed into windows of half this size; once two are closed, the older is waited on (its lane events) and dropped, so a group holding the whole bulk does not keep all of its scratch at once. Peak scratch per group is about this budget plus two calls. `0` keeps every buffer until the group is waited on. Ignored with `MX_NCCL_REFIT_GROUPED_SUBMISSION=1`, where nothing is issued before the group closes. |
| `MX_NCCL_REFIT_SEND_WINDOW_BYTES` | `1073741824` | Trainer-side counterpart of `MX_NCCL_REFIT_RECEIVE_WINDOW_BYTES`: bounds the `pre` scratch one layer group keeps alive while it is sent, for example grouped experts stacked into a fresh buffer per call. Without it every such buffer stays allocated until the finish-time drain, so an integration that empties the allocator cache between rounds allocates and frees the whole set each round. `0` keeps every buffer until the drain. Ignored with `MX_NCCL_REFIT_GROUPED_SUBMISSION=1`. |
| `MX_NCCL_REFIT_REGISTRATION_TTL_S` | `3 x MX_HEARTBEAT_INTERVAL_SECS`, so `90` | How long a participant's registration stays alive without a heartbeat. Derived from `MX_HEARTBEAT_INTERVAL_SECS` (default `30`), so raising the heartbeat interval raises this with it. |

## Generator refit connection settings

vLLM's `init_info` and SGLang's `--modelexpress-config` select the checkpoint
reader with `object_storage_type`: `S3` or `AZURE`. SGLang defaults to `S3` when
the field is omitted. For S3, `object_storage_endpoint_url` and
`object_storage_region_name` map to the SDK's
`ObjectStorageGeneratorConfig.endpoint_url` and `.region_name`. Omit the endpoint
to use the S3 default. Miles uses `object_storage_uri_prefix` for trainer
publication. See the [backend setup examples](S3_DELTA_WEIGHT_REFIT.md).

Azure generator reads require the Python `azure` extra and `az://container/blob`
sources. Authentication uses `AZURE_STORAGE_CONNECTION_STRING` first. Otherwise,
set `AZURE_STORAGE_ACCOUNT_URL` or `AZURE_STORAGE_ACCOUNT_NAME` and use
`DefaultAzureCredential`, including managed identity or AKS workload identity.
Azure rejects `object_storage_endpoint_url` and `object_storage_region_name`.
The generator supports full checkpoints and XOR delta replay. The server still
rejects Azure publications, and exact desired-version startup remains S3-only.
Azure generator configuration does not yet provide a complete publication workflow.

## Staging, streaming, and verification

Trainer staging choices govern how framework tensors remain valid while a version is published; see [FSDP precision and trainer staging](#fsdp-precision-and-trainer-staging) below. Generator streaming limits bound registered receive arenas; scratch and engine-owned materialization need additional memory. Read [staged transfer architecture](ARCHITECTURE.md#staged-transfer-caching-and-failure-handling) before tuning arena sizes, descriptor reuse, checksum verification, or throughput thresholds.


## Generator refit checkpoint cache


In vLLM weight-transfer initialization, `refit_checkpoint_max_size_gb` defaults to
`MX_REFIT_CHECKPOINT_MAX_SIZE_GB` (2000 decimal GB per model). An explicit
init-info value, including `null`, takes precedence over the environment setting.
Other initialization paths retain their existing 2000 GB default.

Streamed full replacement may need space for the protected old full checkpoint,
a materialized checkpoint with deltas applied, and the incoming full checkpoint
at the same time. Size the quota and backing volume for this peak plus retained
delta artifacts and headroom; increasing the quota does not add disk capacity.

At initialization, the generator caps this quota at existing cache bytes plus
free disk space and
logs any reduction. Free space is checked again before known writes and copies,
including when `null` disables the configured quota. See the
[S3 refit cache configuration](S3_DELTA_WEIGHT_REFIT.md#refit_checkpoint_dir)
for setup and eviction behavior.

## FSDP precision and trainer staging


The FSDP publisher transfers floating tensors in BF16 by default. Frameworks
whose models require higher precision for selected tensors can provide exact
state-dict names through the trainer context:

```python
import torch
from modelexpress_rl import FSDPTrainerContext, ModelExpressTrainerConfig

config = ModelExpressTrainerConfig(
    engine_context=FSDPTrainerContext(
        wire_dtype_overrides={"model.layers.0.mlp.router.selection_bias": torch.float32},
    ),
    model_name="my-model",
    device_id=0,
    server_url="localhost:8001",
)
```

Model-specific name selection belongs to the framework. Overrides accept
`torch.float16`, `torch.bfloat16`, and `torch.float32`; these are unquantized
tensor dtypes, not packed FP8/FP4 format support. Other tensors retain the BF16
default. An unknown name or non-floating tensor is rejected rather than silently
ignoring a precision exception. Names must match the state dict supplied by that
trainer, including any wrapper prefixes.

`COPY_TO_HOST` and `COPY_TO_DEVICE` register persistent buffers in each selected
dtype and copy subsequent versions into those buffers. `IN_PLACE` requires the source dtype to
match the selected dtype. The adapter copies the override mapping at creation;
changing the caller's mapping does not change an active adapter. A source dtype
change after initialization is rejected before writing a new version. Recreate
the adapter and rebind when the model's precision policy changes.

Manifests describe each served tensor's dtype and element size, and byte totals
sum their actual sizes. Existing receiver dtype conversion remains available,
but casting a rounded BF16 value back to FP32 cannot recover source precision.
Verify installed parameters and generation separately from transfer completion.

### Choosing trainer staging for synchronous refits

Choose an existing `TrainerStagingMode` explicitly for the integration:

| Mode | Recommendation | Lifetime and memory cost |
|---|---|---|
| `IN_PLACE` | First choice and lowest staging latency for synchronous integrations. Trainers wait for receivers to finish and retire/delete the version before updating its source bytes. | Registers existing contiguous storage without a second weight copy. Addresses must remain stable across updates; bytes must remain immutable while the version can be read. |
| `COPY_TO_HOST` | First choice when `IN_PLACE` is unavailable, including trainer-side dtype conversion or moving source storage. | Keeps a persistent wire-format copy in pinned host RAM on CUDA hosts. GPU-to-host staging and host-to-receiver transfer add latency, but avoid a persistent second weight copy in VRAM. |
| `COPY_TO_DEVICE` | Explicit exception when measured latency justifies the VRAM cost, typically for small models where `IN_PLACE` is unavailable. | Keeps an additional wire-format copy on the trainer device. It is generally faster than host staging but can consume substantial VRAM; budget all rank-local shards, including replicated tensors. |

`IN_PLACE` cannot perform trainer-side conversion: each source must already have
the selected transfer dtype and representation. A conversion plugin also rules
out this mode. CPU offload or state-dict rematerialization can invalidate its
stable-address requirement even when the dtypes match. Do not silently fall back
to a device copy when these checks fail.

The FSDP adapter supports all three modes for full-tensor payloads. Host staging
preserves the per-tensor dtype overrides above and registers its persistent CPU
buffers once. CUDA copy completion is fenced before publication; the served
snapshot and its registration remain live until version retirement. Conversion
can still require temporary device workspace; host staging eliminates the
persistent wire-format device copy, not every transient allocation. Support for
arbitrary conversion plugins remains adapter-specific.

FP32 trainer shards transferred as BF16 with selected FP32 exceptions require
conversion, so they cannot use `IN_PLACE`. Use `COPY_TO_HOST` for this precision
policy unless a measured latency requirement justifies the additional VRAM
needed by `COPY_TO_DEVICE`.

Trainer `COPY_TO_HOST` is independent of the generator's `staging_device="cpu"`
setting: the former changes where source snapshots live, while the latter
changes where receivers land incoming bytes. Host-source manifests use the
internal `mx.reshard.shard_table.v2` format, so both ends must run a host-aware
client. Older readers reject this format. Device-only manifests retain v1.


### Replica metadata diagnostics

`none` sorts workers by ID. `rotate` sorts them and rotates the order by engine
rank; it is the default. `random` shuffles once when the complete group is
constructed; its list order remains stable for that mesh generation. Metadata prefetch is
limited to replica members of logical-shard groups needed by the transfer plan.
An unavailable nonselected replica remains unresolved and can be retried later;
`replica_metadata_prefetch_unavailable` records incomplete prefetch. Digest and
identity errors still fail preparation. Debug validation requires all published
replicas to resolve and fails if any is unavailable.
Replica equivalence compares canonical tensor coverage and geometry while
ignoring physical addresses, which vary by replica.
