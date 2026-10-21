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
| `MX_REFIT_CHECKPOINT_DIR` | (unset) | Host-local cache for RL S3 full checkpoints, deltas, and materialized checkpoints; the S3 strategy skips when it is unset |
| `MX_REFIT_METADATA_PORT` | `7555` | Base NIXL metadata-listener port for RL generator refit; each rank adds its local device ID. Kept separate from `MX_METADATA_PORT`, which may remain owned by the boot-time loader |
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
| `MX_REFIT_PACK_MODULES` | `1` | Coalesce consecutive owning-module batches in a bounded refit up to the same `max_staging_bytes`, trading a larger arena residency for fewer of them. The packed batch reads exactly the bytes its modules read — same copies, planned bytes and READ descriptors — and modules pulling the same complete source are never packed together. Set to `0` to use one module per batch |
| `MX_RESHARD_MAX_GBPS` | `0` | Per-rank fabric ceiling in Gbps. A measured wire rate above it means the timing is wrong rather than the transfer being fast, so the refit is rejected. `0` disables the check, since only the operator knows the real per-rank limit |
| `MX_RESHARD_MIN_GBPS` | `0` | Per-rank floor in Gbps. Below it, the refit emits a `refit-slow-throughput-v1` JSON warning naming the rate, the bound and the shortfall. Applies to both receivers — the Megatron slice-reshard receiver and the staged receiver the FSDP trainers pull over, including its peer-to-peer pull — so one setting covers a job whichever path it refits on. Warns rather than aborting, unlike the ceiling: an impossible rate means the payload never moved, but a slow one is still correct, so enforcement belongs in a CI gate reading the record rather than in a running job. `0` disables it. Worth setting for any throughput run — a 20x collapse has been observed with byte counts exact, descriptor counts exact, coverage 100%, fallback 0 and no error anywhere, and without a lower bound there is nothing in the telemetry that dissents. When choosing a value, note that it is compared per rank against the rate one receiver sees *while its siblings are also receiving*, which is well below the rate a single receiver reaches alone; sizing it against the solo number puts the floor above the healthy concurrent rate and it will fire on good runs. Aim between the two — roughly the geometric mean of the collapsed rate and the healthy concurrent rate leaves both verdicts off a tight margin |
| `MX_RESHARD_PUBLISH_DIGEST` | `0` | Have each trainer publish a position-sensitive digest of every shard it advertises, so a receiver can later confirm it installed the bytes the publisher held. Off by default: the reduction costs a pass over every published tensor, which is large next to a ~1.5 s wire, so turn it on when qualifying a build rather than when measuring throughput |
| `MX_MODEL_NAME_OVERRIDE` | (unset) | Override vLLM's MX model identity without changing its model-loading path. Set before worker startup and use the same name for RL trainers and WeightVersions. Unset or empty preserves vLLM's configured model path/ID, including rewritten S3 cache paths. |
| `MX_REFIT_DELTA_BUCKET_BYTES` | `536870912` (512 MiB) | Optional framework integration override for canonical delta tensor-bucket size; native framework bucket settings take precedence. |
| `MX_REFIT_DELTA_WORKERS` | `min(32, CPU count)` | CPU worker pool for XOR delta computation and application. |
| `MX_REFIT_CHECKSUM_FORMAT` | `adler32` | Checksum format written by canonical S3 trainers. |
| `MX_REFIT_FULL_CHECKPOINT_BATCH_BYTES` | `4294967296` (4 GiB) | Maximum tensor bytes grouped into one full-checkpoint safetensors object; oversized tensors occupy a separate object. |
| `MX_REFIT_DELTA_SURGICAL` | `false` | Experimental vLLM delta install: reload only the destination modules touched by the delta chain instead of the whole prepared checkpoint. Falls back to a full reload when lineage or the live version is unknown, or when a fed module is incomplete. See the [S3 delta refit guide](S3_DELTA_WEIGHT_REFIT.md). |
| `MX_REFIT_DEBUG_VALIDATE_PLAN` | `0` | Recheck cached whole-model and owning-module transfer coverage. |
| `MX_REFIT_DEBUG_VALIDATE_GENERATOR_LAYOUT` | `0` | Validate that the generator layout has not changed. |
| `MX_REFIT_PREFETCH_REPLICA_METADATA` | `0` | Fetch stable metadata for all replicas in needed logical-shard groups on first preparation. |
| `MX_REFIT_DEBUG_VALIDATE_REPLICAS` | `0` | Check that replicas in each logical-shard group have equivalent canonical coverage. |

| `MX_REFIT_DELTA_SURGICAL` | `false` | Experimental vLLM delta install: reload only the modules whose checkpoint tensors the delta chain changed since the engine's last checkpoint install, instead of the whole prepared checkpoint. Falls back to a full reload when the lineage or live version is unknown, or when a fed module is incomplete. See the S3 delta refit guide. |
| `MX_REFIT_FULL_STREAMING` | `false` | Experimental vLLM full-HF refit: use ModelStreamer once and tee tensors to a bounded background disk writer. Requires distributed ModelStreamer for TP greater than one. Requires `runai-model-streamer>=0.17.0`. Checksummed artifacts and delta replay retain canonical preparation. For S3-compatible storage, set matching `AWS_ENDPOINT_URL` and `AWS_DEFAULT_REGION` on all generator workers; `object_storage_endpoint_url` / `object_storage_region_name` configure only the index reader, not streamed shard reads. See the architecture section on streamed full refit. |
| `MX_REFIT_STREAM_WINDOW_LAYERS` | `2` | Decoder layers per ranged ModelStreamer request during streamed full refit. Bounds GPU memory held by partially loaded layers; `0` streams the whole checkpoint in one request. |
| `MX_REFIT_METADATA_PORT` | `7555` | Base NIXL listen port for an RL generator's refit client; effective port is `MX_REFIT_METADATA_PORT + device_id`, separate from a boot-time loader manager |

## Staging, streaming, and verification

Trainer staging choices govern how framework tensors remain valid while a version is published; see [FSDP precision and trainer staging](#fsdp-precision-and-trainer-staging) below. Generator streaming limits bound registered receive arenas; scratch and engine-owned materialization need additional memory. Read [staged transfer architecture](ARCHITECTURE.md#staged-transfer-caching-and-failure-handling) before tuning arena sizes, descriptor reuse, checksum verification, or throughput thresholds.


## Generator refit checkpoint cache


`refit_checkpoint_max_size_gb` defaults to 2000 GB per model. At initialization,
the generator caps this quota at existing cache bytes plus free disk space and
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
