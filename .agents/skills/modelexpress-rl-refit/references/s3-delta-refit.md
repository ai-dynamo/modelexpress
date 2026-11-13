# S3 Delta Refit

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Public API: the `modelexpress_rl` top-level namespace. Authoritative source:
`docs/rl/S3_DELTA_WEIGHT_REFIT.md` (setup, both backends, cache layout, S3
artifact contract). Reference integrations:
`examples/rl/dynamo_vllm_s3_delta_refit/` (framework-free lifecycle) and
`examples/rl/vime_dynamo_delta_refit/` (Megatron training loop).

## Model

- Trainer and generators start from the same local safetensors seed, registered
  as a catalog-only READY base version (`v0`); no seed bytes are uploaded.
- Each update is an immutable `WeightVersion` whose S3 URI is the exact
  `model.safetensors.index.json`. A version is either an `XOR_DELTA` against
  its `base_version_id` (zstd-compressed XOR bytes plus per-tensor Adler-32
  checksums of the reconstructed tensor) or a `FULL_HF_CHECKPOINT` with no base,
  which starts a new chain.
- Generators resolve the READY chain from their serving version to the target,
  download and apply it into a host-local checkpoint cache, then reload the
  live engine from that checkpoint.
- Trainer and generators never talk to each other: S3 and the MX catalog are
  the only shared state. No RDMA path between them is needed.

```text
trainer ranks ── stage_shard/publish ──▶ S3: v1/ (delta)  v2/ (delta)  v10/ (full) ...
     │                                        ▲
     └── create_weight_version / READY ──▶ MX catalog (Redis)
                                               │ resolve chain
generator ◀── download, XOR-apply, checksum ───┘ ──▶ reload live engine
```

## Trainer Side

One client per publishing rank, kept for the whole run; it retains the last
published version as its delta base.

```python
from modelexpress_rl import (
    ModelExpressControlClient,
    ModelExpressTrainerClient,
    ModelExpressTrainerConfig,
    ObjectStorageConfig,
    ObjectStorageSource,
    ObjectStorageType,
    TrainerStagingMode,
    WeightPayloadFormat,
    WeightVersionRef,
    WeightVersionState,
)

# Once, before any client initializes: catalog-only READY base.
control.create_weight_version(
    uid="v0", model_name=model_name, idempotency_key="init-v0",
    payload_format=WeightPayloadFormat.FULL_TENSOR,
    object_storage=ObjectStorageSource(
        storage_type=ObjectStorageType.S3,
        uri=f"{prefix}/v0/model.safetensors.index.json",  # never read
    ),
    state=WeightVersionState.READY,
)

trainer = ModelExpressTrainerClient.initialize(
    ModelExpressTrainerConfig(
        model_name=model_name,
        staging_mode=TrainerStagingMode.WRITE_TO_STORAGE,
        payload_format=WeightPayloadFormat.XOR_DELTA,
        server_url=mx_server,
        process_group=refit_group,          # Gloo group of publishing ranks only
        object_storage=ObjectStorageConfig(
            storage_type=ObjectStorageType.S3,
            uri_prefix=prefix,
            initial_base_version_id="v0",
            seed_checkpoint_path="/models/Qwen3-30B-A3B",
            region_name="us-west-2",        # endpoint_url= for MinIO/SeaweedFS
        ),
    )
)
trainer.prepare_delta_base(tensor_iter=hf_tensor_buckets())  # before the first optimizer step
```

Per update, in this order:

1. Rank 0 of `refit_group`: `create_weight_version(uid="vN", ...,
   payload_format=XOR_DELTA, base_version_id="vN-1", state=STAGING)` with the
   exact index URI, which must start with the trainer's `uri_prefix`. For a
   full checkpoint use `FULL_HF_CHECKPOINT` and no `base_version_id`. Barrier.
2. Every rank in `refit_group`: `staged = trainer.stage_shard(version=WeightVersionRef("vN"), tensor_iter=...)`
   or one `stage_shard(..., tensors=bucket)` per bucket, then `staged.publish()`.
   A rank with no tensors still submits `tensors=[]` once and publishes. Barrier.
3. Rank 0: `update_weight_version_state("vN", WeightVersionState.READY)`.
4. Apply on the generators (below). Publish `vN` before staging `vN+1`.

Tensor names must be canonical HF names that exist in the seed checkpoint.

## Generator Side: vLLM

```bash
export VLLM_SERVER_DEV_MODE=1 VLLM_PLUGINS=modelexpress_rl
export MX_SERVER_ADDRESS=modelexpress:8001 MX_MODEL_NAME_OVERRIDE="$MODEL_NAME"
export MX_GENERATOR_SOURCE_ORDER=OBJECT_STORAGE   # S3 only; see Source Order below
vllm serve /models/Qwen3-30B-A3B --tensor-parallel-size 4 \
  --weight-transfer-config '{"backend":"modelexpress"}'
```

Once per server, `POST /init_weight_transfer_engine` with `init_info`:
`model_name`, `server_url`, `object_storage_type: "S3"`,
`initial_base_version_id`, `seed_checkpoint_path`, `refit_checkpoint_dir`,
`refit_checkpoint_max_size_gb`, `object_storage_region_name` (and
`object_storage_endpoint_url` for non-AWS S3). It validates the seed and that
the base is READY on every worker before any update.

Per version: `POST /pause?mode=abort` → `/start_weight_update` →
`/update_weights` with `{"update_info": {"version_id": "vN"}}` →
`/finish_weight_update` with `{"weight_version": "vN"}` → `/resume`.

## Generator Side: SGLang

Requires an SGLang build with `--modelexpress-config` and
`POST /update_weights_from_modelexpress`, and MX's
`modelexpress_rl.inference.engines.sglang`. The same receiver keys go in the
JSON config; each TP worker creates its MX client on the first refit, so seed
and base-version errors surface there (that first refit also waits for the
seed working copy). SGLang refits from S3 only.

Per version: `/pause_generation` (`{"mode": "abort"}`) → `/flush_cache` →
`/update_weights_from_modelexpress` with
`{"weight_version": "vN", "flush_cache": false}` → `/continue_generation`.

## Miles Integration (Megatron + Miles-managed SGLang)

Requires a Miles build with the `modelexpress` transfer mode and the SGLang
build above. Miles owns the base version, publication, and engine refit:

```bash
--update-weight-transfer-mode modelexpress \
--modelexpress-config '{
  "model_name": "policy-run-unique-id",
  "server_url": "modelexpress:8001",
  "initial_base_version_id": "policy-run-unique-id-v0",
  "seed_checkpoint_path": "/models/Qwen3-30B-A3B",
  "refit_checkpoint_dir": "/local-nvme/mx-refit",
  "object_storage_uri_prefix": "s3://weight-updates/policy-run-unique-id",
  "object_storage_region_name": "us-west-2",
  "full_hf_checkpoint_interval": 10
}'
```

- Miles validates: Megatron backend, no colocation, no LoRA, and abort or
  retract pausing (not `in_place`).
- Use a fresh `model_name`, `initial_base_version_id`, and S3 prefix per run;
  version idempotency keys derive from the prefix and step.
- The first sync registers `v0` and captures the seed (`prepare_delta_base`)
  without uploading. Later syncs: create `vN`, every sender stages each bucket
  with `stage_shard(tensors=bucket)` and publishes, mark READY, pause and flush
  engines, `update_weights_from_modelexpress`, restore Miles' numeric
  version label through `/update_weight_version`, resume.
- Every `full_hf_checkpoint_interval` steps the version is a full checkpoint;
  omit the key to publish only deltas.
- Only sender ranks join the MX publication group; the others still take part
  in Megatron's weight gather.
- A failed install raises on every trainer rank and stops the run; nothing
  re-requests the version. A restarted run re-registers `v0` and re-captures
  the seed, so give it fresh ids and a fresh prefix.
- For large full checkpoints raise `--update-weights-timeout` (for example
  1800) and `--update-weight-engine-request-timeout` (for example 1200).

## Source Order (vLLM)

Generator peers are a source only when vLLM was loaded with
`--load-format modelexpress`; otherwise refit is S3-only regardless of
`MX_GENERATOR_SOURCE_ORDER`. With both `GENERATOR` and `OBJECT_STORAGE`
active, every S3 refit replays from the last `FULL_HF_CHECKPOINT`:

- a catalog-only `v0` cannot be that root, so publish a real full checkpoint
  first (the Dynamo reference example uploads its first version in full);
- `max_replay_chain_length` then bounds every refit, so publish full
  checkpoints at least that often.

In S3-only mode a generator replays only the revisions after the version it
serves.

## Checkpoint Cache Sizing

Per generator host, under `<refit_checkpoint_dir>/<quoted model name>/`:
`full/` (the seed is symlinked, not copied), `deltas/`, `chains/`,
`materialized/`, plus `state.json` and `active.json`.

- The first delta after a full checkpoint applies to a working copy of it;
  later deltas apply in place to the previous materialization.
- Size `refit_checkpoint_max_size_gb` (decimal GB, default 2000, capped at
  free disk) for the active lineage plus two copies of a downloaded full
  checkpoint. Without room for the working copy, the next delta copies the full
  checkpoint inline and its apply time jumps.
- Put the cache on node-local NVMe. All ranks on one host can share a cache;
  hosts without a shared filesystem each need their own.

## Settings Worth Knowing

| Setting | Default | Why |
|---|---|---|
| `MX_GENERATOR_SOURCE_ORDER` | peers, then S3, when peers exist | `OBJECT_STORAGE` for S3-only refit |
| `max_replay_chain_length` (receiver config) | `64` | Longest chain a refit replays; full checkpoints reset it |
| `MX_MAX_REPLAY_CHAIN_LENGTH` | `64` | Same limit for desired-version cold start |
| `MX_REFIT_DELTA_WORKERS` | `min(32, CPU count)` | CPU workers that compute and apply XOR deltas |
| `MX_S3_UPLOAD_WORKERS` / `MX_REFIT_DOWNLOAD_WORKERS` | `8` / `16` | Concurrent multipart uploads per trainer rank / shard downloads per generator |
| `MX_REFIT_FULL_CHECKPOINT_BATCH_BYTES` | 4 GiB | Tensor bytes per full-checkpoint safetensors object |
| `MX_REFIT_DELTA_SURGICAL` | `false` | vLLM reloads only modules the deltas touched |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | unset | Only when no IAM role or workload identity; trainers need write, generators read |

## Failure Semantics

- Catalog, chain-length, manifest, and capacity failures leave the cache as it
  was: fix the cause and request the same version again.
- A failed delta download, XOR apply, or checksum resets the cache head to the
  initial base. In S3-only mode the next delta then fails with
  `local checkpoint version 'v0' does not match exact base ...`; publish a
  `FULL_HF_CHECKPOINT` to start a new chain (or reinitialize the generator).
- None of the above touches live weights.
- If installation starts and fails, keep generation paused and request the
  same version, or a newer version built on it, again: the prepared checkpoint
  is reloaded in full (vLLM reports not ready until an update succeeds).
  Restart the engine only if that also fails. Re-requesting the previous
  version works only when it is a full checkpoint.
- A crash during preparation leaves `state.json` at `UPDATING`; initialization
  resets the cache to the initial base and the next refit replays from there.
