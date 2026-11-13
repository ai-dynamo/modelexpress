---
name: modelexpress-rl-refit
description: Wire an RL trainer (Megatron, FSDP, or Miles) and a vLLM or SGLang generator through ModelExpress for live weight refit. Use to choose between the NCCL M2N collective path, the NIXL reshard path, and the S3 delta path, integrate any of them, run the reference end-to-end refit, or diagnose refit failures such as group admission, epoch changes, version leases, deadline expiry, and delta lineage or checkpoint-cache errors; not for initial model loading.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - modelexpress
    - reinforcement-learning
    - weight-refit
    - nccl
    - nixl
    - s3
    - vllm
    - sglang
    - miles
---

# ModelExpress RL Refit

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Purpose

Move new policy weights from RL trainer ranks into live generator engines through
ModelExpress (MX). Pick the right transfer path, wire both sides with the
public `modelexpress_rl` APIs, prove the refit with the checked-in example,
and, when a refit fails, name the failing layer and the exact recovery action.

## Prerequisites

- A ModelExpress server reachable from every trainer and generator rank. Both
  refit services are Redis-only: start the server with
  `MX_METADATA_BACKEND=redis` and `REDIS_URL` (see `docs/CONFIGURATION.md`).
- The `modelexpress` Python distribution on trainer and generator images
  (`pip install modelexpress`). RL APIs import from `modelexpress_rl`.
- NCCL M2N path only: `nccl4py`, `nccl-extensions[cu12]` or `[cu13]` matching
  the CUDA major, and libnccl 2.30.7 or newer mapped into the process.
- NIXL path only: a working NIXL install and, for vLLM, the `modelexpress_rl`
  plugin loaded (automatic unless `VLLM_PLUGINS` filters it out).
- S3 delta path only: an S3 or S3-compatible bucket (trainers write, generators
  read), the same safetensors seed checkpoint on every trainer and generator
  host, and node-local disk for the generator checkpoint cache. SGLang needs a
  build with `--modelexpress-config` and `/update_weights_from_modelexpress`;
  Miles needs its `modelexpress` transfer mode.

## Instructions

### 1. Preflight The Worker Environment

Run inside the trainer or generator container with the worker's environment:

```bash
python3 scripts/refit_preflight.py --path m2n --endpoint "${MX_ENDPOINT}"
python3 scripts/refit_preflight.py --path nixl --endpoint "${MX_ENDPOINT}"
```

It is read-only: it resolves the libnccl the process actually maps, checks the
required packages, validates every `MX_NCCL_REFIT_*` override the way the
client does, and probes TCP reachability of the server. Fix every `fail`
before writing integration code. The script has no S3 delta mode: on that path
vLLM's `init_weight_transfer_engine` validates the seed, the cache directory,
and the READY base version before any update; SGLang does so in its first
refit.

### 2. Choose The Transfer Path

The three paths are siblings; none imports another.

| Question | NCCL M2N collective (`modelexpress_rl.collective`) | NIXL reshard (`modelexpress_rl`, `modelexpress_rl.refit.reshard`) | S3 delta (`modelexpress_rl`, `WRITE_TO_STORAGE`) |
|---|---|---|---|
| Transfer model | Trainer and generator ranks enter one `nccl.m2n.reshard` collective per round; MX brokers rendezvous, admission, rank assignment, and the NCCL unique id | Trainer publishes an immutable `WeightVersion`; generators pull one-sided over NIXL when they choose | Trainer uploads XOR deltas (periodically a full HF checkpoint) to S3 as immutable versions; generators download, apply them to a local checkpoint cache, and reload |
| Engine adapters | Integrator supplies `Publisher` and `Loader`. No Megatron `Publisher` exists; the vLLM `Loader` and FSDP2/DeepSpeed `Publisher`s are reference code under `examples/` | Shipped: `MegatronTrainerContext`, `FSDPTrainerContext`, vLLM and SGLang plugins | Trainer: any framework that yields canonical HF tensor buckets; Miles' `modelexpress` mode. Generator: vLLM plugin, SGLang `--modelexpress-config` |
| Topology | Single node only; a cross-node reshard has failed inside NCCL's transport | No single-node restriction documented; Kubernetes and Dynamo examples | No network path between trainer and generator needed; both reach S3 and the MX server |
| Extra features | Layer-group pipelining, plan-digest agreement | Exact-version cold start, generator-to-generator peer reads, version leases | Durable, replayable versions; multi-revision catch-up; vLLM peer fallback; full checkpoints reset the chain |

S3 delta pays for its decoupling on every update: CPU XOR and zstd work on both
sides, an S3 upload and download, and a reload of the live engine from the
prepared checkpoint.

Decide with these rules, in order:

1. Trainer and generators have no RDMA path between them (separate clusters,
   clouds, or networks), each version must be durable in object storage, or
   the trainer is Miles, whose only ModelExpress transfer mode
   (`--update-weight-transfer-mode modelexpress`) is S3 delta: use **S3 delta**.
2. Trainer is Megatron, deployment is multi-node, or generators must refit on
   their own schedule: use **NIXL**.
3. Exact-version cold start or generator peer reads are needed: use **NIXL**.
4. Trainer and generator ranks can run each round in lockstep on one node, the
   NCCL requirements hold, and you will own the `Publisher`/`Loader`: **M2N**
   is an option. Validate it with `DIFF=1` before relying on it.

State which rule decided it.

### 3. Wire The Trainer And Generator

NIXL path (full detail: `references/nixl-reshard-refit.md`):

1. Trainer rank: `ModelExpressTrainerClient.initialize(ModelExpressTrainerConfig(engine_context=...))`
   with `FSDPTrainerContext()` (pass the FSDP `state_dict()`) or
   `MegatronTrainerContext()` (pass `MegatronTensorSpec`s from
   `build_megatron_tensor_specs`), then `bind_tensors(...)`.
2. Orchestrator: `ModelExpressControlClient.connect(...)`,
   `create_trainer_mesh(...)` with every worker's binding,
   `create_weight_version(..., trainer_mesh_id=...)`.
3. Trainer: `publish_version(version=...)`; the version becomes `READY` when
   every logical shard is published.
4. vLLM: start with `--weight-transfer-config '{"backend":"modelexpress"}'`,
   pause generation, then drive `init_weight_transfer_engine`,
   `start_weight_update`, `update_weights` with
   `{"update_info": {"version_id": ...}}`, and `finish_weight_update` on every
   worker; verify the installed version before resuming.
5. Retire: `delete_weight_version`, `release_version`, `delete_trainer_mesh`.

M2N path (full detail: `references/m2n-collective-refit.md`):

1. Each worker process builds one `CollectiveRendezvous(grpc_channel)`.
2. Trainer: `RefitClientTrainer(...)`, `initialize(publisher, source_partition=...)`.
   Generator: `RefitClientGenerator(...)`, `initialize(loader)`. Same
   `trainer_slots`, `generator_slots`, and `model_name` on all participants.
3. Every worker: `compute_plan()`; all must return before any round starts.
4. Per round on every participant: `create_operation(version, key_prefix=...)`,
   `start_weight_update(version, operation_id)`, `publish_weights` or
   `update_weights` per layer group in the same order, `finish_weight_update(version)`.
5. Terminal: `cleanup()`, then `CollectiveRendezvous.close()`.

S3 delta path (full detail: `references/s3-delta-refit.md`):

1. Orchestrator, once: `create_weight_version(uid="v0", payload_format=FULL_TENSOR, state=READY, ...)`,
   a catalog-only base for the shared seed checkpoint.
2. Each publishing rank: `ModelExpressTrainerClient.initialize(ModelExpressTrainerConfig(staging_mode=WRITE_TO_STORAGE, payload_format=XOR_DELTA, process_group=..., object_storage=ObjectStorageConfig(...)))`,
   then `prepare_delta_base(tensor_iter=...)` before the first optimizer step.
3. Generators: vLLM with `--weight-transfer-config '{"backend":"modelexpress"}'`
   and one `POST /init_weight_transfer_engine`, or SGLang with
   `--modelexpress-config`. Same model name, base id, and seed as the trainer.
4. Per update: rank 0 `create_weight_version(uid="vN", base_version_id="vN-1", state=STAGING)`
   (or `FULL_HF_CHECKPOINT` with no base); every publishing rank
   `stage_shard(...)` then `publish()`; rank 0 marks `vN` READY.
5. Pause generation, apply `vN` (vLLM: `start_weight_update`, `update_weights`
   with `{"update_info": {"version_id": "vN"}}`, `finish_weight_update`;
   SGLang: `update_weights_from_modelexpress`), resume.

Miles does steps 1, 2, 4, and 5 itself with `--update-weight-transfer-mode modelexpress`
and `--modelexpress-config`; set `full_hf_checkpoint_interval` and use a fresh
model name, base id, and S3 prefix per run.

### 4. Run The Reference End-To-End Refit

M2N, one node with `T + G` GPUs, from `examples/rl/m2n_collective_refit/`:

```bash
export MODEL=/models/Qwen3-4B
export MX_ENDPOINT=modelexpress-server.<namespace>.svc.cluster.local:8001
DIFF=1 PROC_TIMEOUT_S=1800 ./run_e2e.sh
```

Pass is `E2E PASS` in `generator.log` and every `.rc` file reading `0` under
`$OUT` (default `/work/out/$RUN`). Always pass `DIFF=1`: the per-parameter
comparison is the acceptance test, and token equality alone is not sound on
larger models. Set `PROC_TIMEOUT_S` for unattended runs; an `.rc` of `124` or
`137` means the process bound fired.

NIXL with Dynamo and vLLM on Kubernetes: follow
`examples/rl/dynamo_vllm_reshard_refit/README.md`; success is `E2E PASS` in
`kubectl logs -n "$NAMESPACE" job/mx-vllm-rl-job`.

S3 delta with Dynamo and vLLM on Kubernetes, two GPUs, no training framework:
follow `examples/rl/dynamo_vllm_s3_delta_refit/README.md` and run
`examples/rl/dynamo_vllm_s3_delta_refit/run.sh`. It publishes a full `v1` and
deltas `d2`, `d3`, checks that one generator loaded from S3 and the other from
it over P2P, then replaces a generator; the P2P checks need RDMA between the
two GPUs. Success is both `E2E PASS: ...` and `RESTART ...` at the end of the
output. For a Megatron training loop that
publishes a delta every step, `examples/rl/vime_dynamo_delta_refit/` ends with
`TRAINING COMPLETE`.

### 5. Diagnose A Failed Refit

Classify into one bucket with `references/failure-decision-tree.md`:

- environment/dependencies (preflight fails, `NcclUnavailableError`)
- group admission and formation (`GroupNotReadyError`, join rejected)
- epoch fencing (`EpochChangedError`, `TransferBindingError`, rejected reports)
- deadline expiry and wedged communicators (`CommunicatorWedgedError`, hangs)
- plan coverage and correctness (`PlanCoverageError`, `DIFF=1` mismatches)
- version leases and publication (NIXL: version not `READY`, active-lease rejections)
- delta lineage and checkpoint cache (S3: base mismatch, checksum mismatch,
  replay chain limit, cache capacity, failed install)

Recover in this order: fix the layer, re-run `compute_plan` (M2N), re-stage
(NIXL), or re-publish or re-request the version (S3), and restart any worker
whose engine state is uncertain.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/refit_preflight.py` | Read-only environment and reachability check for one refit worker | `--path {m2n,nixl}`, `--endpoint host:port`, `--timeout` |

Invoke via the agentskills.io `run_script()` protocol:

```python
run_script("scripts/refit_preflight.py", args=["--path", "m2n", "--endpoint", "mx-server:8001"])
```

## Examples

Check a generator image before an M2N run:

```bash
python3 scripts/refit_preflight.py --path m2n --endpoint mx-server:8001
```

Check a vLLM worker that filters plugins:

```bash
VLLM_PLUGINS=modelexpress_rl python3 scripts/refit_preflight.py --path nixl
```

## Output Contract

Return:

- chosen path and the rule that decided it
- trainer and generator wiring, as the exact API calls in order
- the run or verification command and its pass signal
- for failures: bucket, strongest signal (exception class or server message),
  likely cause, exact next action, and whether the engine must be restarted

## Limitations

- The M2N path ships no Megatron `Publisher` or production vLLM `Loader`; the
  example's `VllmLoader` is a reference implementation.
- M2N is validated on one node only, and the example hardcodes
  `source_partition_count=1` (no pipeline stages).
- `MX_NCCL_REFIT_TRANSFER_TIMEOUT_S` cannot interrupt a reshard blocked in
  native code; the launcher must bound each process.
- Neither M2N nor NIXL rolls back a failed install; a failed refit needs an
  engine restart before serving resumes. S3 delta keeps the last installed
  checkpoint: request the same version (or a newer one built on it) again and
  it is reloaded in full; generation stays paused until that succeeds.
- S3 delta: Miles' `modelexpress` mode requires Megatron, no colocation, no
  LoRA, and abort or retract pausing. The vLLM HTTP update routes need
  `VLLM_SERVER_DEV_MODE=1`; keep them network-isolated.
- The preflight script cannot see the server's Redis backend or the remote
  peers; it checks only the local process.

## Troubleshooting

| Symptom | Likely cause | Next step |
|---|---|---|
| `GroupNotReadyError` naming slots in `.missing` | Those ranks never joined, or their registration expired | Start or fix the missing ranks, then retry `compute_plan` |
| `EpochChangedError` with `.disagreeing` set; group stays `FORMING` | Plan digest differs: different plan, `m2n_abi_version`, or `MX_NCCL_REFIT_BLOCKING_DATA_PLANE`/`GROUPED_SUBMISSION` | Make the plan and those settings identical everywhere |
| `CommunicatorWedgedError` | Lane abort overran `MX_NCCL_REFIT_ABORT_TIMEOUT_S` | Exit the process with non-zero `os._exit`; restart with a new `worker_id` |
| vLLM refit fails before mutation with `MX_P2P_METADATA=0` | Readers of central-coordinator addresses cannot be accounted for | Unset `MX_P2P_METADATA` (default `1`) |
| Trainer: `target base ... does not match retained base ...` | The new version's `base_version_id` is not the version this trainer client last published (client recreated, or a publish was skipped) | Publish the next version as a `FULL_HF_CHECKPOINT` (no base) to start a new chain |
| Generator: `canonical target checksum differs for ...` | Trainer and generator seeds differ, or the published tensors are not canonical HF tensors | Point both sides at the same seed snapshot; publish a full checkpoint to recover |
| Generator: `replay exceeds maximum chain length 64 ...` | Too many deltas since the last full checkpoint: the generator is far behind, cold-starting, or replays from the full root on every refit (vLLM with peers) | Set `full_hf_checkpoint_interval` (Miles) or publish full checkpoints; raise `max_replay_chain_length` only as a stopgap |

## References

- `references/m2n-collective-refit.md`: M2N client API, SPI, call sequence, errors, configuration.
- `references/nixl-reshard-refit.md`: NIXL trainer, control, and generator lifecycle and settings.
- `references/s3-delta-refit.md`: S3 delta trainer, vLLM, SGLang, and Miles lifecycle, cache sizing, and settings.
- `references/failure-decision-tree.md`: bucket-by-bucket signals, checks, and actions.
- Source docs: `docs/NCCL_M2N_REFIT.md`, `docs/rl/README.md`, `docs/rl/ARCHITECTURE.md`, `docs/rl/CONFIGURATION.md`, `docs/rl/S3_DELTA_WEIGHT_REFIT.md`.
