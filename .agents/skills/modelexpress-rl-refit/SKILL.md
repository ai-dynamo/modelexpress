---
name: modelexpress-rl-refit
description: Wire an RL trainer (Megatron or FSDP) and a vLLM generator through ModelExpress for live weight refit. Use to choose between the NCCL M2N collective path and the NIXL reshard path, integrate either one, run the reference end-to-end refit, or diagnose refit failures such as group admission, epoch changes, version leases, and deadline expiry; not for initial model loading.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - modelexpress
    - reinforcement-learning
    - weight-refit
    - nccl
    - nixl
    - vllm
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
before writing integration code.

### 2. Choose The Transfer Path

The two paths are siblings; neither imports the other.

| Question | NCCL M2N collective (`modelexpress_rl.collective`) | NIXL reshard (`modelexpress_rl`, `modelexpress_rl.refit.reshard`) |
|---|---|---|
| Transfer model | Trainer and generator ranks enter one `nccl.m2n.reshard` collective per round; MX brokers rendezvous, admission, rank assignment, and the NCCL unique id | Trainer publishes an immutable `WeightVersion`; generators pull one-sided over NIXL when they choose |
| Engine adapters | Integrator supplies `Publisher` and `Loader`. No Megatron `Publisher` exists; the vLLM `Loader` and FSDP2/DeepSpeed `Publisher`s are reference code under `examples/` | Shipped: `MegatronTrainerContext`, `FSDPTrainerContext`, vLLM and SGLang plugins |
| Topology | Single node only; a cross-node reshard has failed inside NCCL's transport | No single-node restriction documented; Kubernetes and Dynamo examples |
| Extra features | Layer-group pipelining, plan-digest agreement | Exact-version cold start, generator-to-generator peer reads, version leases, S3/Azure object storage |

Decide with these rules, in order:

1. Trainer is Megatron, deployment is multi-node, or generators must refit on
   their own schedule: use **NIXL**.
2. Exact-version cold start or generator peer reads are needed: use **NIXL**.
3. Trainer and generator ranks can run each round in lockstep on one node, the
   NCCL requirements hold, and you will own the `Publisher`/`Loader`: **M2N**
   is an option. Validate it with `DIFF=1` before relying on it.
4. Weights must go through object storage (S3 delta): neither; follow
   `docs/rl/S3_DELTA_WEIGHT_REFIT.md`.

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

### 5. Diagnose A Failed Refit

Classify into one bucket with `references/failure-decision-tree.md`:

- environment/dependencies (preflight fails, `NcclUnavailableError`)
- group admission and formation (`GroupNotReadyError`, join rejected)
- epoch fencing (`EpochChangedError`, `TransferBindingError`, rejected reports)
- deadline expiry and wedged communicators (`CommunicatorWedgedError`, hangs)
- plan coverage and correctness (`PlanCoverageError`, `DIFF=1` mismatches)
- version leases and publication (NIXL: version not `READY`, active-lease rejections)

Recover in this order: fix the layer, re-run `compute_plan` (M2N) or re-stage
(NIXL), and restart any worker whose engine state is uncertain.

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
- Neither path rolls back a failed install; a failed refit needs an engine
  restart before serving resumes.
- The preflight script cannot see the server's Redis backend or the remote
  peers; it checks only the local process.

## Troubleshooting

| Symptom | Likely cause | Next step |
|---|---|---|
| `GroupNotReadyError` naming slots in `.missing` | Those ranks never joined, or their registration expired | Start or fix the missing ranks, then retry `compute_plan` |
| `EpochChangedError` with `.disagreeing` set; group stays `FORMING` | Plan digest differs: different plan, `m2n_abi_version`, or `MX_NCCL_REFIT_BLOCKING_DATA_PLANE`/`GROUPED_SUBMISSION` | Make the plan and those settings identical everywhere |
| `CommunicatorWedgedError` | Lane abort overran `MX_NCCL_REFIT_ABORT_TIMEOUT_S` | Exit the process with non-zero `os._exit`; restart with a new `worker_id` |
| vLLM refit fails before mutation with `MX_P2P_METADATA=0` | Readers of central-coordinator addresses cannot be accounted for | Unset `MX_P2P_METADATA` (default `1`) |

## References

- `references/m2n-collective-refit.md`: M2N client API, SPI, call sequence, errors, configuration.
- `references/nixl-reshard-refit.md`: NIXL trainer, control, and generator lifecycle and settings.
- `references/failure-decision-tree.md`: bucket-by-bucket signals, checks, and actions.
- Source docs: `docs/NCCL_M2N_REFIT.md`, `docs/rl/README.md`, `docs/rl/ARCHITECTURE.md`, `docs/rl/CONFIGURATION.md`.
