# RL Refit Failure Decision Tree

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

The characteristic M2N failure is a hang; every mechanism below exists to turn
one into an attributable error. Classify first, then act on one layer.

## Environment Or Dependencies

Signals:

- `NcclUnavailableError` from `compute_plan`
- `RuntimeError` naming `NCCL_COMM_BLOCKING`
- `ValueError` naming an `MX_NCCL_REFIT_*` variable at client construction
- vLLM refit routes missing, or the weight-transfer backend not found

Checks:

```bash
python3 scripts/refit_preflight.py --path m2n --endpoint "${MX_ENDPOINT}"
python3 scripts/refit_preflight.py --path nixl --endpoint "${MX_ENDPOINT}"
python3 -c 'import ctypes; v=ctypes.c_int(); ctypes.CDLL("libnccl.so.2").ncclGetVersion(ctypes.byref(v)); print(v.value)'
```

Next action: install `nccl4py` and the `nccl-extensions` extra matching the
CUDA major; make sure libnccl 2.30.7 or newer is the one mapped (a value of
`23007` or more above); unset `NCCL_COMM_BLOCKING`; add `modelexpress_rl` to
`VLLM_PLUGINS` when it is set.

## Group Admission And Formation (M2N)

Signals:

- `GroupNotReadyError` after `MX_NCCL_REFIT_GROUP_TIMEOUT_S`; `.missing` lists absent slots
- join rejected by the server with one of:
  - `slot ... is not expected for role ...` (slot not in the declared slot lists)
  - `worker ... has no live matching registration` (registration expired or model name differs)
  - `slot ... is already bound to a different role or ordinal`
  - `another slot already owns ... index ...` (two workers claim the same `index_in_role`)
  - `worker ... is already admitted under another slot`

Checks:

- Are the processes for the `.missing` slots running and pointed at the same `MX_ENDPOINT`?
- Do all participants pass identical `trainer_slots`, `generator_slots`, and `model_name`?
- Is each `slot_id` / `index_in_role` pair unique and consistent with its position?
- Did a worker stall longer than `MX_NCCL_REFIT_REGISTRATION_TTL_S` (default 90 s) without heartbeating?

Next action: start or fix the missing ranks and retry `compute_plan`. vLLM
needs about a minute to reach the rendezvous; a trainer that arrives first
waits, so raise `MX_NCCL_REFIT_GROUP_TIMEOUT_S` rather than restarting early
ranks.

## Epoch Fencing (M2N)

Signals:

- `EpochChangedError` from `compute_plan` or `start_weight_update`
- `EpochChangedError.disagreeing` non-empty: the group stays `FORMING`
- `TransferBindingError` from `start_weight_update`
- server: `collective group ... is ... at epoch ..., not READY at the expected epoch ...; no transfer was created`
- server: `collective transfer ... is still in flight at this group epoch`
- server: `report for epoch ... was rejected; the group is at epoch ...`

Causes: a worker restarted (new `worker_id`, new generation), a registration
expired, a previous round failed (failure reports move the epoch), or the plan
digest differs between participants.

Next action:

- Plain epoch change: re-run `compute_plan` on every worker, then create a new
  operation at the new epoch.
- `.disagreeing` set: make the plan identical (same checkpoint, layer
  groupings, `m2n_abi_version`, `MX_NCCL_REFIT_BLOCKING_DATA_PLANE`,
  `MX_NCCL_REFIT_GROUPED_SUBMISSION`) on the named slots first.
- Operation still in flight: let it finish, or report it FAILED
  (`report_failure` on the trainer); the next epoch change also clears it.

## Deadline Expiry And Wedged Communicators (M2N)

Signals:

- a round call raises after `MX_NCCL_REFIT_TRANSFER_TIMEOUT_S`; the operation is FAILED
- `CommunicatorWedgedError` carrying `group_id`, `epoch`, `version`, `lane_id`
- the log repeats that a reshard call is still blocking, with no error
- `run_e2e.sh` `.rc` files read `124` or `137`

Checks:

- Did one peer die mid-collective? Its log ends first.
- Is the run cross-node? Only single-node M2N is validated.
- With `MX_NCCL_REFIT_GROUPED_SUBMISSION=1`, is GIN available on every rank?

Next action: on `CommunicatorWedgedError` or a blocked native call, terminate
the process with a non-zero `os._exit` and restart it with a new `worker_id`;
peers re-form at the next epoch. Bound every process from outside
(`PROC_TIMEOUT_S` in the example). Raise `MX_NCCL_REFIT_ABORT_TIMEOUT_S` only if
a healthy abort on the deployed libnccl needs longer than 10 s.

## Plan Coverage And Correctness

Signals:

- `PlanCoverageError` from `initialize` or `compute_plan`
- `build_plan` in the example refuses a parameter whose first dimension does not divide `T`
- `E2E FAIL` with `DIFF=1`, or the positive control does not fire
- `E2E FAIL` without `DIFF=1` on a larger model while weights are correct

Next action: fix the `Publisher`/`Loader` so `capture()` covers
`parameter_names()` on both sides. Re-run with `DIFF=1`: per-parameter equality
is the acceptance test, and greedy token output can diverge with every weight
byte correct. After a failed `DST=sharded` round, restart the engine; it wrote
into live weights.

## Version Leases And Publication (NIXL)

Signals:

- version never reaches `READY` after `publish_version`
- `published version is not READY` in the reshard example coordinator
- server: `cannot change trainer mesh generation while a linked version has an active lease`
- server: `weight version has an active lease` when a trainer releases its published shard
- server: `version lease is owned by a different worker`
- vLLM refit fails before mutation with `MX_P2P_METADATA=0`
- `IncompleteRefit` after a vLLM reload

Checks:

- Did every trainer worker in the mesh call `bind_tensors` and `publish_version`?
- Was the mesh updated or a trainer restarted after the version was created
  (generation mismatch fails trainer source resolution)?
- Is a generator still holding a lease from an unfinished `stage_weight` (missing `release()`)?

Next action: publish from every logical shard; drain generator leases
(`finish_weight_update` or `release()`) before updating the mesh or retiring a
version; unset `MX_P2P_METADATA`. After `IncompleteRefit` or any failure once
installation started, keep generation paused and restart the engine.
