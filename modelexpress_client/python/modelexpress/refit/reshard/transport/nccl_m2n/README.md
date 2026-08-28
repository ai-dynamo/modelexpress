# NCCL M2N transport

This first integration supports NeMo/Megatron-style tensor inputs with
ModelExpress-owned PP transfer resources.

## Ownership and API flow

1. Control plane supplies one `_M2nPPGroupSpec` per locally owned PP pair. Key
   is globally stable `(trainer_stage, generator_stage)`.
2. `_M2nRuntime.create_pp_groups()` sorts specs, creates nonblocking parent
   NCCL4Py communicators, and creates one explicit CUDA stream per PP group.
3. `NcclM2nExecutor` atomically freezes topology and attaches itself to the
   runtime. Runtime keeps a strong reference until successful executor
   teardown. `build_reshard_params()` translates `MegatronTensorSpec` inputs
   into MX planner records.
4. `_M2nCall.from_param()` creates official `nccl.m2n.DistTensor` source and
   destination descriptors.
5. `_M2nRuntime.submit_model_update()` records producer readiness on current
   CUDA stream. Every source PP stream waits for that event.
6. One official `nccl.m2n.group()` records PP groups in canonical key order and
   preserves parameter order inside each group. Calls use
   `Handle.reshard(comm, src, dst, stream=pp_group.stream)`.
7. All destination tensors are received into MX-owned whole-version staging.
   Live parameters are updated only after every local PP stream finishes.
8. Caller invokes `NcclM2nExecutor.teardown()` to drain PP streams, release
   whole-version staging, and detach executor. Only then may
   `_M2nRuntime.close()` destroy single M2N handle while every parent
   communicator is valid, release streams, and destroy communicators in
   canonical PP-group order.

Callers may prepare update inputs concurrently. Each executor serializes
validation, staging, descriptor construction, and M2N submission. PP-group GPU
work uses distinct streams and may overlap after grouped submission.

Runtime shutdown first rejects new top-level operations, then waits up to
`finalize_timeout_s` for already-admitted executor operations and PP-group
creation to finish. An admitted operation may complete nested runtime calls
after shutdown enters `CLOSING`. After admitted operations finish, close
rejects attached executors before any native teardown mutation and restores its
prior state. Dropping the caller's executor reference cannot bypass this check
because runtime attachment is strong. Executor teardown is idempotent;
`execute()` is rejected after successful teardown.

Executor teardown and runtime close are cohort-wide orchestration steps. Every
participating rank must finish executor teardown/detachment before any rank
enters M2N-handle or communicator close. The local attached-executor check is
not a distributed barrier. Allowing one rank to start native close while a peer
still rejects locally can diverge collective teardown order.

If the admitted-operation wait expires, shutdown marks runtime poisoned and
intentionally retains M2N handle, streams, communicators, attached executors,
and staging. Further cleanup is unsafe; process restart is required.

## Failure and timeout semantics

MX polls PP groups in canonical order. Transfer completion requires both NCCL
success and a ready CUDA stream.

Defaults are `comm_init_timeout_s=120`, `transfer_timeout_s=900`, and
`finalize_timeout_s=300`. These bound MX-controlled waits and polling only. In
particular, `transfer_timeout_s` starts after `nccl.m2n.group()` returns;
Python cannot interrupt a native call blocked inside `group_end()`.
`finalize_timeout_s` also cannot interrupt a native `Handle.destroy()` or
another native finalize/destroy call that does not return. Production still
requires corresponding native timeout/bounded-wait support.

After a submitted transfer times out or reports an asynchronous error, MX
enters fail-stop: it poisons all PP groups, retains the M2N handle,
communicators, streams, tensors, and staging buffers, and starts best-effort
communicator abort on a daemon thread. The process cohort must restart;
`close()` does not reclaim quarantined resources.

After the shutdown drain succeeds and M2N-handle destruction begins, native
teardown is one-shot. Any handle destruction, stream release, communicator
finalize/wait/destroy, or final close-bookkeeping failure records the exact
phase and PP-group key, sets restart-only fail-stop, retains remaining
references and singleton ownership, and rejects every later operation.
Runtime does not launch communicator abort and does not retry native teardown
after partial progress: different ranks may have reached different PP groups,
so replay could create collective-order divergence. Canonical PP-group order
prevents healthy teardown cycles but cannot make rank-divergent partial
teardown replay-safe.

Production peer-failure safety therefore requires an M2N build with bounded
`m2nWaitCommReady`. Official main at review time (`5fab732a`) and revision
`abe83984` still wait indefinitely. The bounded implementation exists on the
official `port-m2n-async-wait-timeout` branch at `45c3f9b` and in the nccl-rl
fork at `5768c77`. Use a reviewed, revision-stamped artifact containing that
support.

## Caller contract

Call `execute()` only after weight-producing work is enqueued on current CUDA
stream. If producers use other streams, make current stream wait for them
first. Source tensors must remain allocated and unmodified until `execute()`
returns.

Every rank in one PP communicator must supply identical parameter count and
order. M2N calls are collectives.

Each `execute()` call must include every locally owned PP group exactly once.
Use an empty parameter sequence when one group has no parameters for that
update. Partial PP-group maps are rejected before M2N submission.

PP communicator membership is ordered: source ranks occupy
`[0, source_size)` and destination ranks follow. Within each side, communicator
rank must equal logical tensor shard index. MX validates NeMo
`local_shard_range` against this order before submission.

## Initial-slice boundaries and limitations

- MX owns PP communicators and streams. Caller-owned resources are deferred.
- Direct caller-supplied `DistTensor` descriptors are deferred.
- Dynamic PP membership requires a new runtime and communicator set.
- Any update failure leaves the executor unusable. Once grouped M2N submission
  begins, a transfer or commit failure also poisons the entire runtime and every
  PP group. Recovery requires restarting the process cohort; rebuilding objects
  in-process is unsupported.
- Whole-version staging protects one destination process. Serving must remain
  quiesced until the complete destination cohort reports success.
- Megatron shard ranges must be uniform, aligned, and evenly divide the global
  extent. Non-uniform or unaligned `local_shard_range` values are rejected.
- Cross-rank parameter-plan agreement is not implemented here. Different
  skipped parameters can still cause collective-order divergence.
- Source-only storage overlap across different M2N buckets is intentionally
  accepted for this integration. MX holds references and requires immutability
  until every stream drains. Current upstream M2N documentation describes
  cross-bucket storage overlap as unsupported, so production must pin a
  validated M2N revision and keep a multi-group GPU regression test. Overlap
  within one PP group and any cross-group destination overlap are rejected.
- Current integration requires NCCL 2.30.5 or newer and the current M2N Python
  APIs (`DistTensor`, `group`, and `Handle.reshard`).

Bootstrap transport/distribution belongs to ModelExpress control plane (PR
#516); this module consumes resulting unique-ID bytes and rank membership.
