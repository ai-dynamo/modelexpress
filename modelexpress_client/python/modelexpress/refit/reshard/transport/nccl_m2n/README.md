# NCCL M2N transport

This first integration supports NeMo/Megatron-style tensor inputs with
ModelExpress-owned PP transfer resources. It is the PR #497 NCCL M2N data
plane, not a one-sided `Transport.read()` implementation.

## Scope boundary

This module owns one process-level M2N runtime, M2N handle, PP-pair parent
communicators and streams, tensor-descriptor construction, canonical grouped
submission, destination staging, local safe-point apply, and native teardown.

An external ModelExpress control plane (the PR #516 follow-up) owns cohort
membership and generation, NCCL unique-ID distribution, communicator-rank
assignment, cross-rank parameter-plan agreement, version/round coordination,
serving fencing, stage/apply result aggregation, publication, and coordinated
cohort restart. The data plane consumes already-agreed bootstrap records and
update plans; it does not discover or distribute them. Gloo bootstrap in GPU
test runners is test scaffolding, not a production bootstrap path.

## Public API and ownership flow

1. The control plane supplies one `M2nPPGroupBootstrap` per locally owned PP
   pair. Its key is the globally stable
   `(trainer_stage, generator_stage)` pair.
2. `NcclM2nExecutor.create()` creates the process-owned `_M2nRuntime`, one M2N
   handle, and all local parent communicators in one canonical batch. It also
   creates one explicit CUDA stream per PP group. Production callers do not
   construct or close `_M2nRuntime` directly.
3. `build_reshard_params()` translates `MegatronTensorSpec` inputs into planner
   records. `stage()` requires a complete mapping for every local PP group,
   records producer readiness, creates official `nccl.m2n.DistTensor`
   descriptors, and submits one `nccl.m2n.group()` in canonical PP-group and
   parameter order.
4. `stage()` waits for local transfer completion and returns an opaque
   `M2nStagedUpdate`. Destination data remains in MX-owned whole-version
   staging; live weights have not changed.
5. After the external control plane aggregates successful staging across the
   destination cohort and establishes the serving safe point, every local rank
   calls `apply(update)`. Destination ranks copy staged data into live weights;
   source-only ranks advance the same token state without copying.
6. After externally coordinated apply success, call `release(update)` to drop
   the token's transfer references. The control plane may instead discard a
   successfully staged but unapplied update by calling `release(update)`.
7. At cohort shutdown, after all updates are released, every participating
   process calls `NcclM2nExecutor.close()`. It is the only public teardown
   entry point.

The intended successful lifecycle is:

```python
executor = NcclM2nExecutor.create(device_id, pp_group_bootstraps)
update = executor.stage(updates_by_pp_group)
# External cohort agreement and serving fence authorize local apply.
executor.apply(update)
# External cohort agreement authorizes publication.
executor.release(update)
# Later, at coordinated cohort shutdown:
executor.close()
```

Callers may prepare update inputs concurrently. Each executor serializes
validation, staging, descriptor construction, and M2N submission. PP-group GPU
work uses distinct streams and may overlap after grouped submission.

`NcclM2nExecutor.close()` is sequential and idempotent after success;
concurrent close attempts fail fast. It rejects close while a staged token is
pending, so callers must apply or discard and then release that token first.
Close rejects new work, waits up to `finalize_timeout_s` for admitted work,
drains every PP stream, destroys the single M2N handle while all parent
communicators remain valid, releases streams, and finalizes/destroys parent
communicators in canonical PP-group order.

Close is a cohort-wide orchestration step, not a distributed barrier. The
external control plane must ensure all participating ranks are ready before
any rank enters native teardown. If the bounded admitted-work wait expires,
or any native teardown step fails, close raises
`M2nCohortRestartRequired`, quarantines remaining resources, and requires
process-cohort restart. Native teardown is one-shot: do not retry `close()` in
that process after a fatal close failure.

One narrow retry is safe: if native teardown and the runtime close commit have
already completed, but releasing process-local singleton/executor ownership
raises, call `close()` again. That retry only clears Python ownership; it does
not replay M2N-handle or communicator teardown.

## Initialization assumption

MX assumes `nccl.m2n.init()` is failure-atomic when it raises: no native M2N
state requiring destruction remains. Once `init()` returns a handle, MX owns
that handle until successful destruction and does not recreate it in-process.
Failure after handle creation or after communicator initialization has started
may therefore enter restart-only fail-stop. A partially initialized native
runtime that is retained despite `init()` raising violates the required M2N
contract and cannot be repaired by this Python layer.

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
finalize/wait/destroy, or runtime-close-commit failure records the exact phase
and PP-group key, sets restart-only fail-stop, retains remaining references and
singleton ownership, and rejects every later operation. A later failure while
only clearing the already-closed runtime's Python singleton/owner bookkeeping
is different: retrying `close()` completes that bookkeeping without replaying
native teardown.
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

For development, install the optional `nccl-m2n` extra; it currently resolves
`nccl-extensions==0.1.0`. That version string does not uniquely identify an
M2N source revision. Production and peer-loss validation must instead use a
revision-stamped wheel or container built from `45c3f9b` (or a reviewed merged
successor with the same bounded-wait behavior), and should record the artifact
revision in build provenance.

## Caller contract

Call `stage()` only after weight-producing work is enqueued on current CUDA
stream. If producers use other streams, make current stream wait for them
first. Source tensors must remain allocated and unmodified until the staged
update is released.

Every rank in one PP communicator must supply identical parameter count and
order. M2N calls are collectives.

Each `stage()` call must include every locally owned PP group exactly once.
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
- Pre-submission validation failures leave live weights unchanged and do not
  poison the executor. Once grouped M2N submission begins, a transfer failure
  poisons the entire runtime and every PP group. An apply failure after live
  copying starts does the same. Recovery requires restarting the process
  cohort; rebuilding objects in-process is unsupported.
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
