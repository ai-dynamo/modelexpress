# NCCL M2N bootstrap through ModelExpress

Status: implemented locally as an unpublished follow-up to ModelExpress PR #497.

## Decision

Replace PR #497's Torch/Gloo UID broadcast with a dedicated ModelExpress
bootstrap service. NeMo RL/Dynamo remains topology authority; MX only brokers
one immutable NCCL UID record.

This first integration does not add `NcclM2nSession`, `BootstrapProvider`,
`CreateAttempt`, participant discovery, or rank assignment.

## Terminology and identity

| Term | Meaning | When it changes |
|---|---|---|
| `job_id` | Coordinator-defined training job or run containing one or more logical transfer groups. It is a namespace, not a communicator identity. | New training job or run. |
| `cohort_id` | Coordinator-defined identity of one logical source/destination group with a fixed ordered membership and NCCL rank mapping inside a job. It names the intended group, not one NCCL initialization. | Membership, role, source/destination partition, or NCCL rank mapping changes. |
| `attempt_id` | Globally unique UUIDv4 for one concrete attempt to create an NCCL communicator for a cohort. MX uses it as the storage and fencing identity. | Every bootstrap retry, restart, abort, expiry, or communicator rebuild, even if the cohort is unchanged. It is never reused. |
| roster | Ordered list of all participants in the cohort. Each entry identifies the participant, source/destination role, and assigned NCCL rank. Ordering is NCCL-rank order; source ranks precede destination ranks. | Frozen before an attempt begins. Any roster change requires a new `cohort_id` and `attempt_id`. |
| `roster_digest` | SHA-256 consistency token for the complete ordered roster. Every rank in an attempt receives identical digest bytes. | Changes whenever any roster entry or its order changes. |
| `config_digest` | SHA-256 consistency token for immutable M2N/NCCL settings not represented by the roster. | Changes whenever those settings change; applying changed configuration requires a new `attempt_id`. |

One cohort may have multiple sequential attempts. For example, if bootstrap
times out without changing membership, the coordinator keeps `job_id` and
`cohort_id`, generates a new `attempt_id`, and creates a fresh UID and
communicator. Replacing a participant or changing its NCCL rank creates a new
`cohort_id` as well as a new `attempt_id`. The coordinator must run only one
active attempt per cohort; MX does not enforce this cross-attempt invariant.

```text
job_id = training-run-42
  cohort_id = trainer-A-to-generator-A
    attempt_id = 550e...   # failed or expired communicator construction
    attempt_id = b71c...   # retry: same roster, new UID and communicator
  cohort_id = trainer-A-to-generators-A-plus-B
    attempt_id = 198a...   # changed roster, therefore new cohort and attempt
```

The roster itself is coordinator-owned and is not stored or reconstructed by
MX. The current API accepts `roster_digest` as opaque 32-byte data and checks
only its length and exact equality across publication/fetch. MX can detect
that ranks received different digests; it cannot prove that a digest describes
the real participants or rank mapping. A shared, versioned canonical roster
serializer and digest helper is required before independent coordinator
implementations can safely compute this value.

## Current and implemented flows

PR #497:

```text
launcher -> Torch/Gloo world -> rank 0 ncclGetUniqueId
         -> Gloo broadcast -> blocking ncclCommInitRank
```

Implemented flow:

[![Implemented NCCL M2N bootstrap flow](docs/images/nccl-m2n-bootstrap-flow.png)](docs/images/nccl-m2n-bootstrap-flow.svg)

[Open the full-size SVG](docs/images/nccl-m2n-bootstrap-flow.svg).

No rank may call an M2N collective before coordinator release.

## Assumptions

- NeMo RL/Dynamo knows exact source and destination participants.
- Coordinator freezes the ordered roster and immutable M2N/NCCL configuration
  for one attempt before UID generation.
- Source ranks occupy `[0, source_size)`; destination ranks follow.
- Coordinator assigns one process per GPU and permits one active attempt for
  each fixed cohort.
- Coordinator supplies a canonical random UUIDv4 `attempt_id`; it never reuses
  an attempt ID, including after failure or expiry.
- Source NCCL rank 0 is designated UID publisher; all ranks receive its
  participant identity.
- Coordinator invokes every worker, gathers local results, releases all on
  success, and terminates or restarts the cohort on failure.
- Communicator may span weight versions; weight version is not bootstrap
  identity.
- Initial deployment uses trusted MX control network. Authentication and TLS
  remain production hardening.
- Selected M2N build must support a nonblocking NCCL communicator. NCCL
  management calls made inside M2N must reach async success before M2N reads
  their outputs.

## Ownership

| Owner | Responsibility |
|---|---|
| NeMo RL/Dynamo | Roster, rank mapping, attempt ID, dispatch, all-rank result barrier, release/abort, restart |
| ModelExpress | Atomic UID publication/read, immutable metadata validation, abort tombstone, expiry |
| NCCL | Create predetermined communicator |
| NCCL M2N | Run predetermined reshard collectives on that communicator |

The `nccl/nccl-extensions` M2N API accepts an existing `ncclComm_t`; its
benchmarks use MPI only as external rendezvous. MX replaces that rendezvous,
not M2N communicator or topology ownership.

## Coordinator descriptor

Every rank receives:

```text
job_id
attempt_id
cohort_id
participant_id
uid_publisher_participant_id
assigned_nccl_rank
source_world_size
destination_world_size
roster_digest
config_digest
timeout
```

The coordinator sends the same `job_id`, `cohort_id`, `attempt_id`, ordered
roster digest, and configuration digest to every participant. It sends each
participant its own `participant_id` and assigned rank. Each rank validates
the fetched key, sizes, digests, publisher, UID length, and local rank
assignment before initializing NCCL. MX never discovers participants or
assigns ranks.

## ModelExpress API

Use dedicated `modelexpress_common/proto/m2n_bootstrap.proto` on existing MX
server and port. Do not extend `p2p.proto`, `WorkerMetadata`, or
`MxClientBase`.

```protobuf
service M2nBootstrapService {
  rpc PublishBootstrap(PublishM2nBootstrapRequest)
      returns (PublishM2nBootstrapResponse);
  rpc GetBootstrap(GetM2nBootstrapRequest)
      returns (GetM2nBootstrapResponse);
  rpc AbortBootstrap(AbortM2nBootstrapRequest)
      returns (AbortM2nBootstrapResponse);
}
```

States: `PUBLISHED`, `ABORTED`, `EXPIRED`. Publication is write-once;
retries with the same UID and immutable metadata are idempotent. Abort may
create a tombstone before
publication. Terminal attempts reject later publication.

## NCCL initialization and deadlines

Native shim compiles against selected `nccl.h` and exposes:

```text
ncclCommInitRankConfig
ncclCommGetAsyncError
ncclCommAbort
```

It sets `ncclConfig_t.blocking = 0`, resolves symbols through exact
`ctypes.CDLL` handles, verifies selected NCCL, M2N's NCCL dependency, and the
process-wide NCCL resolve to the same loaded object, and requires exact
header/runtime NCCL version match. Launcher must preload same NCCL used to
build M2N before Torch or any other NCCL consumer.

Polling and RPC timeouts provide cooperative deadline. `ncclCommAbort` itself
is synchronous and may hang in a broken runtime; a process supervisor is
required for hard wall-clock termination.

The communicator remains nonblocking after initialization. MX therefore polls
`ncclCommGetAsyncError` after `ncclCommWindowRegister`, even when registration
returns `ncclSuccess`, and does not consume the window handle until async
success. M2N must do the same for its internal NCCL management operations.
Revision `1623765eadf82b773f1debaa544f4c14d9fd6d80` needs an upstream fix to
poll completion of `ncclDevCommCreate`; without it, M2N reads an incomplete
device communicator and reshard fails. This is a merge/runtime prerequisite,
not an MX bootstrap responsibility.

## Storage

- Memory backend: tests and local development only.
- Redis backend: Lua scripts atomically publish/get/abort two same-slot keys.
  UID key expires at bootstrap deadline; record key keeps a 24-hour tombstone.
- Initial Redis claim assumes one non-failing primary. Production requires
  durable acknowledged writes and failover semantics that cannot lose fencing
  tombstones.
- Kubernetes backend is unsupported until it implements atomic
  `resourceVersion` compare-and-swap.

## Failure rules

- During communicator bootstrap, any valid-assignment failure calls
  `AbortBootstrap`; a non-null communicator also gets `ncclCommAbort` best-effort.
- Abort failures are attached to, never replace, root bootstrap error.
- Local NCCL success does not imply cohort success.
- Missing participant, MX abort/expiry, NCCL async error, or deadline failure
  fails whole cohort.
- Recovery always creates a new attempt ID, UID, and communicator. It preserves
  the cohort ID only when ordered membership, roles, and rank mapping remain
  unchanged.

## Implementation points

- Proto and generated bindings: `m2n_bootstrap.proto`.
- MX server: dedicated state manager, gRPC service, memory backend, Redis
  backend.
- Python control client: `modelexpress.m2n_bootstrap`.
- NCCL orchestration: `bootstrap_comm_from_mx`.
- Native ABI shim: `_nccl_bootstrap_ext.cpp`.
- PR #497 GPU driver: no `torch.distributed`; selected NCCL is preloaded and
  companion external test coordinator observes exact `ready-*` cohort.
  Coordinator release and worker failure contend to create one immutable
  attempt-scoped `decision` containing `release` or `abort:<reason>`; a worker
  tears down only when abort wins. Release is the gate commit point; its loser
  continues, while hard failures after release belong to NCCL and the process
  supervisor.
- NeMo RL/Dynamo wiring remains external to this repository.
- Initial GPU validation uses two source plus two destination ranks, matching
  the smallest configuration in the M2N test matrix.

## Out of scope

- Elastic membership or joining ranks to existing communicator
- Multiple overlapping pipeline-parallel cohorts
- Generic session/bootstrap-provider abstractions
- Graceful live communicator replacement
- Coordinator collective ordering

## Appendix: design choices and rationale

| Choice | Reason |
|---|---|
| Add `m2n_bootstrap.proto`; do not extend `p2p.proto` | P2P records describe model sources and worker metadata. Bootstrap attempts need different identity, expiry, immutability, and abort semantics. Separation avoids coupling to `MxClientBase`. |
| Keep topology ownership in NeMo RL/Dynamo | Existing coordinator already knows participants, roles, and ranks. Re-discovering them in MX would create competing authorities and a larger retry state machine. |
| Let source NCCL rank 0 generate and publish UID | NCCL requires one UID origin. Fixed publisher identity makes retries and metadata validation deterministic. |
| Use canonical random UUIDv4 per attempt | Globally unique, never-reused IDs fence stale UID records, retries, and delayed participants without central ID allocation. |
| Use MX instead of MPI or Torch/Gloo for OOB rendezvous | Disaggregated trainer/generator processes may not share a launcher process group. Reusing MX avoids a second always-on control plane; M2N only needs resulting `ncclComm_t`. |
| Initialize with `ncclCommInitRankConfig(blocking=0)` | Async polling exposes NCCL errors and permits cooperative deadlines. Blocking fallback would defeat timeout and abort handling. |
| Keep the communicator nonblocking through window setup and M2N | Creating a blocking child communicator adds lifecycle and topology complexity and weakens deadline handling. MX polls its window operation; M2N must poll its own internal NCCL operations. |
| Register 4 KiB-aligned symmetric windows with flag `0x01` | These are NCCL `NCCL_WIN_REQUIRED_ALIGNMENT` and `NCCL_WIN_COLL_SYMMETRIC` contracts. Flag `0x02` means strict ordering, not collective symmetry. |
| Require process supervisor for hard timeout | `ncclCommAbort` is synchronous and may hang after runtime failure. In-process polling cannot guarantee wall-clock termination. |
| Verify selected, M2N-linked, and process-wide NCCL identity | Loading two NCCL builds in one process can produce ABI mismatch or split communicator state. Exact header/runtime version match fails closed. |
| Store Redis record and UID under same hash slot | Lua can atomically publish, expire, and abort both keys on Redis Cluster. Short UID TTL limits capability exposure; longer tombstone TTL preserves fencing. |
| Keep memory backend test-only; mark Kubernetes unsupported | Memory lacks cross-process durability. Kubernetes backend has no atomic compare-and-swap implementation for this state contract yet. |
| Keep all-rank release outside bootstrap RPCs | MX brokers communicator material; coordinator owns cohort success and collective ordering. This keeps topology policy out of MX. |
| Serialize test release and abort through one immutable decision | Ready worker can still fail before release. Both outcomes must contend for one commit point; only abort winner tears down, while release winner proceeds. |
| Start on trusted control network | TLS and participant authorization are required before untrusted deployment, but do not change bootstrap state semantics. |
| Rebuild communicator for membership change | NCCL communicator membership is immutable. New or restarted generator ranks require new attempt ID, UID, and communicator. |
