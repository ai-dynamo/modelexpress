<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# RL refit architecture

This document describes the optional RL refit package and its server protocol, staged transfer flow, cache behavior, and failure handling. The base inference loader and P2P architecture are documented in [ModelExpress architecture](../ARCHITECTURE.md).

## Refit service and version model

### refit.proto - RefitService (Redis only)

`RefitService` is a new RL-specific control plane. It does not reuse or modify the
legacy `WeightSyncService`. The initial slice stores worker registrations,
immutable weight versions, and compact physical shard publications in Redis.
The control plane stores long-lived `TrainerMesh` membership as
`workers: worker_id -> TrainerTensorsMetadata(logical_shard_id, metadata_endpoint)`.
`bind_tensors()` computes address-independent coverage locally and returns
this compact reference. Adapters derive wire coverage without copying weights,
registering source buffers, or publishing a version. The adapter binding and
trainer `logical_shard_id` identify the same logical shard. Logical IDs hash canonical
coverage, so equivalent replicas share an ID. The orchestrator supplies the complete
trainer worker set and is responsible for logical-shard completeness. Mesh creation
resolves idempotent retries before contacting trainers and validates every binding
only for a new mesh. Mesh updates validate new workers and changed endpoints or
logical shards; unchanged members are not refetched. Large manifests never live
in Redis.
Admission first preflights worker registrations, then validates bindings with at
most 32 concurrent callbacks per request. Redis rechecks registrations atomically
when committing membership. Callback logs include the worker, endpoint, elapsed
time, and failures.
`CreateTrainerMesh` is idempotent; `UpdateTrainerMesh` replaces the complete
worker map with generation compare-and-swap while preserving logical coverage.
Every member requires an active model-matching trainer registration whose
`refit_endpoint` matches the binding endpoint. A worker-sharded `WeightVersion`
requires a trainer mesh; MX checks its existence and model atomically with version
creation. Object-storage versions omit the mesh. For worker-sharded versions,
`logical_shard_id` identifies a logical shard. MX accepts publications only from
workers in that shard's current membership and marks the version READY
when every logical shard has a current, live publication. Mesh-backed readiness
is computed from physical publications and their binding endpoints, not historical
coverage. Generator discovery reads the mesh
for complete expected shard coverage and filters out publications from replaced
workers and expired registrations. Generation changes invalidate cached transfer
plans; discovery rechecks the mesh generation after fetching worker manifests
to reject a snapshot that changed during resolution. Version leases protect
sources during installation. Publication release retries only the active-reader
lease rejection until its RPC deadline, retaining source buffers until deletion
succeeds. Mesh-backed trainers
can release published buffers before version retirement after all reader leases
drain. Removing a worker or rebinding its endpoint atomically retires its
publications and rejects the update while those versions have reader leases.
`ListWeightVersions` filters by model and optionally mesh, newest first. Mesh-filtered
listing uses the mesh's version index; metadata reads are pipelined.
`version_number` is optional caller correlation metadata, not version identity.
Object-storage callers manage version state; mesh callers do not manage readiness.

Mesh-backed `WeightVersion` records require a positive `trainer_mesh_generation`.
The unchanged creation API supplies a mesh ID; the server reads and stamps its
generation atomically while creating the version. Idempotent retries return the
original stamp. Shard publication, including the final READY transition and
repeated publications, rejects a changed mesh generation before writing. Trainer
resolution compares the recorded generation with the current mesh. Stored versions
require the generation field; decoding parses it as a uint64 without repeating
creation-time consistency checks. Versions without a trainer mesh, including object-storage
versions, carry generation zero. A requested version whose recorded mesh
generation differs from the current trainer mesh fails trainer source resolution.
The session tries configured source kinds in order, so this failure can fall back
to a generator peer serving the requested version. Bounded staging applies to
trainer transfers; peers retain their normal runtime-tensor preparation path.
An engine left uncertain by a failed installation can recover through a complete
peer tensor transfer even when the requested version was published as a delta.
The publication format does not determine the peer's transfer representation;
failed-handle and undrained-transfer protections still apply.
Trainer and peer paths check compatibility during plan selection without a
separate preflight pass. Object-storage replay retains preflight validation of
every revision before acquiring the chain's leases.

Any trainer mesh update that changes its generation is rejected while a linked
weight version has an active consumer lease, including additive replicas that
leave existing publications intact. An update with unchanged membership remains a no-op.
After readers release their leases, membership may advance and new weight
versions capture the new generation; old version stamps are never changed.

DIRECT installation and pipelined worker streaming are
not implemented by this control-plane slice.
NIXL manifest endpoints belong to their physical worker shards; a typed object
storage source belongs directly to its durable `WeightVersion`. The protocol
can identify S3, Azure Blob Storage, or GCS, while this initial implementation
accepts only S3. On the NIXL path, weight bytes and full tensor manifests remain
on trainer workers. On the S3 path, rank zero uploads the version's global index
after every trainer rank has uploaded its owned shard. `XOR_DELTA` versions use
compressed XOR shards; `FULL_HF_CHECKPOINT` versions use native Hugging Face
safetensor shards and omit `base_version_id`. Frameworks own
tensor gathering, Hugging Face conversion, and bucketization, then pass a lazy
canonical bucket stream to the trainer client, which owns delta processing,
full-checkpoint serialization, and S3 shard construction.
Full-checkpoint staging uses the same bounded worker-pool pattern as delta
staging to update the rank-local snapshot directly with immutable CPU tensors.
`publish()` groups that snapshot into safetensors objects capped by
`MX_REFIT_FULL_CHECKPOINT_BATCH_BYTES` (4 GiB by default) and uploads the
rank-local objects concurrently. A tensor larger than the cap occupies its own
object. The trainer does not materialize a temporary HF checkpoint on disk.
During framework initialization, the same bucket stream seeds only that rank's
owned seed-checkpoint tensors through direct, concurrent
`prepare_delta_base()` bucket reads. The first real delta stage uses the
prepared snapshot without checkpoint I/O.
Each canonical S3 version owns an exact caller-supplied `object_storage.uri`
under the configured `uri_prefix`. That URI names the global index; delta shards
or full-checkpoint safetensors objects are stored as siblings. A delta index's
`compression_format` selects the receiver's decompressor. The current
implementation supports Zstandard. Unknown compression formats are rejected
before shard download. Full versions use a standard safetensors index, carry
optional per-tensor Adler-32 checksums under tensor-name keys in shard metadata
when the index declares `checksum_format="adler32"`, and reset the exact base
for later deltas.
Canonical S3 versions are retained for rollout fault recovery. Trainer
processes keep only the current base snapshot; older
immutable S3 objects are governed by external bucket lifecycle policy.

#### RL refit client architecture

The RL framework orchestrator calls only the public
`ModelExpressControlClient`. Framework-native RPCs invoke rank-local trainer and
generator clients inside their existing actors. Those clients use internal MX
control-plane and worker APIs; tensor bytes never pass through `RefitService`.

```mermaid
flowchart LR
    subgraph Framework["RL framework"]
        O["RL orchestrator<br/>ModelExpressControlClient"]
        T["Trainer actor<br/>ModelExpressTrainerClient"]
        G["Generator actor<br/>ModelExpressGeneratorClient"]
    end

    S["MX server<br/>RefitService<br/>Redis metadata"]
    P["MX server<br/>P2pService"]

    subgraph Trainer["Trainer rank"]
        TR["TrainerRuntime"]
        TA["Explicit engine context<br/>Megatron or FSDP"]
        PM["Publication method<br/>full tensor NIXL or canonical checkpoint"]
        B["Registered source buffers"]
        M["RefitWorkerService<br/>metadata endpoint"]
        C["Canonical HF gather<br/>XOR staging"]
    end

    D["S3<br/>safetensors shards + index"]

    subgraph Generator["Generator rank"]
        GR["GeneratorRuntime<br/>source policy + session"]
        SR["Source resolvers<br/>generator, trainer, object storage"]
        UM["Update methods<br/>full tensor NIXL, canonical checkpoint"]
        I["Engine installer<br/>vLLM or SGLang"]
    end

    PG["Peer generator rank<br/>verified canonical buffers"]

    O -->|"Create / Get / Delete / update version state"| S
    O -->|"Framework-native RPC"| T
    O -->|"Framework-native RPC"| G
    T --> TR --> TA --> PM
    PM --> B
    PM --> C --> D
    T -->|"NIXL: register worker; publish shard metadata"| S
    T --> M
    G -->|"Register worker; lease version; NIXL: discover shards"| S
    G -->|"Discover exact-version same-rank peer"| P
    G --> GR --> SR --> UM --> I
    G -->|"Fetch exact-version manifest"| M
    B -->|"NIXL reads"| UM
    PG -->|"Publish READY version identity"| P
    PG -->|"Peer manifest + NIXL reads"| UM
```

`RefitService` is the central metadata service defined by `refit.proto`. It
coordinates immutable versions, worker registrations, shard advertisements,
and leases, but it does not discover engine tensor layouts or transfer weights.
For NIXL, `RefitWorkerService.GetTrainerShardMetadata` serves stable coverage,
geometry, addresses, registrations, and counts by digest. Coverage exists after
`bind_tensors()`; physical metadata follows buffer registration. Mesh creation
validates every binding, while updates validate only changed members.
`GetWeightVersionShardMetadata` binds a version and shard to that digest and its
ordered checksums. Generators verify identities, digests, and checksum coverage,
then cache stable metadata by mesh generation, shard, and worker. Each update
checks its publication and uses fresh checksums; disabled verification skips the
version metadata RPC. Redis pins each worker/shard digest for the generation and
rejects changes until a successful generation update.
The worker publishes stable metadata before optional version checksums and central
advertisement. It keeps one current physical record and a separate coverage
binding; replacement waits for all owned versions to retire. Version deletion
passes the lease fence before removing its checksum record, while stable metadata
remains available for reuse. The FSDP and Megatron builders and worker service
share `modelexpress_rl/train/manifest.py`.
Trainer source resolution retains per-update results, checksums, and timing in a
resolution round. Transfer preparation separates connection setup from plan
selection, and streaming posts batches through a dedicated method.
`build_sources()` constructs read-only mappings and frozen source/shard records
with tuple geometry and shard collections directly; both staging modes retain the
transport's
read descriptors directly instead of copying them into another descriptor type.
For S3, `WeightVersion.object_storage` identifies the storage type and global
`model.safetensors.index.json` URI directly; the server validates only this
typed location and does not contact S3.
Trainer integrations provide the matching `ObjectStorageConfig` through
`ModelExpressTrainerConfig.object_storage`; the initial implementation rejects
providers other than S3 before creating the storage client.

The synchronous generator client returns a staged handle after source-specific
preparation, without changing live weights. Trainer-source preparation receives
a full copy; generator-peer preparation reserves a read that runs during apply.
The handle owns the version lease through installation at the caller's safe
point. Applying the weights or releasing an unapplied handle ends the lease.
With bounded staging configured, the same handle holds deferred batch reads;
`apply_weight()` performs pipelined transfer and installation.

The initial worker manifest channel uses plaintext gRPC and does not authenticate
the publishing worker. Manifest digests detect corruption but do not establish
source identity. Deploy RL refit only on a trusted, network-isolated cluster and
prevent untrusted clients from reaching MX server and worker gRPC endpoints.
Transport authentication and TLS require a separate protocol and deployment
design; they are not provided by this implementation.

`GeneratorRuntime` composes three independent seams. Source resolvers discover
candidate locations without moving bytes; update methods prepare a typed update;
the engine installer commits it at the caller's safe point. Trainer and S3
methods transfer or reconstruct private artifacts during preparation. The
generator-peer method instead validates the peer and exact runtime tensor catalog
during preparation, then transfers directly into live storage at the safe point.
Source fallback and retry limits live in the planner/session rather than in
engine integrations.

- `nixl_staged_transfer.py` owns exact-manifest decoding, transfer planning,
  reusable registered buffers, NIXL reads, transforms, and digest verification.
- `inference/checkpoint_store.py` owns the host-local immutable lineage layout,
  locking, temporary-directory promotion, atomic JSON persistence, artifact
  fingerprints, and activation state.
- `inference/receiver.py` owns canonical S3 downloads, safetensors validation,
  and XOR reconstruction into derived checkpoints.
- `inference/engines/vllm/installer.py` owns vLLM load-layout capture and
  graph-safe installation through vLLM's layerwise reload and post-load path.
- `inference/engines/sglang/installer.py` reloads a prepared canonical checkpoint
  through SGLang's native safetensors loader.

Canonical preparation requests queue on a separate `.prepare.lock`. Once the
leader has reconstructed a target, followers verify and attach to that READY
checkpoint with shared installation and cache locks. They can therefore prepare
while another rank loads the same checkpoint. A cache miss releases the shared
locks, acquires the exclusive installation/cache fences, and rechecks the state
before reconstructing. Cache-hit attachment performs no eviction; capacity is
enforced by initialization, reconstruction, and activation.

| Canonical checkpoint operation | `.prepare.lock` | `.install.lock` | `.lock` |
|---|---|---|---|
| Attach to an already prepared target | Exclusive | Shared | Shared |
| Reconstruct a target | Exclusive | Exclusive | Exclusive |
| Load engine weights | None | Shared | Shared |
| Commit activation when the installation context exits | None | Shared | Exclusive |

The corresponding trainer composition is owned by `TrainerRuntime`. Public
`FSDPTrainerContext` and `MegatronTrainerContext` select only engine capture;
full-tensor NIXL and canonical-checkpoint object-storage publication remain separate
method implementations. This keeps transport, payload preparation, engine
geometry, and framework orchestration independently replaceable.

`ModelExpressGeneratorConfig.staging_buffer_bytes` sets bounded-streaming
capacity per arena, and `staging_buffers_count` defaults to one. The streaming
client uses their product as its total staging limit. `staging_device` selects
CUDA or pinned host receive memory.
The setting fixes the receiver's staging mode at construction; load-time
updates use one `prepare()` entry point, which selects full-copy or bounded
preparation from that fixed configuration. Both modes record planning and cache
reuse during preparation; streaming transfer bytes and durations are recorded
when installation exits, including failure paths.

The `stage_weight()` / `apply_weight()` generator APIs hold a version lease
across metadata preparation and incremental installation. The NIXL receiver
preserves recorded source-write order for vLLM and uses
one or two registered byte arenas (`staging_buffers`) on CUDA or pinned host
memory (`staging_device`). Receive tensors, wire-dtype conversion buffers,
full-source reconstruction buffers, and alignment all count toward each
buffer's capacity. With two arenas the READ for batch
`i + 1` is posted before batch `i` is yielded for commit, allowing asynchronous
READs to overlap installation; an arena is refilled only after its commit has
synchronized. Host arenas are registered as NIXL DRAM and read with a DRAM local
memory type. The vLLM installer copies each batch into engine-owned load-time
tensors, then uses native post-load processing to restore existing kernel
storage. Receive-arena views are never passed to engine callbacks. For trainer
sources without bounded staging configured, `stage_weight()` transfers a full
independent copy before installation.

The generator compiles one logical plan per mesh ID and generation from its
fixed installation plan and needed shard groups. Each leased update selects
replicas and binds fresh checksums, sessions, and addresses. A replica switch
rebinds physical sources; a mesh change recompiles logical reads.
Replica groups persist until the mesh changes; replicas are selected in worker-ID order.
The cache retains decoded metadata; the plan limits replica selection and agent connections
to the groups it needs.
`MX_REFIT_PREFETCH_REPLICA_METADATA=1` fetches all needed replicas up front;
otherwise metadata is fetched on selection. `MX_REFIT_DEBUG_VALIDATE_REPLICAS`
compares coverage and geometry across replicas. Failed native preparation
discards physical state while retaining the logical plan.

`MX_REFIT_PACK_MODULES` coalesces consecutive owning-module batches up to the
same staging limit, trading a larger arena residency for fewer of them. It never
changes which source bytes are read: the packed batch preserves source ranges,
byte counts and descriptor counts, while destination offsets follow the packed
arena layout. Modules that pull the same complete source stay in separate
batches. It is on by default; disabling it retains one owning module per batch.

Before vLLM rebuilds per-module load-time parameter skeletons, the adapter records
shared parameter objects and reconnects those aliases afterward. Alias owners
must remain at the same module paths during restoration and final processing;
replacement or removal fails the refit, including owners with no separately
published parameter and multiple paths to the same module. Capture then
counts tied weights once, and installation preserves their shared kernel storage.
Alias groups with inconsistent load-time shapes or dtypes are rejected.
Streaming owner checks resolve aliases against the live parameter objects on
each batch. An owner may reuse a shared parameter installed by an earlier batch
only while it still points to that exact committed object; a missing or rebound
parameter is rejected. Installation reconnects shared bindings after each
managed owner's kernel-storage restoration, before dependent hooks execute.

Released full-copy and bounded updates may alternate on one client. The transfer
owner tears down its agent before clearing the mode-specific buffers and reloads
source metadata into the new registrations. The update method invalidates cached
full-copy descriptors before entering bounded preparation. Preparation retries
retain one version lease and discard failed setup state; installation failures
still fence the engine rather than retrying a partially committed update.

Streaming is opt-in and trainer-only for supported vLLM models, including
quantized models. It does not publish generator peers or roll back partially installed
versions. An installation failure marks the client engine state uncertain and
makes the failed handle unusable for another installation. A subsequent
`stage_weight()` releases the failed transaction after proven cleanup, acquires
a new version lease, rediscovers sources, and prepares a complete full-tensor
replay. The engine stays uncertain until that fresh transaction installs
successfully. Unproven transport or GPU cleanup retains resources and leases
and requires a process restart. The hosting framework must keep all replicas
paused throughout recovery. Only completion of all replicas
permits resuming generation. The staging limit excludes live model weights,
engine-owned post-load workspace, CUDA allocator overhead, and transport metadata.

| RPC | Request | Response | Purpose |
|-----|---------|----------|---------|
| `RegisterWorker` | `RegisterWorkerRequest` | `RegisterWorkerResponse` | Register or refresh one TTL-bound worker process |
| `CreateWeightVersion` | `CreateWeightVersionRequest` | `CreateWeightVersionResponse` | Idempotently create one immutable `STAGING` or already-published `READY` version |
| `GetWeightVersion` | `GetWeightVersionRequest` | `GetWeightVersionResponse` | Read the version and its lifecycle state |
| `DeleteWeightVersion` | `DeleteWeightVersionRequest` | `DeleteWeightVersionResponse` | Cancel a `STAGING` version or move a `READY` version to `RELEASING` for retirement |
| `UpdateWeightVersionState` | `UpdateWeightVersionStateRequest` | `UpdateWeightVersionStateResponse` | Explicitly update lifecycle state, including S3 `STAGING` to `READY` |
| `CreateWeightVersionShard` | `CreateWeightVersionShardRequest` | `CreateWeightVersionShardResponse` | Publish one worker metadata reference for a required logical shard |
| `ListWeightVersionShards` | `ListWeightVersionShardsRequest` | `ListWeightVersionShardsResponse` | List the version's physical source publications |
| `DeleteWeightVersionShard` | `DeleteWeightVersionShardRequest` | `DeleteWeightVersionShardResponse` | Evict one source shard after release when no lease protects the version |
| `RegisterVersionLease` | `RegisterVersionLeaseRequest` | `RegisterVersionLeaseResponse` | Acquire or renew protection while installing a version |
| `DeleteVersionLease` | `DeleteVersionLeaseRequest` | `DeleteVersionLeaseResponse` | Release a generator's protection of the version shards |

`RefitWorkerService.GetTrainerShardMetadata` and
`RefitWorkerService.GetWeightVersionShardMetadata` are unary metadata RPCs; tensor
bytes remain on the advertised data-plane transport.

The final missing NIXL source slot atomically changes the version to `READY`.
For S3, the orchestrator marks the version `READY` after publication completes.

Peer generators reuse the existing inference P2P metadata service rather than
creating a second Refit peer registry. A generator serving an exact version
publishes its normal `SourceIdentity` with `revision=WeightVersion.uid`; another
generator queries `P2pService.ListSources` with the same engine-compatible
identity and selects a READY source for its worker rank before falling back to
trainer shard publications. Applied generators publish their complete post-load
runtime tensors, including registered runtime buffers. An identical-rank peer
validates that exact runtime representation before mutation, then pulls it
directly into the existing live tensor storage at the engine safe point. This
avoids a second model-sized GPU allocation and copy.
The peer holds a donor read lease through NIXL completion. Before the donor's
next active refit, it rejects new reads and waits for all leases to drain; a
drain timeout aborts before any live tensor is overwritten.
Trainer NIXL inputs remain a separate load-time representation that runs through
vLLM's normal post-weight-load processing.

The vLLM inference loader discovers this post-load tensor set once and retains
the device-local mapping. The RL runtime reuses that mapping for peer layout
validation, publication, and in-place installation instead of walking the model
again after warmup or compilation. It also borrows the loader-owned NIXL agent:
the already-registered live tensors are the peer receive destination, and the
loader retains responsibility for shutting down the rank-local transport. A
failure after the direct transfer starts leaves the engine state uncertain, so
MX fences it rather than attempting in-process source fallback. This path
requires a nonempty loader-owned runtime tensor mapping and an initialized
NIXL manager. Model quantization, FP8 KV-cache dtype, and execution mode do not
gate peer selection. With `model_config.enforce_eager=True`, after validating
the live destinations, the RL installer uses the same host-scale refresh as
cold RDMA loading: q/k/v Python scalars and CPU mirrors are updated from the
received accelerator tensors (using the maximum for per-head scales), with
tensor storage preserved. It also invalidates the
FlashInfer BMM and output-scale caches together so the next eager forward
recomputes them. It does not rerun post-weight-load processing on tensors that
are already in runtime format. Invalid scale contracts fail installation and
leave the mutated engine fenced.

Without `enforce_eager`, peer refits retain the existing direct-copy behavior
and do not run this host-scale refresh. Updating host mirrors alone cannot
update scalar values already captured in CUDA graphs; graph-scale refresh and
recapture remain outside this eager-mode fix.
Cold RDMA loading requires lazily populated attention caches to be uninitialized.
Q/K/V-only attention paths, including MiniMax-M3 sparse attention, do not need an
`_o_scale_float` field; that field is specific to output quantization. DeepSeek
V4/V4.1 FlashInfer instead recomputes its constructor-initialized BMM scalars from
the received custom Q/KV scale buffers. Its GPU scale buffers stay unchanged.
Recognized DeepSeek packed KV attention (`fp8_ds_mla` / `nvfp4_ds_mla`) carries
per-block scales in the cache format and needs no standard host mirrors. Each
`AttentionLayerBase` owner declaring FP8 KV cache must have a recognized scale
contract, even when another layer was refreshed or uses a supported packed format.
Container modules and `MambaBase` state-space layers are not attention scale owners.

An object-storage generator with full-tensor engine support defaults to a
same-rank generator peer first and the version-level object-storage source
second. Engines without that support use object storage directly. An explicit
`source_order` may select object storage only or combine `GENERATOR` and
`OBJECT_STORAGE` in either order. A memory-backed generator defaults to a peer
first and trainer manifests second. If the rank-local NIXL transport cannot be
initialized, an object-storage generator drops the peer attempt and remains
available with object storage as its only source.

For an active refit, the peer lookup is for the exact target UID. If no peer can
prepare that version, the generator resolves the full canonical lineage from
its `FULL_HF_CHECKPOINT` root through the target deltas. Under the local cache
lock, the receiver reuses its verified checkpoint when that version is on the
target lineage, downloading and applying only the missing revisions before
installing the target. Full-root reconstruction is retained when no matching,
source-verified local checkpoint exists, including an unverified launch seed.
A successful peer refit does not rebuild the canonical checkpoint in the
background, so the local checkpoint may lag the engine's serving version.
Fallback therefore resumes from the local checkpoint version. Object storage
is consulted only when a later refit cannot use P2P.

The canonical receiver retains each full checkpoint and delta payload under its
version, then writes a resolved chain manifest. A full target is directly
installable. The first delta after a full checkpoint copies that immutable full
checkpoint into a version-scoped derived checkpoint. On Linux the checkpoint
store attempts `FICLONE` per regular file, sharing blocks copy-on-write where
the filesystem supports reflinks. Unsupported filesystems, cross-mount copies,
and other platforms use an ordinary copy. This includes `EBADF` from `FICLONE`
on already-opened files when the source filesystem cannot reflink.
File metadata and symlink dereferencing
retain `copytree`/`copy2` behavior; mutable files never use hard links. Clone I/O,
permission, and capacity failures abort preparation and clean its temporary
directory before promotion. Later sequential deltas
rename the active materialization and apply only the incoming XOR delta in
place, avoiding another full-model copy. Canonical artifacts are never modified
during reconstruction, and derived checkpoints can be rebuilt from the lineage.
If an in-place delta fails, the cache remains `UPDATING` until the active refit
session or the next initialization restores the immutable full root. The running
engine retains its previously installed weights.

The local checkpoint store caps its configured quota at the existing model cache
size plus free disk space. It logs any cap applied at initialization and rechecks
free space before known writes and copies. Capacity checks evict stale
checkpoints or reject the update while preserving protected lineage.
Reflinks retain logical-size accounting and the full first-materialization
free-space check: delta writes can allocate private blocks. This is an admission
check, not a filesystem reservation against concurrent users. Reflink support
must be tested on the actual checkpoint cache mount; a Kubernetes `emptyDir`
does not establish support. First-refit latency savings require a fresh benchmark
on that mount; existing sequential-reuse timings do not measure this optimization.

Under the local checkpoint lock, preparation state advances from `READY` to
`UPDATING` before artifact construction and back to `READY(target)` only after
verification. `active.json` is a separate commit point: it advances only after
the engine installer returns successfully. Download, reconstruction, and install
failures therefore retain the previously active engine version and immutable
lineage. A separate installation fence is shared by co-located installers and
exclusive to preparation, so preparation cannot enter between engine reload and
activation.

RL cold-start bootstrap populates only the immutable full-checkpoint cache; it
does not rewrite preparation or activation state. Generator ranks sharing that
cache use local rank 0 as the preparer. Global rank 0 first broadcasts one
immutable desired UID, and every rank verifies its local configuration against
that pinned value before agreeing on the replay chain. Node-local followers attach to the
verified `READY` checkpoint. A CPU-process-group barrier keeps every rank out of
the engine's distributed checkpoint loader until all nodes are prepared. The
cache activation commit runs only after every engine rank reports a successful
load; configuration drift, phase disagreement, or a partial load fails the
cold start without advancing `active.json`.

Startup P2P does not populate the canonical disk checkpoint. If hotload starts
with a missing or unrecorded cached seed and no preparation state, the S3 method
defers initialization until object-storage fallback is needed. It resolves the
full replay lineage and downloads a source-verified full root before applying
deltas. This does not activate the disk checkpoint or change the serving UID;
activation still follows successful engine installation. Recorded artifacts
continue to undergo integrity verification. The bootstrap requirement is checked
on every replay request. Once initialized, an object-storage-only runtime can
resolve from its serving version instead of revisiting the full lineage. Mixed
P2P/object-storage runtimes still resolve from the full root because their disk
checkpoint may lag the serving version.

The engine's serving version remains separate from checkpoint-cache state.
vLLM constructs its Control server only after EngineCore and its workers finish
loading. Because an explicit desired UID makes the MX loader fail closed, the
Control endpoint is reachable only after every rank loads that UID. The MX
startup probe then calls vLLM's native `UpdateWeightVersion`, reads it back with
`GetWeightVersion`, and succeeds only on an exact match. Dynamo's main sidecar
therefore cannot start or admit the worker before vLLM reports the loaded UID.
When the weight-transfer backend is initialized later, the controller passes
that observed UID as `initial_serving_version_id`; the canonical
`initial_base_version_id` continues to describe only replay lineage.

`WeightVersion.uid` is MX's opaque version identity. A create request may supply
the UID; MX generates one when it is omitted. Creating another version with an
already-used caller-supplied UID returns `ALREADY_EXISTS`. For an `XOR_DELTA`,
`base_version_id` identifies the base version's UID. The canonical index records
these UID strings as `metadata.version` and `metadata.base_version`. MX assigns
no numeric ordering and requires no version-directory naming convention; the
exact `object_storage.uri` identifies the version's global index.
Generators use the registered MX IDs and base relationships for replay. A delta
index's `metadata.version` and `metadata.base_version` are optional descriptive
fields and may differ from the MX IDs. Cache paths and chain records use MX IDs;
MX lineage checks and payload validation remain enabled. The registering caller
is responsible for selecting the correct S3 artifacts and base mapping.
`WeightVersionShard` identifies a per-worker version publication.
Its identity is `(version_id, worker_id, logical_shard_id)`: `logical_shard_id`
identifies the logical shard covered by the publication, and
`worker_id` identifies the publishing process. The trainer engine adapter
derives the logical shard ID from its native topology; the Megatron adapter uses
the logical tensor names and shard geometry, excluding physical process and DP
replica identity. The orchestrator deduplicates those adapter-defined logical shards when
declaring the version's expected contributions. Multiple DP workers may therefore
advertise the same logical shard; generators rotate through those publications on
transfer retry. Each shard carries `metadata_endpoint`, `stable_metadata_digest`, and an optional
`version_metadata_digest`. Tensor counts and byte totals live in stable metadata,
where they are computed once when the physical metadata is initialized.
Deployments configured with
Kubernetes or the test-only memory backend do not expose `RefitService` yet.

`RegisterWorker` is also the heartbeat API. `worker_id` is a fresh process
incarnation identity, so a restarted worker registers a new ID instead of
fencing an old registration. The registration expires with its TTL; source
planning and source discovery will use that liveness when selecting a
publication.

Cross-key compare-and-set operations live in documented Redis scripts
under `modelexpress_server/src/refit/backend/redis/scripts/`. Redis executes
each script atomically. The `refit_service_redis` test drives the public gRPC
API through two server replicas and covers concurrent version creation,
concurrent final source-slot publication, idempotent and conflicting replay,
replacement-source publication, lease-protected release, and safe shard
eviction. CI runs this test against Redis 7.

A future Kubernetes backend can preserve this gRPC contract, but it cannot
translate each Redis key into an independent CRD and retain the same atomic
guarantees. Lease creation and shard deletion race across resources. The
Kubernetes implementation therefore needs one version-scoped coordination
object as the serialization boundary, updated with `resourceVersion`
compare-and-swap; shard and lease objects can remain child records.
`RefitService` depends on the domain-level `RefitBackend` contract, with Redis
implemented under `refit/backend/redis.rs`. Kubernetes remains a separate
end-to-end backend slice implementing the same transaction boundaries.

## Python engine integration

### vLLM Refit Installation

The vLLM integration exposes engine installation and tensor geometry to
`GeneratorRuntime`; it does not select a transport or source. Runtime composition
uses separate NIXL methods for trainer load-time tensors and generator runtime
tensors, and adds the canonical-checkpoint method when object storage is
configured. XOR deltas mutate an exact base; full HF checkpoints stream tensors
into the existing mmap-backed checkpoint and become the base for later deltas.
All methods feed the shared private installer, which commits staged tensors or
reloads a prepared checkpoint through vLLM's graph-safe layerwise reload path.
Before a reload, the installer restores registered slots for live kernel buffers
that were created after vLLM recorded its load-time metadata. PWAL can then
replace those buffers without turning them into conflicting plain attributes,
and vLLM's normal finalizer copies them back into their original graph-bound
storage.

The ModelExpress vLLM plugin registers one `modelexpress` native weight-transfer
backend for both paths. An empty initialization payload preserves the existing
NIXL path. The payload can override the logical ModelExpress model name when it
differs from vLLM's model path. Supplying `object_storage_type` with the READY
seed-base version ID, seed checkpoint path, and refit checkpoint directory
selects the object-storage path; the initial implementation accepts only `S3`. The payload
can also override the MX server through `server_url` and the storage connection
through `object_storage_endpoint_url` and `object_storage_region_name`. Each
update carries the opaque MX `version_id`.
`start_weight_update()` opens the update window, `receive_weights()` stages and
applies the version through `ModelExpressGeneratorClient`, and
`finish_weight_update()` releases its staged handle. Each session accepts one
version; calls before initialization, out-of-order calls, and a second update
raise instead of reporting success. vLLM's reset hook releases staged resources
and clears the session after an update error, including invalid payloads rejected
before receiving weights. A failed finish retains the session so cleanup can be
retried; already applied model weights are not rolled back. Draft-model updates remain
unsupported, and trainer-pushed bytes are ignored because trainers publish
WeightVersions through `ModelExpressTrainerClient`. After a successful apply,
the bridge merges staging
and installation metrics and logs each numeric `perf/` phase as a separate INFO
line in deterministic key order. `perf/mx_receive_stage_weight_time` covers the
complete `ModelExpressGeneratorClient.stage_weight()` wall-clock call.

The active vLLM refit installer lives in
`modelexpress_rl.inference.engines.vllm.installer`. It captures canonical
load-time destinations and aliases once for the generator client's lifetime,
then installs updates into the live model. Checkpoint reloads use vLLM's
layerwise loader and post-load processing so quantized and graph-bound weights
retain their engine-owned behavior. Replacing the engine or changing its loader
semantics requires a new client.

RL refit timing and transfer planning live under `modelexpress_rl.refit`.
The RL-specific `MX_REFIT_*` and `MX_RESHARD_*` settings are registered in
`modelexpress_rl/envs.py`; shared client configuration remains in
`modelexpress.envs`.

### No-gather Refit Resharding

`modelexpress_rl.refit.reshard` captures the slices and destination views selected
by an engine's own weight loader, intersects them with trainer-published
shards, and compiles one-sided read descriptors without materializing a full
trainer tensor. Geometry, slice planning, transfer planning, and the transport
protocol are engine-agnostic.

Geometry capture passes all lazy source weights through the engine loader in
one call, amortizing model-wide parameter and expert-map construction. Each
lazy tensor retains its original source identity, including native-to-HF view
conversions. If an unsupported operation interrupts the bulk attempt, the
recorder rolls every accumulator it advanced back to the attempt's starting
state before retrying each source individually for precise diagnostics, so bulk
capture still never produces an incorrect partial plan. The bulk cause is logged
even when the per-source retry then succeeds for every source, because bulk order
can trip a loader that per-source order does not. Unexpected engine errors
propagate, and parameter loader hooks are restored on every exit. The RL installer
retains its first successful canonical capture for the generator-client lifetime;
transfer planning composes trainer source views with that fixed capture.

The minimal rendezvous publisher is called only after its NIXL agent and source
buffers are registered, so `publish()` stores the worker as READY and repeated
publication replaces that worker record. The shared `PublisherThread` sends a
READY status heartbeat every `MX_HEARTBEAT_INTERVAL_SECS` seconds (30 by
default), keeping the record fresh for READY-only discovery. `close()` uses the
same publisher lifecycle to stop the heartbeat and best-effort mark the source
STALE. Long-lived framework integrations still need to call `close()` from
their lifecycle; SIGKILL and mid-transfer failure recovery remain follow-up
work.

`modelexpress_rl/inference/nixl_staged_transfer.py` owns exact-manifest planning,
trainer staging buffers, direct generator-peer transfer, and verification. The
vLLM-specific
`installer.py` captures canonical engine load-time geometry through layerwise reload and
installs it through `process_weights_after_loading`. For tensor and checkpoint
reloads, vLLM owns quantization-aware refresh of MLA's `W_UV` and `W_UK_T`;
ModelExpress preserves their graph-bound storage when post-load processing
replaces these bare tensor attributes. The RL installer does not recompute MLA
weights from `kv_b_proj.weight` or reject quantized MLA models. After reload,
replacement tensors are copied back into the original graph-bound storage.
The engine's loader semantics remain fixed for the client lifetime; a failed
refit requires engine recovery before serving resumes.
Generator peers instead
transfer the complete post-load runtime tensor set directly into existing live
storage without re-running PWAL. The adapter rebuilds a trainer plan when
validated source manifests change; an incompatible destination staging layout
requires an engine restart.

See the [RL refit package](../../modelexpress_client/python/modelexpress_rl/refit/__init__.py)
for its public timing and resharding exports. The deployment behavior and
validation details are described in this architecture section and the
[deployment reference](../DEPLOYMENT.md).

## Staged transfer, caching, and failure handling

### Bounded vLLM streaming installation

The generator streams captured source writes through reusable receive arenas in
loader order, including writes that revisit an earlier owner. Each source write
must fit one arena; preparation rejects oversized writes before connecting sources.
Streaming uses vLLM's reload lifecycle for supported models, including
quantized models. Quantized MoE graphs captured between refits can retain stale
scale references; use eager execution or startup-captured graphs.

```mermaid
flowchart LR
    T[Trainer shards in HBM or DRAM] -->|NIXL READ| A[Bounded receive arenas]
    A -->|Replay captured writes in order| E[Engine-owned load-time tensors]
    E --> P[vLLM post-load processing]
    P --> L[Original live parameter storage]
    L --> F[vLLM deferred attention finalization]
    G[Framework update guard] -. held through transfer and installation .-> A
    G -.-> F
```

The installer opens one reload window per update. It materializes the engine's
load-time destinations and replays captured views, casts and copies into them.
An owner is processed only after its last write and overlapping-owner dependencies
are complete. vLLM
performs post-load processing and restores the original kernel tensor bindings;
finalization handles attention dependencies after all inputs arrive. Native
materialization selects its recorded device, while host bookkeeping can be
recreated on CPU. Bare tensors retained by graph consumers keep their storage.
After finalization, the meta-device guard checks runtime parameter names, which
can differ from load-time names after quantization post-processing.

Receive-arena tensors are never attached to the model or passed to post-load
callbacks. The copy boundary requires ordinary Tensor storage views and rejects
custom dispatch modes. A callback may retain an engine-owned tensor without
preventing arena reuse; CUDA completion is required before another READ overwrites
an arena. Installation retains aliases needed for tied parameters and runtime
bindings, but removes the separate alias-lookup, local-materialization and custom
assignment paths. The fixed installation plan resolves current native targets.

The vLLM installer lazily creates one canonical installation plan on the first
successful capture for a generator client. Captured loader operations, destination
geometry, owning module groups, alias restoration metadata and graph destinations
belong to that plan. Construction validates complete capture and restores the
live model before publishing it. Unrecorded writes to persistent destination
storage are rejected before mutation. Missing parameter coverage also fails
capture. Failed construction leaves the client without a plan so a later attempt
can retry.

The plan is reused for every update without source-schema keys, independent
mapping caches, deep copies or comparisons against current model parameters.
Architecture, canonical loader-input schema, module ownership, alias topology and
loader semantics are fixed for
the client lifetime; changing them requires a new client. Execution resolves the
current leaf tensors after native materialization. It checks actual stream
completeness and preserves copy, completion and cleanup safety. It does not detect unsupported structural mutations by callbacks.

Trainer-native conversion and composition of source view operations with the
canonical capture belong to transfer planning. The first transfer supplies the
canonical loader-input schema needed for lazy capture; the installer retains no
trainer identity, source conversion or source-bound mapping. Transfer consumers
must not mutate the retained canonical capture or destination metadata.

`modelexpress_rl.refit.reshard.plan_transfer` builds an immutable
`LogicalTransferPlan` directly from captured geometry. Exact reads, conversions
and full-source pulls index a shared source-buffer table and carry shard-relative
byte offsets. The table identifies logical source slots and tensor shards; it
contains no addresses or transport handles. Operation capture normalizes view
arguments once; `RecordedCopy` is a plain immutable record. Direct planner callers
supply immutable tuple shapes, strides and operation arguments. Capture rejects
advanced list/tensor indexing; basic integer,
slice, ellipsis and new-axis indexing remain supported. Grouped plans inherit
the complete plan’s full-source verification policy.
`plan_transfer_groups` derives module plans from a complete
plan, keeping the shared source table and its index map inside the planner.
Module batching and packing preserve the same read order for legacy captures.
Source replay instead batches consecutive writes without regrouping by owner;
module packing does not reorder those writes. Each write stages the exact
representable source-view prefix and locally replays the remaining operations.
Digest verification stages the complete source and checks the current version
checksums before replay. When an exact prefix exceeds
`MX_RESHARD_MAX_SEGMENTS_PER_COPY`, the planner considers contiguous earlier
prefixes and replays their remaining operations locally. It chooses the least
wire overhead that meets the descriptor target, or the greatest available
reduction when fragmentation prevents meeting it. Candidates must fit the aligned
per-arena capacity; if none do, exact reads remain. Arithmetic cost counting
avoids constructing discarded descriptor lists. Preparation reports exact and
planned descriptor counts, descriptor savings and extra wire bytes, including on
warm updates. Whole-source checksum verification retains its complete reads.

`bind_transfer(plan, resolve_source, resolve_destination)` returns a separate
`PhysicalTransferPlan`. It resolves each used source buffer and each named
receive, conversion or full-source destination once. The staged and legacy
receivers both use this binder. The staged receiver explicitly associates each
selected source slot with the agent parsed from its corresponding manifest;
metadata, leases, transport connections and staging allocation remain outside
the shared planner. Equivalent replicas can bind the same logical reads to
current sessions and addresses without rebuilding geometry.

The generator's installation plan owns its canonical capture and destination
layout for the client's lifetime. Transfer preparation composes source views
with that capture once per mesh generation; unchanged mesh identity and
generation reuse the resulting logical reads without repeating source
conversion. Physical session and address bindings remain outside the logical
plan. Validation runs when each layout is established, and warm reuse trusts
the retained state. Per-update readiness checks, checksums, stream completeness
and resource-safety checks remain in place.

Alias restoration distinguishes repeated paths to one slot from ties between
slots. vLLM's recorded parameter metadata separates load-time ties from runtime
aliases such as WNA16's `w13_weight` and `w2_weight`; native post-processing
recreates runtime aliases. Supported quantized full-copy and bounded streaming
refits retain the same fixed-client contract.

The staging limit bounds registered receive arenas. Replay operations can require
additional local scratch, reported as `replay_scratch_peak_bytes`. Estimated
installation owner and materialization peaks are also reported separately.
Engine-owned materialized destinations and post-load workspaces consume
additional memory, as do the live model and runtime state. The staging limit is
therefore not a total GPU-memory bound. Two arenas can overlap the next READ with
installation of the current group; the benefit must be measured on the target
hardware. Trainer staging mode and receiver arena location are separate choices.

The update plan holds one descriptor-cache tuple: one entry for full-copy mode,
or one per bounded batch. Each entry caches READ descriptors by source group and
replica, populated on selection. Switching one group builds only missing replica descriptors;
returning to a cached replica reuses them in the original read order. Bounded
batches recreate views over the staging arenas during execution. Each batch
zeroes padding and posts fresh READs.
Mesh changes and destination teardown clear execution caches. Failed attempts
invalidate the active binding while retaining descriptors if destination
resources remain valid; existing drain and recovery rules still apply.
Descriptor setup stays outside wire timing.

Trainer discovery selects replicas only for needed shard groups.
Replica fingerprints reuse parsed sources without rescanning reads or batches;
snapshots hold no tensors, handles, checksums, or leases. Coverage and capacity
are checked at compilation. Discovery reports cache, fetch, verification, and
stable RPC timing per candidate, excluding time spent in transfer or install.

The prepared streaming artifact owns its iterator and remains protected by the
version lease. An installation failure fences the engine and never falls back
after a possible write. CUDA work must drain before iterator cleanup and source
release. A failed drain, iterator close or source iteration retains the active
handle, receive workspace and source lease and requires a process restart: CUDA
synchronization alone cannot prove failed RDMA reads have completed. Simultaneous
installation and cleanup errors preserve the original exception with the cleanup
failure chained as its cause. Partial installation has no rollback.

The adapter depends on vLLM's layerwise reload helpers. Runtime compatibility must
be qualified with changing weights, shared parameters, graph-bound addresses,
post-load state and failure cleanup. Quantized bounded installation is supported
for qualified vLLM models. The same generic path is used for small-model
validation and GLM; passing the former does not establish full-model correctness
or performance.


Full-copy and bounded updates enter one session staging flow. Immutable streaming
settings bind the mode, placement, buffer count and total budget at runtime
construction; each transfer retains that configuration for its lifetime. Public
`staging_buffer_bytes` remains the capacity of each buffer, multiplied by the
buffer count for the internal total budget. Bounded device compatibility is
checked before creating its NIXL manager. Streaming installer support is checked
once during runtime initialization, before allocating transfers. Runtime
construction selects the trainer method; per-update candidate selection does not
repeat method-type or streaming-installer checks. Generator peer reads retain a separate
full-copy transfer borrowing the engine manager. The version lease remains held through
installation or release. Preparation retry and recovery behavior stays scoped
to the selected update strategy; failures after a possible write fence the engine.

The immutable update plan combines the fixed engine layout, trainer conversion,
and source-relative logical reads. It is reused for the mesh ID and generation
in both transfer modes. Physical bindings and transports follow the selected
replica fingerprint; checksums, leases, and preparation metrics are per update.
Full-copy updates with an unchanged physical binding skip workspace layout
checks; bounded capacity is validated during compilation. Optional
`MX_REFIT_DEBUG_VALIDATE_PLAN` rechecks cached plan coverage.
`prepare()` handles lookup, binding, connection, and cleanup, then publishes
only after setup succeeds. Full-copy and bounded modes manage their own
workspaces and descriptors. Failed preparation or READ invalidates physical
state while retaining the logical plan; outstanding reads must still drain
before native resources can be reused.

The transfer owns cleanup of partial native preparation. Metadata, layout and
compilation failures leave registered storage and existing connections intact.
A failed preparation that attempted manager initialization, source connection or
registration resets an owned manager before releasing its storage; cleanup failure
remains fatal with its cause retained. Borrowed managers cannot be reset by the
transfer. Streaming READ drain, retained leases and uncertain-resource quarantine
keep their existing installation behavior.

## RL refit CI harness

`ci/bench/` shares one Kubernetes lifecycle and pause/refit/verify/resume protocol
between manual runs and the reusable `.github/workflows/rl-refit-ci.yml` workflow.
The comment-triggered `bench-ci.yml` authorizes requests and selects inputs; the
reusable workflow independently verifies the caller, approved PR revision, model,
scenario and runtime before building images.
The trusted-PR GPU workflow also calls it automatically for Nemotron S3 delta refit,
using the approved mirrored harness revision. Its result, including cleanup, gates
the required `CI status check`; Kimi remains comment-triggered. Required Nemotron
Both automatic and comment-triggered CI reuse the Vime example's SeaweedFS Deployment
and Service in an isolated namespace, then seed the pinned snapshot before timing.
`scripts/ci.py` owns namespace ownership, GPU quotas, storage and image credentials. `harness/lifecycle.py` owns
workload execution, evidence collection and cleanup for both entry points.

`harness/scenario.py` selects the registered scenario. The implemented delta/S3
scenario in `scenarios/delta/scenario.py` configures preparation modules,
publication location, initial/target versions and explicit worker sources. The
`DeltaScenario` instance owns checkpoint, installed tensor and inventory validation.
`RefitRunner` holds the run configuration, HTTP endpoints and evidence directory
without executing the protocol at import time. The shared driver does not derive
versions or dispatch sources from role names. Another refit scenario can supply
its own preparation and verification without changing the common RL protocol;
no reshard scenario or CI case is registered yet.

The worker extension exposes direct RPCs for initialization, refit, tensor hashes,
host scales, checkpoint verification and installed runtime tensor verification.
Installed embedding verification uses vLLM's vocabulary-parallel weight loader
on a separate CPU parameter, preserving its sharding, padding and dtype rules.
Unsupported packed embeddings fail verification. Validation remains outside
transfer and installation timing.

Cleanup stops all benchmark producers before deleting the exact run's S3 artifact
prefix, including incomplete publications. A separate CPU pod performs cleanup,
so failed or expired benchmark pods are not required for recovery. CI deletes the
owned namespace only after cleanup is verified; a failed attempt releases GPUs
and preserves namespace credentials for retry.

Runtime code is organized into `harness/` for shared orchestration and validation,
`scenarios/delta/` for seed download, delta publication and delta-specific checks,
and `engines/vllm/` for the vLLM server and worker extension. `scripts/` contains
CLI entry points and CI authorization/provisioning. The renderer bundles shared
runtime code, the selected scenario and the vLLM adapter; host rendering and
Kubernetes lifecycle code are not mounted in workload pods. ConfigMap item paths
preserve Python packages, and pod commands use `python3 -m` with ordinary package
imports. Configuration is loaded explicitly when a command starts.


## RL cold-start deployment

For RL cold start and active weight refit, see
[`../../examples/rl/dynamo_vllm_reshard_refit/README.md`](../../examples/rl/dynamo_vllm_reshard_refit/README.md).
Its vLLM startup probe directly reconciles the desired MX UID through vLLM's
native Control gRPC service. Keep that probe on the restartable vLLM init
container: Kubernetes does not start the Dynamo sidecar until vLLM reports the
desired UID, so a failed or mismatched reconciliation cannot admit the worker.


## Generator peer lease behavior

RL generator-to-generator active refit calls `PrepareTensorRead` while preparing the update, then holds the bounded lease until the staged update is released or the NIXL transfer completes. These donor-timed leases protect the donor's live tensor storage while another generator reads it; the donor drains existing readers before overwriting that storage, and if the drain times out, refit fails before mutation. Completion and ordinary failures release immediately. A pre-transfer retry obtains a fresh lease and revalidates the exact source version and worker instead of reusing the consumed manifest. If the target pod exits without cleanup because of `SIGKILL`, OOM, or node loss, the donor expires the abandoned lease after its bounded lifetime. Ordinary inference loading continues to use `GetTensorManifest` without this RL-specific mutation fence. For file-backed cache artifacts, targets call `PrepareArtifactChunk` and `ReleaseArtifactChunk` while bytes move through NIXL into target-local staging, then install the staged artifact into the runtime cache directory. The division of responsibility depends on which metadata backend is in use:

Active refit of a generator that publishes runtime tensors requires this default P2P metadata path. With `MX_P2P_METADATA=0`, ModelExpress cannot account for readers of tensor addresses stored in the central coordinator, so it fails the refit before mutation instead of risking a mixed-version transfer.

Generator-to-generator active refit validates the peer first, then receives
directly into the target's registered live runtime tensors at the engine safe
point. It does not allocate a second model-sized GPU buffer. If that direct
transfer fails after mutation starts, ModelExpress marks the engine state
uncertain; reset or restart the worker before serving or retrying. Quantized
models and FP8 KV-cache configurations use canonical object-storage refit because
their non-tensor derived state cannot currently be refreshed by a direct copy.


## RL protocol trust and upgrades

RL refit has the same trusted-network requirement. Its trainer-local
`RefitWorkerService` serves stable trainer metadata and separate version checksum
records over plaintext gRPC. Metadata digests detect corruption but do not
authenticate the trainer.

The split worker metadata protocol requires a coordinated upgrade of the MX server,
trainers, and generators. It replaces `GetWeightVersionShardManifest` and retires the
old `WeightVersionShard` manifest, endpoint, and count fields; mixed protocol versions
are unsupported. Retire active refit rounds and their leases before stopping workers.
Restart workers and remove obsolete refit metadata during the upgrade, preserving
unrelated Redis and model-cache state.


## Canonical S3 checkpoint publication and cache operations

Canonical S3 trainers consume Hugging Face tensor buckets produced by the
training framework. Framework-native bucket settings remain the default.
Integrations may use `MX_REFIT_DELTA_BUCKET_BYTES` as an explicit override, or
its 512 MiB default when they have no native setting. `MX_REFIT_DELTA_WORKERS`
(default `min(32, CPU count)`) controls trainer CPU work;
`MX_S3_UPLOAD_WORKERS` controls concurrent full-checkpoint batch uploads.
Framework integrations read the bucket-size setting before constructing the
stream; ModelExpress preserves the supplied bucket boundaries.
Full checkpoints are grouped into concurrently uploaded safetensors objects of
up to `MX_REFIT_FULL_CHECKPOINT_BATCH_BYTES` tensor bytes (4 GiB by default).
A single tensor larger than the configured size is uploaded on its own.
Generators download those objects concurrently through the ModelExpress S3
client. Each download worker validates one batch and overwrites the matching
regions in the existing local checkpoint mmaps.
The receiver marks its local checkpoint `UPDATING` before mutation and `READY`
only after success. An interrupted update remains unavailable until a fresh
initialization reseeds it from `seed_checkpoint_path`.

S3 versions normally use `XOR_DELTA`. Integrations may opt into a
framework-selected cadence of `FULL_HF_CHECKPOINT` versions; these publish
native HF safetensor shards, omit `base_version_id`, and become the exact base
for the next delta. The cadence is disabled by default. Slime exposes
`--modelexpress-full-hf-checkpoint-interval N`; Vime accepts
`full_hf_checkpoint_interval` in `--modelexpress-config`. In both cases, `N`
counts published ModelExpress weight versions.

With `MX_REFIT_DELTA_SURGICAL=true` (default `false`), vLLM checkpoint refits
track delta tensor names and the last completely installed checkpoint version.
Known lineage selects whole destination modules for graph-safe layerwise reload;
unknown lineage or a partially loaded module falls back to a full reload. Peer,
bounded tensor streaming, and failed installs clear the tracked checkpoint
version. When disabled, the receiver skips delta-name lineage reads and the
installer uses the full-checkpoint path. See the
[S3 delta refit guide](S3_DELTA_WEIGHT_REFIT.md).

The `vime_dynamo_delta_refit` and `dynamo_vllm_s3_delta_refit` Kubernetes examples
use a disposable SeaweedFS 4.48 S3 fixture. Both manifests pin the upstream
public `docker.io/chrislusf/seaweedfs` image by digest and run `weed mini` with
S3 on port 9000. The `mx-s3-creds` Secret supplies `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` to both SeaweedFS and the S3 clients; Vime CI creates
this Secret before deploying the stack. The `/healthz` readiness probe checks
that the S3 listener is serving; it does not validate the backing storage.
Data stays in an `emptyDir` volume. Volume counts are bounded for the 64 GiB
Vime and 16 GiB lifecycle fixtures, with 1 GiB SeaweedFS volumes.
Each storage pod requests 1 GiB of memory and has a 4 GiB limit for concurrent
trainer multipart uploads. `GOMEMLIMIT=3GiB` leaves headroom below that limit;
the previous 1 GiB container limit is insufficient for two trainer ranks.
The Vime CI smoke test consumes the example manifest directly; update both
manifests together when changing the fixture image.


## Bounded GPU refit deployment guidance

### Bounded GPU refit

Framework integrations choose the public staging, installation and memory
settings. The framework also owns the safe update point and the pause/restart
policy described below.

| Integration choice | Public setting or operation |
| --- | --- |
| Trainer storage | Set `ModelExpressTrainerConfig.staging_mode` explicitly. Prefer `IN_PLACE` when its lifetime and dtype requirements hold; otherwise start with `COPY_TO_HOST`. |
| Generator installation | Use `stage_weight()` followed by `apply_weight()`, then release the handle. Bounded staging defers batch reads until apply. |
| Receiver memory | Set `ModelExpressGeneratorConfig.staging_buffer_bytes` per buffer and `staging_buffers_count` for bounded installation. `staging_device` selects host or device receive memory. |

For unquantized vLLM models whose weights leave insufficient memory for a second
complete weight copy, initialize the generator with
`source_order=(WeightSource.TRAINER,)` and bounded staging:

```python
generator = ModelExpressGeneratorClient.initialize(
    ModelExpressGeneratorConfig(
        engine_context=engine_context,
        model_name=model_name,
        source_order=(WeightSource.TRAINER,),
        staging_buffer_bytes=4 * 1024**3,
        staging_buffers_count=2,
        staging_device="cuda",  # or "cpu" for pinned host staging
    )
)
# Pause generation on every replica before installing live weights.
staged = generator.stage_weight(version=WeightVersionRef(version_uid))
try:
    metrics = generator.apply_weight(staged)
finally:
    staged.release()
# Resume only after every replica completes successfully.
```

This API interleaves NIXL reads and per-module installation. It supports mixed
floating-point wire/engine dtypes through the existing conversion planner. The
per-buffer limit covers receive, conversion, full-pull scratch, and alignment.
Total arena capacity is `staging_buffer_bytes * staging_buffers_count` per
generator worker. A module larger than one buffer fails during preparation. Engine
post-load workspaces and live weights require additional headroom; the limit is
not a total process-memory cap.

`staging_device` selects where the arenas live. `"cuda"` (default) lands RDMA
in VRAM and commits with a device copy. `"cpu"` allocates pinned host memory,
registers it as NIXL DRAM, and commits with a host-to-device copy, so the arena
costs no VRAM. Host staging adds a host-to-device copy and depends on the
available host-memory and CPU-to-GPU bandwidth. `staging_buffers_count=2` posts the
next batch's READ into the other arena before the current batch is committed,
allowing transfer and installation to overlap. The benefit depends on the
hardware and the relative transfer and installation times; it does not guarantee
that the copy is hidden. Host arenas are an option for VRAM-constrained
deployments. Changing either option between updates is a workspace switch (see
below).

Any failure requires keeping the deployment paused and restarting its engines.
Some modules may already contain the new version, so a failed operation cannot
be treated as a usable old version. The framework owns this pause/restart policy.
Quantized engines, generator-peer publication, and object-storage delta replay
are not supported by this API. Trainer-source `stage_weight()` keeps its
full-copy behavior.
After releasing an update, callers can switch between full-copy and bounded
staging on the same trainer-only client. A mode switch disconnects the NIXL
agent and deregisters its workspace before freeing the old buffers, then
reinitializes registrations and plans for the new mode. Same-mode updates retain
their reusable workspace. Never switch while an update handle remains active.

Streaming preparation retries transient RPC, runtime, and manifest-validation
failures up to `max_transfer_attempts`, keeping the version lease across attempts.
Failed preparation storage is reset before retrying; a reset failure requires an
engine restart. Once installation starts, failures are not automatically retried.
Release errors remain visible to callers, but a locally released update no longer
holds the client's active slot even when lease deletion fails.
Metrics include `staging_peak_bytes`, `batches`, `bytes_received`, `wire_s`, and
`reconstruct_s`. `wire_s` measures READ posting through completion observation;
`wire_wait_s` measures the blocking completion wait. With two arenas, a READ can
overlap the previous batch's installation, so wire and installation times must
not be added as disjoint intervals. GPU validation is required for each target
model and topology before performance qualification.

Streaming reports independent `streaming_total_s`, `streaming_prepare_s`,
`streaming_apply_s`, and `streaming_release_s` intervals. Preparation contains
`source_metadata_s`, `layout_capture_s`, `transfer_planning_s`, and
`connection_registration_s`; the remaining preparation time includes version
discovery and lease/control operations. Application contains NIXL `wire_s`,
`reconstruct_s`, `install_commit_s` (including CUDA completion), `reload_s`, and
`post_install_sync_s`. In the generic installer, vLLM's post-load processing
refreshes attention-derived weights within `reload_s`. The ordinary
`perf/mx_receive_install_time` is not emitted for streaming installation because
its application interval includes network reads. With refit timing enabled,
DIRECT updates emit the same stage record as staged updates. The installation
stage's `materialization_s`, `receive_copy_s`, and `post_load_processing_s`
metadata report host-call durations inside the native engine lifecycle. They
exclude subsequent CUDA completion waits and are not standalone GPU latencies.
Do not sum nested parent and child intervals or maxima from different ranks.
Preserve raw samples and expose residual/unattributed time against the independent
total rather than describing the entire streaming operation as wire or install.


## Bounded streaming internals and diagnostics

Bounded streaming requires the framework to hold its update guard through
transfer, installation and verification. Each group is copied from a receive
arena into engine-owned load-time storage; vLLM performs post-load processing and
restores live parameter bindings. Partial installation can change the live
version. Engine-owned temporary tensors and post-load workspaces require VRAM
beyond the arena cap. Trainer staging is a separate choice. See
[bounded vLLM streaming installation](ARCHITECTURE.md#bounded-vllm-streaming-installation)
for the ownership and failure contract.

The vLLM installer retains its first successful canonical installation plan for
the generator client's lifetime, including supported quantized full-copy and
bounded streaming updates.
Canonical input schema, model architecture, module ownership, alias topology,
configuration and loader behavior must remain fixed for that lifetime. The
installer reuses its captured operations and destination metadata without
comparing them against the live model or copying cached plan fragments.
Trainer source conversion and its composition with these operations belong to
transfer planning.

Received batches are checked against the retained installation plan, including
expected names, shapes, dtypes and complete module inputs. Engine callbacks
receive independent storage, so they cannot retain a view into the receive
arena. Every update still obtains its current version and lease and reads fresh
tensor values. Transfer plans and physical bindings contain no cached weight
values or readiness. Workspace reset discards physical bindings; the logical
transfer-plan reuse policy is described in the architecture reference. Resetting
transport resources does not recreate the client's installation plan.
Wire-to-engine dtype conversion respects the captured destination slice, strides
and arena storage offset, including padding surrounding the destination view.
Bounded staging views are zeroed before each READ so untouched loader padding
cannot retain bytes from a previous batch or version.


## ModelExpress benchmark CI harness

[`ci/bench/`](../../ci/bench/) runs native vLLM cold-load and refit checks using
registered, pinned model profiles. `profiles.json` lists supported profile keys
and the defaults (`nemotron`, scenario `delta`). Only delta refit is implemented.
Add a profile to the catalog to extend the harness;
the trigger and lifecycle scripts do not contain model-specific allowlists.

| Profile | Pinned model | Resources per worker |
| --- | --- | --- |
| `nemotron` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, revision `bee7596271d1495f6992ae224aefde4410e816b8` | TP1, 16 CPU, 128 GiB RAM |
| `kimi` | `moonshotai/Kimi-K2.7-Code`, revision `74797c9c62378b951a1f6fcf5c4631024e9b8bef` | TP8, 64 CPU, 1 TiB RAM |

There has been **no live AWS validation**. Profile support means the model can be
selected, rendered, and tested; it does not establish GPU/kernel or refit
compatibility. Kimi requires eight GPUs on one node with sufficient host RAM.
Its quantized MLA refit path needs qualification; the experimental WNA16 workaround
is not included or applied. Failures remain failures and stop before resume.

### Trigger benchmark CI with a PR comment

After the workflow lands on the default branch, a repository writer can post:

```text
/bench
/bench --model kimi --scenario delta
/bench --model nemotron --sha <full-40-character-PR-head-SHA>
```

Bare `/bench` uses the catalog's default profile. `--model` selects a registered
profile, `--scenario` selects a registered experiment, and `--sha` optionally binds the request to an explicit PR head. Otherwise
the authorization job resolves the current head once. In both cases that exact
revision must match copy-pr-bot's `pull-request/<PR-number>` mirror. First use the
existing `/ok to test <SHA>` approval process if needed, wait for mirroring, then
post the benchmark comment. The new command does not grant copy-pr-bot approval. Closed
PRs, edited comments, stale explicit SHAs, unmirrored heads, unknown models/scenarios/options,
and commenters without write permission are rejected. New revisions need new
requests. Labels do not trigger this workflow.

The hosted gate checks permission and resolves immutable source/model/scenario outputs.
`bench-ci.yml` handles comment authorization and configured runtime selection, then
calls `rl-refit-ci.yml`. The reusable workflow accepts PR number, exact source SHA,
model, scenario and digest-pinned runtime inputs. Comment runs repeat the caller
permission, current-head and mirror checks before privileged builds. Automatic
runs derive the PR number from the upstream `pull-request/<N>` push and require
the event SHA to match the current open PR and approved mirror. One trusted harness
revision is resolved on the hosted runner and reused by build, test and cleanup:
the approved mirrored SHA for automatic runs, or the default branch for comments.
Privileged jobs build that approved ModelExpress revision using the selected
trusted Dockerfiles and harness scripts. Per-model runtime bases are
configured by digest; built images are also consumed by digest. Harness edits in
a PR run on GPUs after copy-pr-bot approval in automatic Nemotron CI. Opt-in
comment runs use the default-branch harness until the edits merge.
`issue_comment` requires the workflow on the default branch, as documented
by [GitHub](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#issue_comment).

Both automatic and comment-triggered CI reuse the SeaweedFS Deployment and
Service from `examples/rl/vime_dynamo_delta_refit/stack.yaml` in the benchmark's
isolated namespace. The benchmark reserves four CPUs for SeaweedFS to serve
concurrent model-weight reads. Setup creates per-run credentials, downloads the pinned model
snapshot on the CPU control pod, verifies uploads, and publishes its snapshot manifest
before the timed workload starts. Redis runs as an init sidecar with a startup
probe so it accepts connections before the ModelExpress server starts. The
separate S3 cleanup pod uses a digest-pinned CPU Python image containing only
boto3 and its dependencies, with boto3 taken from the client lockfile. It runs
without vLLM, Torch, Redis, or the server. The benchmark collects
run evidence once on success or failure; CI collects separately only if the run
report is missing. Cleanup releases the GPU pods before removing S3 objects
and records object and Kubernetes cleanup receipts. CI mounts the cluster's
retained `shared-model-cache` FSx claim only on the control pod and sets
`HF_HUB_CACHE` to `/model-cache/.cache/mx-rl-refit/hub`. Hugging Face reuses cached
files for the pinned revision across runs; a cold cache downloads them once.
Each run still uploads and verifies its isolated S3 seed. GPU workers do not
mount this cache, and cleanup leaves the shared source files intact.
Worker checkpoint preparation and delta publication finish before the timed
engine startup, so reference downloads do not compete with S3 cold loading.
Installed embedding verification resolves checkpoint names through the model's
native weight mapper before applying its shard loader and comparing tensor bytes.
The configured S3 addressing style is applied to both boto3 and Run:ai
Model Streamer (`RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0` for CI's path-style
endpoint). Worker logs include native Model Streamer warnings and errors.
Seed uploads use boto3's standard transfer client
for the custom HTTP endpoint, matching seed downloads. The existing `HF_TOKEN` secret is forwarded for
model access. Delta objects are verified absent before namespace deletion; SeaweedFS
and its seed data remain available if cleanup fails. If setup fails before bucket
creation, an explicit S3 `NoSuchBucket` response also verifies that run objects
are absent; other storage errors still fail cleanup. This path needs no AWS IAM role
or pre-populated bucket. The Nemotron profile defaults to digest-pinned vLLM 0.27.1,
the version documented in the model's deployment recipe, with
`MX_BENCH_RUNTIME_BASES.nemotron` available as an override.

Optional repository Actions variables configure placement and runtime overrides:

| Variable | Value |
| --- | --- |
| `MX_BENCH_KUBE_CONTEXT` | Kubernetes context available in `/teleport/kubeconfig.yaml`; defaults to `nv-prd-dgxc.teleport.sh-dynamo-aws-dev-02` |
| `MX_BENCH_RUNTIME_BASES` | JSON mapping profile keys to compatible digest-pinned vLLM base images, e.g. `{"nemotron":"registry/runtime@sha256:...","kimi":"registry/kimi-runtime@sha256:..."}` with full digests |

Runtime overrides take precedence over the profile default. Kimi requires an
override because its profile has no qualified default runtime. Each image must include
compatible Torch, vLLM, NIXL, NumPy, requests, and safetensors; the build installs
ModelExpress and verifies imports. Compatibility with each model's quantization
and refit path still requires hardware qualification. The existing `NGC_API_KEY`
secret supplies registry access. CI seeds its own S3 store and does not use the
Vime model directory; its Hugging Face cache has a separate directory on the same
FSx volume. Test and cleanup jobs provision Python 3.12 before installing
PyYAML. S3 cold-load qualification requires a streamer completion marker from
every TP rank and rejects streamer fallback. The harness retains the runtime's
FlashInfer compatibility checks. The seed-download step reads the snapshot
manifest and checks downloaded file sizes against S3 object sizes. Manual runs can use external S3
through their environment configuration.

The `RL weight refit CI` comment workflow delegates execution to the shared
`RL refit CI` workflow for the selected profile and scenario. The GPU job is named
`RL refit / Refit (PROFILE)` for the delta scenario. Every trusted
PR automatically calls the same workflow for Nemotron, under
`RL generator-only delta refit (vllm, s3) / Refit (nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)`.
Workflow callers can pass a profile key or its registered full model name; validation
resolves it to the profile key for resource names and configuration. The required `CI status check`
depends on this workflow, including cleanup; seed download, storage or refit failures
fail the gate. Kimi remains opt-in and requires a runtime override. Automatic runs check out the approved mirrored
SHA for the harness; comment runs use the default-branch harness. Offline
contracts run on every PR in the `RL refit harness tests` job. The namespace GPU quota follows the rendered profile's
TP size (one for Nemotron, eight for Kimi). Image builds have a 60-minute timeout;
the test job has a 60-minute timeout, its experiment step 45 minutes, and pods a
55-minute lifetime. Large models may exceed this initial qualification budget;
a timeout is a failure. Automatic trusted-PR and comment-triggered runs use per-PR,
per-job concurrency: when a replacement job becomes eligible, it cancels the older
queued or running job. Matrix entries use separate groups. Cleanup jobs remain
outside these groups, so newer runs can start while earlier cleanup is running or
waiting for a runner. Existing `always()` cleanup steps still run after job cancellation.
Actions runs expose the tested SHA and artifacts; the workflow does not post PR
comments.

### Manual CLI and environments

```bash
export KUBE_CONTEXT='<aws-context>' NAMESPACE='<existing-run-namespace>'
export SERVER_IMAGE='<registry/server:full-commit-sha>'
export WORKER_IMAGE='<registry/runtime:full-commit-sha>'
export CLEANUP_IMAGE='<registry/cleanup:full-commit-sha>'
export MX_CI_S3_BUCKET='<snapshot-bucket>' MX_CI_S3_REGION='<aws-region>'
export RESULTS_DIR=/tmp/mx-bench-unique-run
python3 ci/bench/scripts/prepare.py kimi "$RESULTS_DIR" \
  --environment aws-ci --run-id kimi-unique-run
# Inspect rendered manifests before deploying.
python3 ci/bench/scripts/lifecycle.py run "$RESULTS_DIR"
python3 ci/bench/scripts/lifecycle.py collect "$RESULTS_DIR"
python3 ci/bench/scripts/lifecycle.py cleanup "$RESULTS_DIR"
```

Manual callers provision their namespace, service account, secrets, and mirror.
The Python `lifecycle.py` CLI reads the rendered JSON configuration directly.
Seed download and publication share a two-hour deadline from process startup;
either process failing stops the run, including during worker readiness.
The Python `ci.py` entry point manages CI setup, execution, and namespace cleanup.

Use a fresh run ID/output directory; run IDs begin with the selected profile key.
`KUBECONFIG` overrides `/teleport/kubeconfig.yaml`. CLI overrides take precedence
over environment variables and environment-file values. Images require full
commit SHA tags or digests. The AWS preset selects H100 80 GB nodes,
0.70 GPU memory utilization, `ci-nightly-high-priority`, and `nvcr-imagepullsecret`.
S3 is the default with no custom endpoint or RDMA device. Profile defaults supply
TP/CPU/memory and object prefixes.

Use `--environment /path/to/environment.json` for placement/storage changes.
Fields include worker/control node selectors, tolerations, control resources,
service account, image-pull secrets, security context, `addressing_style`,
`pod_env` (including Secret references), `pod_env_from`, and image references.
Runtime tensor inventories can differ between vLLM versions and kernel layouts.
An environment may override `expected_tensors_per_rank` and
`expected_host_scales_per_rank` with a positive integer (or `null` for an
unqualified count). Keep qualified counts explicit and tied to the recorded image
digest; do not disable a failed count check merely to pass a run.

CLI options include context, namespace, region, bucket, endpoint URL, service
account, seed/delta prefixes, TP, CPU, memory, and GPU memory utilization. Literal
credentials must not enter the environment file: resolved settings are retained
as evidence. `--endpoint-url ''` clears an inherited endpoint.

Peer coverage remains manual: `--paths both` requires two suitable nodes plus
explicit `extra_worker_resources`, `worker_env.MX_NIXL_BACKEND`, and
`peer_transfer_marker`. See the [EFA example](../../examples/p2p_transfer_k8s/client/vllm/aws_efa/).
Setting these fields does not qualify transport. Each worker's TP ranks stay on
one node. The report rejects peer fallback to S3, ModelStreamer, or InstantTensor
and requires different donor/peer nodes.

### Shared RL refit experiment contract

The shared `harness/lifecycle.py` owns Kubernetes deployment, process deadlines, evidence collection
and cleanup. `harness/runner.py` exposes `RefitRunner`, which owns the shared pause/refit/verify/resume sequence.
Rendered configuration supplies explicit initial and target versions, sources by
worker role, publication evidence path and preparation commands. These values
keep the sequence independent of delta version naming and S3/peer role names.

Runtime code uses the `harness`, `scenarios.delta` and `engines.vllm` Python
packages. ConfigMap item mappings preserve these paths inside `/opt/benchmark`;
pods invoke tasks with `python3 -m`. The renderer mounts shared runtime code,
the selected scenario and the engine adapter, excluding host-only rendering and
Kubernetes lifecycle code. CLI paths under `scripts/` remain the same.

The `DeltaScenario` class stores its configuration and has four methods: `configure`, `verify_refit`,
`validate_inventory` and `validate_report`. It prepares the changed checkpoint
and validates delta-specific evidence while shared validation checks rank
completeness, source/version, addresses and timing. Reshard can add an experiment
through this contract later; there is no reshard implementation or CI case in
this PR. Comment-triggered GPU runs currently exercise delta over S3 only.

### Validation, timing, and cleanup

The benchmark uses vLLM’s default execution mode without forcing eager execution.

The CI-only `RefitWorkerExtension` in `ci/bench/engines/vllm/worker.py` extends
each vLLM worker with refit timing and weight/checkpoint validation RPC methods.

The driver checks every rank's tensor hashes, reconstructed checkpoint embedding
and its corresponding installed vocabulary embedding,
refit source/version, preserved tensor addresses, and resumed inference. Shared
validation checks GPU/CPU/host scale agreement for profiles with configured scale
counts. Model-specific scale-change counts and FlashInfer cache invalidation are
not asserted. A failed rank stops before resume. Native HTTP
pause/resume is not Dynamo administration. The publisher preserves the selected
checkpoint tensor's stored dtype and changes low-order bits to produce a delta.
The configured payload size (64 MiB for Nemotron, 1 GiB for Kimi) is a best-effort
target; reports include actual bytes, and size differences do not fail validation.
Peer refit transfers full runtime tensors.
Each rank verifies the reconstructed checkpoint tensor hash against the publisher's
expected hash. Runtime tensor inventories must change after refit and match between
S3 and peer workers when both paths run. Installed embedding validation resolves the engine vocabulary embedding and
checks each TP rank against the corresponding checkpoint slice, including
vocabulary padding. Profiles still select a checkpoint tensor;
its dtype must be supported by the installed safetensors/PyTorch and refit stack.

`report.json` retains model-load seconds, RPC and per-rank stage/install/total
seconds, failures, image IDs, and restarts. Loading excludes
scheduling, image pulls, engine warmup, and the separately downloaded reconstruction
seed. Stage timing covers the stage operation and completion synchronization;
installation timing covers apply plus CUDA synchronization. Total refit time is
stage plus installation; logging, pointer verification, hashing, and inference
are outside these intervals. The separate RPC duration includes those additional
operations and must not be compared as the same interval.
The first refit after client initialization may use a full checkpoint install
even with `MX_REFIT_DELTA_SURGICAL=1`, because the installer has not established
its live version. Confirm surgical metrics or logs before interpreting an install
measurement as surgical. GPU/CPU/host scale agreement does not establish that
untouched scales retained their exact pre-refit values.

The report separates `validation_status` (`PASS`/`FAILED`) from
`measurement_status` (`VALID`/`INVALID`). Overall `status` passes only when both
pass. Successful `latency_summary` entries include the model-load time, RPC time,
each rank's stage/install/total, and the maximum per-rank total. Model-load time
is the maximum of one timer per TP rank, with individual rank times retained.
Missing, duplicate, or unidentified rank timers invalidate measurements; untagged
model-load timers are accepted only for TP1. Every required
duration must be finite and nonnegative; missing timings invalidate a benchmark.
Failed runs keep raw measurements as evidence but have an empty summary.
Malformed `RESULT` log records are retained as parse-error metadata and make
the report fail, rather than preventing the failure report from being written.

This is one trial per run,
with no percentiles, baseline comparison, or performance threshold.
No AWS performance baseline has been established. Do not run assertion-based
validators with Python `-O`.

CI uploads evidence on success/failure and runs a separate cleanup job. It stops
publisher/workers, removes objects under the rendered `artifact_prefix`
(delta-prefix/run-ID for the delta scenario, including partial publications), and deletes the namespace after checking its
ownership label and verifying S3 deletion. Cleanup logs are retained separately; artifacts expire after
14 days. Cluster outages or forced workflow stops can still require manual
exact-run cleanup. S3 version history follows bucket lifecycle policy.

Run resources use the `mx-benchmark` label for selection and cleanup.
Manual and CI cleanup share exact-run prefix deletion, including incomplete
publication and retries after partial deletion. Manual cleanup preserves the
namespace, snapshot and local evidence. CI deletes its owned namespace only after
object deletion is verified; failures release benchmark GPU pods and retain the
namespace and service account for a retry. Retry `ci.py cleanup` with the original
run ID, namespace, GitHub run ID/attempt and image/environment settings. Rerunning
the whole GitHub workflow uses a new attempt and does not recover an older retained
namespace. Never delete the shared model prefix or bucket. Every fresh run
has isolated server/Redis state and empty worker volumes.
