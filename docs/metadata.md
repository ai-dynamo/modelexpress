# ModelExpress Metadata Architecture

This document describes the metadata storage and coordination layers for ModelExpress P2P transfers and model-cache lifecycle tracking.

## Overview

ModelExpress stores two related classes of metadata:

1. **P2P source metadata**: source workers publish tensor descriptors or artifact summaries plus transfer-engine metadata so target workers can discover compatible sources and fetch detailed manifests.
2. **Model-cache lifecycle metadata**: the server tracks model download state (`DOWNLOADING`, `DOWNLOADED`, `ERROR`) so replicas coordinate downloads and LRU eviction.

Both layers are selected by `MX_METADATA_BACKEND`, but they use separate storage namespaces:

| Backend | P2P source metadata | Model-cache lifecycle metadata |
|---------|---------------------|--------------------------------|
| Redis | `mx:source:*` keys | `mx:model:*` keys |
| Kubernetes | `ModelMetadata` CRDs + tensor ConfigMaps | `ModelCacheEntry` CRDs |

For P2P transfers, coordination between source and target GPU workers works as follows:

1. **Source** loads model weights, registers tensors with a transfer backend (NIXL or Mooncake), and publishes metadata to the MX server.
2. **Target** queries the MX server for available sources, fetches tensor metadata on demand, and executes RDMA transfers.
3. **Status** transitions (`INITIALIZING` -> `READY` -> `STALE`) signal when sources are available for transfers.

## Key Concepts

### Source Identity and Content-Addressed Keys

Every source is identified by a `SourceIdentity` proto containing all fields that affect tensor layout compatibility:

| Field | Example | Purpose |
|-------|---------|---------|
| `mx_version` | `"0.5.0"` | Format compatibility across upgrades |
| `mx_source_type` | `WEIGHTS`, `LORA`, `CUDA_GRAPH`, `TORCH_COMPILE_CACHE`, `TRITON_CACHE`, `DEEP_GEMM_CACHE`, `TILELANG_CACHE`, `CUTE_DSL_CACHE`, `FLASHINFER_CACHE` | Type of source metadata being served |
| `model_name` | `"deepseek-ai/DeepSeek-V3"` | Model identifier |
| `backend_framework` | `VLLM`, `SGLANG`, `TRT_LLM` | Inference framework |
| `tensor_parallel_size` | `8` | TP degree |
| `pipeline_parallel_size` | `2` | PP degree |
| `expert_parallel_size` | `4` | EP degree (MoE models) |
| `dtype` | `"bfloat16"` | Weight data type |
| `quantization` | `"fp8"`, `""` | Quantization method |
| `extra_parameters` | `{}` | Framework-specific config |

The server computes `mx_source_id = SHA256(canonical_json(identity))[:16]` -- a 16-char hex key used to address all metadata for sources with identical configuration. This is content-addressed: two sources with the same identity hash to the same `mx_source_id`, enabling automatic peer discovery.

Runtime accelerator compatibility is deliberately not part of `SourceIdentity` or `mx_source_id`. The source worker's active accelerator backend is stored as runtime `WorkerMetadata.accelerator` so rolling upgrades and existing pinned source-ID hash tests remain stable.

### Multi-Instance Support

Multiple replicas of the same model (same `SourceIdentity`) can coexist. Each GPU worker process generates a unique `worker_id` (`uuid4().hex[:8]`) at startup. The combination `(mx_source_id, worker_id)` uniquely identifies one worker's metadata.

Each worker publishes independently -- no inter-worker coordination or barriers required.

### Worker Rank

Workers use `torch.distributed.get_rank()` as their global rank, which captures both tensor-parallel and pipeline-parallel position. This is stored as `worker_rank` in metadata so targets can find a peer with a matching rank.

### Runtime Accelerator Compatibility

The source worker's runtime accelerator family, such as `cuda`, comes from the active `AcceleratorBackend.name`. It is published on `WorkerMetadata.accelerator` and also surfaced on the lightweight `SourceInstanceRef.accelerator` returned by `ListSources`. This field is used only for source compatibility filtering. It is not folded into `SourceIdentity`, does not affect `mx_source_id`, and does not change the Rust/Python pinned source-ID cross-check hashes.

Targets treat an empty `accelerator` value as unknown and do not reject it, which keeps transfers backward compatible with metadata published before this field existed. If both source and target publish non-empty accelerator values and they differ, the target skips that source. Because `SourceInstanceRef` carries the value, incompatible sources are dropped while handling `ListSources` -- before the selector orders candidates and before the `MAX_SOURCE_RETRIES` slice -- so incompatible sources cannot exhaust the retry budget ahead of a compatible one. The post-`GetMetadata` check on `WorkerMetadata.accelerator` remains as defense-in-depth, before target preparation or RDMA receive.

The same rule guards artifact cache transfers. vLLM JIT and compile caches (Torch compile, Triton, DeepGEMM, TileLang, CuTe DSL, FlashInfer) are accelerator-specific, so `discover_artifact_source` drops sources whose `SourceInstanceRef.accelerator` is incompatible before `GetMetadata`, then re-checks the authoritative `WorkerMetadata.accelerator` after the fetch. Both checks share the single `accelerators_compatible` helper (`metadata/payload.py`) with the RDMA tensor path, so empty-means-unknown behaves identically. The `k8s-service` backend does not yet expose artifact discovery, so this filtering applies to the central-coordinator backends only.

### Tensor and Artifact Source Payloads

`WorkerMetadata.source_payload` selects the source-specific metadata shape:

| Payload | Purpose |
|---------|---------|
| `tensor_source` | Tensor descriptors for weight transfer. Readers fall back to the deprecated top-level `tensors` field for old publishers. |
| `artifact_source` | Lightweight artifact discovery summary: `artifact_id`, `total_size`, `file_count`, `chunk_count`, and the owning `node_rank`. |

Artifact summaries do not contain full file or chunk tables. Targets use the worker's `worker_grpc_endpoint` to call `GetArtifactManifestHeader` and `GetArtifactManifestChunks` on the source worker, then use `PrepareArtifactChunk` and `ReleaseArtifactChunk` around each NIXL transfer. `artifact_id` is SHA-256 over the canonical artifact manifest JSON, encoded as lowercase hex without a prefix. File and chunk `checksum` fields use CRC32C lowercase hex. Manifest file paths are canonical absolute publisher paths and are included in the sealed manifest; transfer helpers may rewrite them to target-local staging paths before installing the artifact.

Artifact source discovery currently requires a central-coordinator backend (`redis` or `kubernetes`). The decentralized `k8s-service` backend fetches tensor manifests with `GetTensorManifest` and does not yet expose `artifact_source` discovery.

## gRPC API

```protobuf
service P2pService {
  rpc PublishMetadata(PublishMetadataRequest) returns (PublishMetadataResponse);
  rpc ListSources(ListSourcesRequest) returns (ListSourcesResponse);
  rpc GetMetadata(GetMetadataRequest) returns (GetMetadataResponse);
  rpc UpdateStatus(UpdateStatusRequest) returns (UpdateStatusResponse);
}
```

Source workers may also expose a per-worker `WorkerService` when P2P metadata is enabled:

```protobuf
service WorkerService {
  rpc GetTensorManifest(GetTensorManifestRequest) returns (GetTensorManifestResponse);
  rpc GetArtifactManifestHeader(GetArtifactManifestHeaderRequest) returns (GetArtifactManifestHeaderResponse);
  rpc GetArtifactManifestChunks(GetArtifactManifestChunksRequest) returns (GetArtifactManifestChunksResponse);
  rpc PrepareArtifactChunk(PrepareArtifactChunkRequest) returns (PrepareArtifactChunkResponse);
  rpc ReleaseArtifactChunk(ReleaseArtifactChunkRequest) returns (ReleaseArtifactChunkResponse);
}
```

### PublishMetadata

Called once per GPU worker after loading weights and registering with the transfer backend. The server computes `mx_source_id` from `identity` and returns it to the client.

```protobuf
PublishMetadataRequest {
  identity: SourceIdentity    // Server computes mx_source_id from this
  worker: WorkerMetadata       // One worker per call (rank, accelerator, backend metadata, tensors)
  worker_id: string            // Unique per GPU process (uuid4 hex[:8])
}
```

### ListSources

Lightweight listing -- returns `SourceInstanceRef` entries (no tensor data). Clients filter by `worker_rank` and `accelerator` to find matching peers, then call `GetMetadata` for the chosen one.

```protobuf
ListSourcesRequest {
  identity: SourceIdentity         // Optional: filter by source identity
  status_filter: SourceStatus      // Optional: e.g., SOURCE_STATUS_READY
}

ListSourcesResponse {
  instances: [SourceInstanceRef]   // One entry per worker
}

SourceInstanceRef {
  mx_source_id: string    // 16-char hex
  worker_id: string       // Unique worker identifier
  model_name: string      // Human-readable
  worker_rank: uint32     // Global rank for peer matching
  accelerator: string     // Runtime accelerator family for pre-fetch compatibility filtering
}
```

### GetMetadata

Fetches full metadata for one specific worker. Called on demand after filtering `ListSources` results. In central metadata mode this can include tensor descriptors directly. In P2P metadata mode the central response carries endpoint pointers and source summaries; targets fetch tensor descriptors or artifact manifests from `WorkerService`.

Accelerator compatibility filtering happens in two places. `SourceInstanceRef` now carries the source's `accelerator`, so the target drops incompatible sources during `ListSources` handling, before ordering and before the `MAX_SOURCE_RETRIES` slice; this prevents incompatible sources from consuming every retry slot and stranding a compatible one. The target then re-checks the authoritative `WorkerMetadata.accelerator` after `GetMetadata` as defense-in-depth against empty refs on older servers, stale records, or metadata drift between list and fetch. In both places, a target skips a source only when both source and target publish non-empty, different accelerator values.

```protobuf
GetMetadataRequest {
  mx_source_id: string   // From ListSources or PublishMetadata response
  worker_id: string      // From ListSources or PublishMetadata response
}
```

### WorkerService Artifact Manifest APIs

Artifact targets use `GetArtifactManifestHeader` to fetch the sealed artifact identity, worker endpoints, aggregate counts, byte chunk size, and the file table for install planning. Chunk metadata is fetched separately through `GetArtifactManifestChunks`, which pages the flat chunk table by global `chunk_index`.

Artifact bytes move through NIXL, not through the metadata service. For each chunk, the target calls `PrepareArtifactChunk` so the source reads that file range into a registered DRAM buffer and returns a NIXL transfer descriptor plus a lease. The target receives that range into a local registered DRAM buffer, verifies the chunk CRC32C, writes it to the target staging file, and then calls `ReleaseArtifactChunk` to free the source lease.

The header currently returns the full file table sorted by manifest path, while chunk metadata is paged. Very high file-count artifacts can make the header large; if that becomes a production shape, the protocol should add a paged file-table RPC instead of increasing gRPC message limits.

`MX_ARTIFACT_TRANSFER_CHUNK_SIZE` controls the manifest chunk size for artifact transfer. The default is 64 MiB and the maximum accepted value is 4 GiB. Larger chunks reduce manifest size and per-chunk RPC overhead, but each source and target worker allocates registered DRAM buffers sized by roughly `chunk_size * max_inflight_chunks`; smaller chunks reduce buffer memory at the cost of more RPCs and checksum work.

### Tarred Cache Artifact Helpers

The Python `P2PArtifactTransfer` interface is the shared lifecycle for cache artifact transfer helpers:

1. Source worker creates a transfer helper, calls `prepare_source()`, and publishes the returned bundle with `publish_artifact_source()`.
2. Target worker creates the same helper type with its own `target_root` and `bundle_root`, calls `discover_and_transfer()` or `transfer_from_worker()`, and receives a target-local staged artifact.
3. Target worker calls `install()` to unpack the staged artifact into the runtime cache directory before the framework starts using that cache.

The current implementation, `TarredP2PArtifactTransfer`, packages the source cache directory into one uncompressed tar file before building the artifact manifest. The source manifest records the publisher's tar path and therefore contributes that path to `artifact_id`. During transfer, the target rewrites the received file table to its own `bundle_root / artifact.tar`, then extracts that tar into `target_root`. This keeps the published manifest sealed while avoiding any requirement that source and target share the same absolute staging path.

Factory helpers provide the cache source types currently expected by loaders:

| Helper | `mx_source_type` | Target cache shape |
|--------|------------------|--------------------|
| `torch_compile_cache_artifact_transfer()` | `TORCH_COMPILE_CACHE` | TorchInductor/vLLM torch compile cache directory |
| `triton_cache_artifact_transfer()` | `TRITON_CACHE` | Triton kernel cache directory |
| `deep_gemm_cache_artifact_transfer()` | `DEEP_GEMM_CACHE` | DeepGEMM JIT cache directory (`DG_JIT_CACHE_DIR`, or `VLLM_CACHE_ROOT/deep_gemm`) |
| `tilelang_cache_artifact_transfer()` | `TILELANG_CACHE` | TileLang JIT cache directory (`TILELANG_CACHE_DIR`, or `~/.tilelang/cache`) |
| `cute_dsl_cache_artifact_transfer()` | `CUTE_DSL_CACHE` | CuTe DSL compiled-kernel cache directory (`CUTE_DSL_CACHE_DIR`, or `$TMPDIR/<user>/cutlass_python_cache`) |
| `flashinfer_cache_artifact_transfer()` | `FLASHINFER_CACHE` | Engine-selected FlashInfer JIT and autotune cache directories packaged as one artifact |

### UpdateStatus

Transitions a worker's lifecycle status. Called periodically by the client-side publisher thread to refresh `updated_at`, and on shutdown to mark `STALE`.

```protobuf
UpdateStatusRequest {
  mx_source_id: string
  worker_id: string
  worker_rank: uint32
  status: SourceStatus   // INITIALIZING -> READY -> STALE
}
```

## Source Lifecycle

```mermaid
stateDiagram-v2
    [*] --> INITIALIZING : PublishMetadata
    INITIALIZING --> READY : Heartbeat (NIXL healthy)
    READY --> READY : Heartbeat refreshes updated_at
    READY --> STALE : atexit or reaper timeout
    INITIALIZING --> STALE : Reaper timeout
    STALE --> Deleted : Reaper GC
```

- **INITIALIZING**: Worker has published metadata but heartbeat hasn't confirmed NIXL health yet
- **READY**: Worker is healthy and accepting RDMA connections. Heartbeat refreshes `updated_at` every `MX_HEARTBEAT_INTERVAL_SECS` (default 30s)
- **STALE**: Worker is no longer available. Set by the client `atexit` handler on clean shutdown (SIGTERM), or by the server-side reaper when `updated_at` exceeds `MX_HEARTBEAT_TIMEOUT_SECS` (default 90s)
- **Deleted**: Reaper garbage-collects stale entries after `MX_GC_TIMEOUT_SECS` (default 3600s)

## Backend Implementations

Configured via `MX_METADATA_BACKEND` environment variable:

| Value | Backend | Use Case |
|-------|---------|----------|
| `redis` | Redis | Production with Redis |
| `kubernetes` / `k8s` / `crd` | Kubernetes CRDs | K8s-native deployments |

### Redis Backend

#### Storage Layout

The Redis backend uses separate key prefixes for P2P source metadata and model-cache lifecycle metadata.

Three types of Redis keys are relevant:

**Source index key** -- `mx:source:{source_id}` (Redis Hash)

| Field | Value | Purpose |
|-------|-------|---------|
| `__attributes__` | JSON of all `SourceIdentity` fields | Stored once per source, avoids duplication |
| `{worker_id}` | `"{global_rank}"` | Presence marker with rank for fast listing |

**Worker data key** -- `mx:source:{source_id}:{worker_id}` (Redis Hash)

| Field | Value | Purpose |
|-------|-------|---------|
| `"{worker_rank}"` | JSON `WorkerRecordJson` | Full tensor metadata for one rank |

**Model lifecycle key** -- `mx:model:{model_name}` (Redis Hash)

| Field | Value | Purpose |
|-------|-------|---------|
| `provider` | `HuggingFace`, `Ngc`, or `Gcs` | Provider associated with the cached model |
| `status` | `DOWNLOADING`, `DOWNLOADED`, or `ERROR` | Download lifecycle state |
| `created_at` | RFC3339 timestamp | First write time, preserved across status updates |
| `last_used_at` | RFC3339 timestamp | Last status write or cache hit time for LRU eviction |
| `message` | Optional string | Download progress, retry, or error detail |

Global listing uses `SCAN` with pattern `mx:source:????????????????` (exactly 16 hex chars) to enumerate source index keys without a secondary index.
Model-cache listing uses `SCAN` with pattern `mx:model:*`; LRU ordering and status counts are computed by pipelined reads and Rust-side sorting/tallying.

No Redis TTL is applied to keys. P2P stale detection and cleanup are handled by the server-side reaper (see Source Lifecycle above). Model lifecycle entries are deleted when cache eviction removes a model.

#### Example Redis State

```
# Source index -- identity stored once, workers as presence markers
mx:source:a1b2c3d4e5f67890
  __attributes__  ->  {"model_name":"deepseek-ai/DeepSeek-V3","mx_version":"0.5.0",...}
  f3a2b1c4        ->  "0"    # worker_id f3a2b1c4, global rank 0
  e7d6c5b8        ->  "1"    # worker_id e7d6c5b8, global rank 1

# Worker data -- full tensor metadata
mx:source:a1b2c3d4e5f67890:f3a2b1c4
  "0"  ->  {"worker_rank":0,"backend_type":"nixl","nixl_metadata":[...],"tensors":[...],"status":2,...}

mx:source:a1b2c3d4e5f67890:e7d6c5b8
  "1"  ->  {"worker_rank":1,"backend_type":"nixl","nixl_metadata":[...],"tensors":[...],"status":2,...}

# Worker data -- artifact source summary
mx:source:b2c3d4e5f67890a1:f3a2b1c4
  "0"  ->  {"worker_rank":0,"backend_type":"none","artifact_source":{"artifact_id":"...","total_size":"67108864","file_count":1,"chunk_count":8},"status":2,...}

# Model lifecycle -- download state for model-cache coordination
mx:model:deepseek-ai/DeepSeek-V3
  provider     ->  "HuggingFace"
  status       ->  "DOWNLOADED"
  created_at   ->  "2026-04-29T22:00:00Z"
  last_used_at ->  "2026-04-29T22:10:00Z"
  message      ->  "Model download completed successfully"
```

#### JSON Schemas

**WorkerRecordJson** (stored per rank in worker data hash):
```json
{
  "worker_rank": 0,
  "backend_type": "nixl",
  "nixl_metadata": [222, 173, 190, 239],
  "transfer_engine_session_id": null,
  "tensors": [
    {
      "name": "model.layers.0.self_attn.q_proj.weight",
      "addr": "139948187451390",
      "size": "134217728",
      "device_id": 0,
      "dtype": "bfloat16"
    }
  ],
  "status": 2,
  "updated_at": 1700000000000
}
```

`addr` and `size` are serialized as strings to avoid JSON precision loss with large u64 values.

Artifact workers use the same worker record shape with `artifact_source` instead of tensor descriptors:

```json
{
  "worker_rank": 0,
  "backend_type": "none",
  "nixl_metadata": [],
  "transfer_engine_session_id": null,
  "tensors": [],
  "status": 2,
  "artifact_source": {
    "artifact_id": "a0f08392f2abc45f78bd59f0fe2c601750c2b270dc5cc37c2166d86a65398466",
    "total_size": "67108864",
    "file_count": 1,
    "chunk_count": 8
  },
  "updated_at": 1700000000000
}
```

### Kubernetes CRD Backend

Uses `ModelMetadata` CRDs for P2P source metadata, `ConfigMap`s for tensor descriptors (to avoid etcd size limits), and `ModelCacheEntry` CRDs for model-cache lifecycle state.

**P2P CRD name format**: `mx-source-{source_id}-{worker_id}`

**ConfigMap name format**: `mx-source-{source_id}-{worker_id}-tensors-worker-{rank}`

ConfigMaps use `ownerReferences` pointing to the parent CRD so they are garbage-collected automatically.

**Model lifecycle CRD name format**: `mx-cache-{sanitized-model-name}-{hash}`

`ModelCacheEntry.spec.modelName` preserves the original model name while `status.phase`, `status.createdAt`, `status.lastUsedAt`, and `status.message` track the same lifecycle fields as the Redis `mx:model:*` hash.

#### Example P2P CRD

```bash
kubectl get modelmetadatas -n <namespace>
kubectl get modelmetadata mx-source-a1b2c3d4e5f67890-f3a2b1c4 -n <namespace> -o yaml
```

```yaml
apiVersion: modelexpress.nvidia.com/v1alpha1
kind: ModelMetadata
metadata:
  name: mx-source-a1b2c3d4e5f67890-f3a2b1c4
  labels:
    modelexpress.nvidia.com/mx-source-id: a1b2c3d4e5f67890
    modelexpress.nvidia.com/mx-worker-id: f3a2b1c4
spec:
  modelName: deepseek-ai/DeepSeek-V3
status:
  worker:
    workerRank: 0
    backendType: nixl
    nixlMetadata: <base64>
    tensorCount: 1327
    tensorConfigMap: mx-source-a1b2c3d4e5f67890-f3a2b1c4-tensors-worker-0
    accelerator: cuda
    status: Ready
    updatedAt: "2025-11-14T22:13:20Z"
  conditions:
    - type: Ready
      status: "True"
      reason: WorkerReady
      message: Worker is ready
      lastTransitionTime: "2025-11-14T22:13:20Z"
  observedGeneration: 1
  publishedAt: "2025-11-14T22:13:20Z"
```

Artifact source CRDs carry an artifact summary instead of a tensor `ConfigMap` reference:

```yaml
apiVersion: modelexpress.nvidia.com/v1alpha1
kind: ModelMetadata
metadata:
  name: mx-source-b2c3d4e5f67890a1-f3a2b1c4
spec:
  modelName: artifact-transfer-e2e
  sourceType: deep_gemm_cache
status:
  worker:
    workerRank: 0
    backendType: none
    accelerator: cuda
    artifactSource:
      artifactId: a0f08392f2abc45f78bd59f0fe2c601750c2b270dc5cc37c2166d86a65398466
      totalSize: 67108864
      fileCount: 1
      chunkCount: 8
      nodeRank: 0
    tensorCount: 0
    status: Ready
    updatedAt: "2025-11-14T22:13:20Z"
```

#### Example Model Lifecycle CRD

```bash
kubectl get modelcacheentries -n <namespace>
kubectl get modelcacheentry mx-cache-deepseek-ai--deepseek-v3-<hash> -n <namespace> -o yaml
```

```yaml
apiVersion: modelexpress.nvidia.com/v1alpha1
kind: ModelCacheEntry
metadata:
  name: mx-cache-deepseek-ai--deepseek-v3-<hash>
spec:
  modelName: deepseek-ai/DeepSeek-V3
  provider: HuggingFace
status:
  phase: Downloaded
  createdAt: "2026-04-29T22:00:00Z"
  lastUsedAt: "2026-04-29T22:10:00Z"
  message: Model download completed successfully
```

## Client Workflow

### Source Path (load from disk, publish metadata)

```mermaid
sequenceDiagram
    participant W as GPU Worker
    participant MX as MX Server
    participant Backend as Redis / K8s

    W->>W: Load weights from disk (or GDS)
    W->>W: process_weights_after_loading()
    W->>W: Collect all post-processed tensors
    W->>W: Initialize NIXL agent, register tensors
    W->>MX: PublishMetadata(identity, worker, worker_id)
    MX->>Backend: Store worker metadata (status=INITIALIZING)
    MX-->>W: mx_source_id
    W->>W: Start PublisherThread
    loop Every MX_HEARTBEAT_INTERVAL_SECS
        W->>MX: UpdateStatus(mx_source_id, worker_id, rank, READY)
        MX->>Backend: Patch status + updated_at
    end
```

### Target Path (receive via RDMA)

```mermaid
sequenceDiagram
    participant W as GPU Worker
    participant MX as MX Server

    W->>MX: ListSources(identity, status=READY)
    MX-->>W: [SourceInstanceRef, ...]
    W->>W: Filter by worker_rank and accelerator, then order via SourceSelector
    W->>W: Load dummy weights, initialize NIXL agent
    loop For each candidate (max MAX_SOURCE_RETRIES) until metadata found
        W->>MX: GetMetadata(mx_source_id, worker_id)
        MX-->>W: WorkerMetadata (tensors, nixl_metadata)
        alt Metadata missing or fetch error
            W->>W: Try next candidate
        else Accelerator mismatch and both values known
          W->>W: Skip candidate before target preparation
        end
    end
    W->>W: Add remote NIXL agent, execute RDMA transfers
    alt Transfer fails (SourceTransferError / ManifestMismatchError)
        W->>W: Abandon RDMA, fall through to next strategy (GDS, disk)
    end
    W->>W: process_weights_after_loading()
    W->>W: Register and publish own metadata (become a source)
```

### Three-Tier Loading Strategy

The `MxModelLoader` (`--load-format modelexpress`; `mx` alias) auto-detects the best loading strategy:

1. **RDMA** -- If `ListSources` returns READY instances with matching rank, and the per-candidate metadata fetch confirms a compatible accelerator, receive weights via NIXL/Mooncake
2. **GDS** -- If no source available and GPUDirect Storage is available, load directly from file to GPU
3. **Disk** -- Standard vLLM `DefaultModelLoader` as final fallback

After loading by any path, the worker registers its tensors and publishes metadata so future workers can discover it as an RDMA source.

## Transfer Backends

`WorkerMetadata` stores runtime compatibility metadata plus a `oneof backend_metadata` field supporting multiple transfer backends. The `accelerator` field records the source worker's accelerator family for target-side filtering; empty means unknown and is accepted for backward compatibility.

| Backend | Field | Description |
|---------|-------|-------------|
| NIXL | `nixl_metadata` (bytes) | Serialized NIXL agent blob for RDMA connections |
| Mooncake | `transfer_engine_session_id` (string) | TransferEngine session ID (`"ip:port"`) |

The `backend_type` discriminator is persisted in storage for unambiguous deserialization.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MX_METADATA_BACKEND` | (required) | `redis` or `kubernetes` |
| `MX_SERVER_ADDRESS` | `localhost:8001` | gRPC server address (recommended) |
| `MODEL_EXPRESS_URL` | `localhost:8001` | Deprecated, pending removal in a future release. Still read by all client paths and takes precedence when both are set; keep setting it during the transition. |
| `MX_REDIS_HOST` / `REDIS_HOST` | `localhost` | Redis host |
| `MX_REDIS_PORT` / `REDIS_PORT` | `6379` | Redis port |
| `REDIS_URL` | (computed) | Full Redis URL (overrides host/port) |
| `MX_METADATA_NAMESPACE` / `POD_NAMESPACE` | (required for Kubernetes) | K8s namespace for CRD backend |
| `MX_HEARTBEAT_INTERVAL_SECS` | `30` | Client heartbeat frequency |
| `MX_HEARTBEAT_TIMEOUT_SECS` | `90` | Server reaper staleness threshold |
| `MX_REAPER_SCAN_INTERVAL_SECS` | `30` | Server reaper scan frequency |
| `MX_GC_TIMEOUT_SECS` | `3600` | Time before stale entries are deleted |
| `MX_POOL_REG` | `0` | Register each unique cudaMalloc allocation instead of each tensor (allocation-level NIXL registration) |

## Debugging

### Verify server connectivity

```bash
grpcurl -plaintext <server_host>:8001 list
grpcurl -plaintext -d '{}' <server_host>:8001 model_express.p2p.P2pService/ListSources
```

### Inspect Redis state

```bash
redis-cli KEYS "mx:source:*"
redis-cli HGETALL "mx:source:<source_id>"
redis-cli HGETALL "mx:source:<source_id>:<worker_id>"
redis-cli KEYS "mx:model:*"
redis-cli HGETALL "mx:model:<model_name>"
```

### Inspect K8s state

```bash
kubectl get modelmetadatas -n <namespace>
kubectl get modelcacheentries -n <namespace>
kubectl get configmaps -l modelexpress.nvidia.com/mx-source-id=<source_id> -n <namespace>
```

### Common failures

| Symptom | Likely Cause |
|---------|-------------|
| `ListSources` returns empty | No source has published + updated status to READY yet |
| `GetMetadata` returns `found: false` | Worker was garbage-collected by reaper, or wrong `mx_source_id`/`worker_id` |
| Target stuck waiting | Source still loading (check source pod logs for progress) |
| K8s CRs missing | RBAC issue -- check source logs and service account permissions for both `modelmetadatas` and `modelcacheentries` |
| Stale P2P metadata after redeploy | Reaper marks stale within 90s. For immediate Redis cleanup: delete `mx:source:*` keys or `FLUSHDB` in a dedicated Redis DB |
| Stale model lifecycle metadata after redeploy | Inspect `mx:model:*` or `modelcacheentries`; delete the stale lifecycle entry if it no longer matches cache contents |
| Transfer failure with address errors | Source pod restarted -- GPU addresses are invalid. RDMA is aborted and loading falls through to GDS, then disk; the target does not retry another source once transfer has started |
