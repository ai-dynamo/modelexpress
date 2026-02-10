# ModelExpress Metadata Architecture

This document describes the metadata storage layer for ModelExpress P2P transfers, covering both the current Redis-based implementation and the proposed Kubernetes CRD alternative.

## Overview

ModelExpress P2P transfers require coordination between source and target instances:
1. **Source** publishes NIXL metadata (agent info + tensor descriptors) after loading model weights
2. **Target** queries for source metadata to establish RDMA connections
3. **Coordination** signals ensure targets wait for sources to be fully ready

## Current Redis-Based Implementation

### Redis Keys and Data Structures

| Key Pattern | Type | Purpose | TTL |
|-------------|------|---------|-----|
| `mx:model:{model_name}` | String (JSON) | Model metadata with all workers | None |
| `mx:models` | Set | Index of all registered model names | None |
| `mx:nixl_ready:{model}:worker:{id}` | String (JSON) | Source readiness signal per worker | 4 hours |

### Data Schemas

#### ModelMetadataRecord (`mx:model:{model_name}`)

```json
{
  "model_name": "deepseek-ai/DeepSeek-V3",
  "workers": [
    {
      "worker_rank": 0,
      "nixl_metadata": "<base64-encoded NIXL agent blob>",
      "tensors": [
        {
          "name": "model.layers.0.self_attn.q_proj.weight",
          "addr": "139948187451390",
          "size": "134217728",
          "device_id": 0,
          "dtype": "float8_e4m3fn"
        }
      ]
    }
  ],
  "published_at": 1769568000
}
```

**Notes:**
- `addr` and `size` are serialized as strings to avoid JSON precision loss with large u64 values
- Workers are merged atomically via Lua script to handle concurrent publishing from multiple workers
- Sorted by `worker_rank` after merge

#### NIXL Ready Signal (`mx:nixl_ready:{model}:worker:{id}`)

```json
{
  "session_id": "0e2dcc70-1234-5678-90ab-cdef12345678",
  "nixl_ready": true,
  "stability_verified": true
}
```

**Purpose:** Prevents targets from connecting before sources are fully initialized. Published by source after:
1. vLLM health endpoint returns 200
2. 30-second grace period
3. Successful test inference

### Workflow: Redis

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SOURCE INSTANCE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Load model weights from disk                                              │
│ 2. Register tensors with NIXL                                                │
│ 3. Publish metadata via gRPC → Server → Redis                                │
│    - Server receives PublishMetadata(model_name, workers)                    │
│    - Server runs atomic Lua merge: mx:model:{model_name}                     │
│ 4. Wait for health check + warmup                                            │
│ 5. Publish NIXL ready flag directly to Redis:                                │
│    - mx:nixl_ready:{model}:worker:{id} (for each worker)                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            TARGET INSTANCE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Initialize dummy model structure                                          │
│ 2. Register target tensors with NIXL                                         │
│ 3. Poll Redis for NIXL ready flag:                                           │
│    - mx:nixl_ready:{model}:worker:{id}                                       │
│    - Wait until nixl_ready=true AND stability_verified=true                  │
│ 4. Query metadata via gRPC → Server → Redis                                  │
│    - Server receives GetMetadata(model_name)                                 │
│    - Server returns workers from mx:model:{model_name}                       │
│ 5. Add remote NIXL agents using nixl_metadata                                │
│ 6. Execute RDMA transfers (get_zcopy)                                        │
│ 7. Run FP8 processing on received weights                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### gRPC API (Current)

```protobuf
service P2pService {
  rpc PublishMetadata(PublishMetadataRequest) returns (PublishMetadataResponse);
  rpc GetMetadata(GetMetadataRequest) returns (GetMetadataResponse);
}

message WorkerMetadata {
  uint32 worker_rank = 1;
  bytes nixl_metadata = 2;           // Serialized NIXL agent blob
  repeated TensorDescriptor tensors = 3;
}

message TensorDescriptor {
  string name = 1;
  uint64 addr = 2;
  uint64 size = 3;
  uint32 device_id = 4;
  string dtype = 5;
}
```

### Limitations of Redis Approach

1. **External Dependency**: Requires Redis sidecar container
2. **No Native K8s Integration**: Doesn't leverage Kubernetes APIs for discovery
3. **Session Management**: TTL-based cleanup can be unreliable
4. **Observability**: Redis state is opaque to K8s tooling (kubectl, dashboards)
5. **Scaling**: Single Redis instance can become a bottleneck

---

## Layered Architecture (Recommended)

The recommended production architecture uses a layered approach: an **in-memory cache** for fast reads, with optional **write-through persistence** to Redis or Kubernetes CRDs for high availability.

### Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│ MX Server                                                             │
│                                                                       │
│  gRPC Request                                                         │
│       │                                                               │
│       ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │              In-Memory Cache (always present)                    │ │
│  │  HashMap<String, ModelMetadataRecord> behind RwLock              │ │
│  │  - Nanosecond reads (no network hop)                            │ │
│  │  - Atomic merge for concurrent workers                          │ │
│  │  - Ready flags stored here (always ephemeral)                   │ │
│  └────────────────────────┬────────────────────────────────────────┘ │
│                           │ write-through (if configured)            │
│                           ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │              Persistent Backend (optional)                       │ │
│  │                                                                  │ │
│  │  Redis:      mx:model:{name} → JSON (Lua atomic merge)          │ │
│  │  Kubernetes: ModelMetadata CRD + ConfigMap (tensor descriptors)  │ │
│  │                                                                  │ │
│  │  On startup: hydrates in-memory cache from persistent backend    │ │
│  │  On write:   write-through (best-effort, warns on failure)       │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Backend Modes

Configured via `MX_METADATA_BACKEND` environment variable:

| Mode | Env Value | In-Memory | Persistent | Use Case |
|------|-----------|-----------|------------|----------|
| **Standalone** | `memory` (default) | Primary | None | Dev, testing, single-server |
| **Redis HA** | `redis` | Cache | Redis write-through | Production with Redis sidecar |
| **K8s Native** | `kubernetes` | Cache | CRD write-through | K8s-native deployments |
| **Redis Only** | `redis-only` | None | Redis direct | Legacy / backward compat |
| **K8s Only** | `kubernetes-only` | None | CRD direct | Minimal memory footprint |

### Startup Hydration

When using a layered mode (`redis` or `kubernetes`), the server hydrates the in-memory cache from the persistent backend on startup:

1. Server starts, connects to persistent backend
2. Lists all model names from persistent store
3. Loads each model's metadata into in-memory cache
4. Subsequent reads hit only the cache (no backend round-trip)

This means the server recovers its state after a restart **without** requiring sources to re-publish.

### Ready Coordination

Ready flags (`PublishReady` / `GetReady`) are **always stored in-memory** regardless of backend mode. This is because:

- Ready flags are tied to running GPU processes (ephemeral by nature)
- GPU memory addresses become invalid after source pod restart
- Sub-millisecond latency is important for coordination polling
- No benefit from persisting stale ready state

### Implementation

```rust
// Backend trait (all backends implement this)
#[async_trait]
pub trait MetadataBackend: Send + Sync {
    async fn connect(&self) -> MetadataResult<()>;
    async fn publish_metadata(&self, model_name: &str, workers: Vec<WorkerMetadata>) -> MetadataResult<()>;
    async fn get_metadata(&self, model_name: &str) -> MetadataResult<Option<ModelMetadataRecord>>;
    async fn remove_metadata(&self, model_name: &str) -> MetadataResult<()>;
    async fn list_models(&self) -> MetadataResult<Vec<String>>;
}

// Layered backend (in-memory + optional persistent)
pub struct LayeredBackend {
    cache: InMemoryBackend,                        // Always present
    persistent: Option<Arc<dyn MetadataBackend>>,  // Redis or K8s
}
```

---

## Kubernetes CRD Alternative

### Design Goals

1. **Native K8s Integration**: Use CRDs for metadata storage
2. **No External Dependencies**: Eliminate Redis sidecar
3. **kubectl-Friendly**: Inspect state with `kubectl get modelmetadata`
4. **Built-in Lifecycle**: Use owner references for automatic cleanup
5. **Watch-Based**: Use K8s watch for efficient polling

### Custom Resource Definitions

#### ModelMetadata CRD

```yaml
apiVersion: modelexpress.nvidia.com/v1alpha1
kind: ModelMetadata
metadata:
  name: deepseek-ai-deepseek-v3  # Sanitized model name
  namespace: kavin
  labels:
    modelexpress.nvidia.com/model: deepseek-ai/DeepSeek-V3
  ownerReferences:
    - apiVersion: apps/v1
      kind: Deployment
      name: mx-source
      uid: <source-deployment-uid>
spec:
  modelName: deepseek-ai/DeepSeek-V3
  expectedWorkers: 8
status:
  phase: Ready  # Pending | Initializing | Ready | Stale
  workers:
    - workerRank: 0
      nixlMetadata: <base64-encoded blob>
      tensorCount: 1327
      publishedAt: "2026-01-27T18:02:47Z"
      ready: true
      stabilityVerified: true
    - workerRank: 1
      # ...
  conditions:
    - type: AllWorkersPublished
      status: "True"
      lastTransitionTime: "2026-01-27T18:02:50Z"
    - type: StabilityVerified
      status: "True"
      lastTransitionTime: "2026-01-27T18:03:27Z"
  observedGeneration: 1
  publishedAt: "2026-01-27T18:02:47Z"
```

#### TensorDescriptors ConfigMap (Large Data)

To avoid hitting etcd size limits (~1.5MB), tensor descriptors are stored in a separate ConfigMap:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: deepseek-ai-deepseek-v3-tensors-worker-0
  namespace: kavin
  labels:
    modelexpress.nvidia.com/model: deepseek-ai/DeepSeek-V3
    modelexpress.nvidia.com/worker: "0"
  ownerReferences:
    - apiVersion: modelexpress.nvidia.com/v1alpha1
      kind: ModelMetadata
      name: deepseek-ai-deepseek-v3
data:
  tensors.json: |
    [
      {"name": "model.embed_tokens.weight", "addr": "139948187451390", "size": "134217728", "device_id": 0, "dtype": "bfloat16"},
      {"name": "model.layers.0.self_attn.q_proj.weight", ...}
    ]
```

### Workflow: CRD-Based

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SOURCE INSTANCE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Load model weights from disk                                              │
│ 2. Register tensors with NIXL                                                │
│ 3. Check if ModelMetadata CR exists:                                         │
│    - If not: Create ModelMetadata CR with ownerRef to source Deployment      │
│    - If exists: Patch status.workers[rank]                                   │
│ 4. Create/Update ConfigMap with tensor descriptors                           │
│ 5. Wait for health check + warmup                                            │
│ 6. Patch ModelMetadata status:                                               │
│    - status.workers[rank].ready = true                                       │
│    - status.workers[rank].stabilityVerified = true                           │
│ 7. Controller reconciles and updates conditions                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            TARGET INSTANCE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Initialize dummy model structure                                          │
│ 2. Register target tensors with NIXL                                         │
│ 3. Watch/Poll ModelMetadata CR for status:                                   │
│    - Wait for condition: StabilityVerified=True                              │
│    - Wait for all workers: status.workers[*].ready=true                      │
│ 4. Read ConfigMaps for tensor descriptors:                                   │
│    - Get deepseek-ai-deepseek-v3-tensors-worker-{rank}                       │
│ 5. Add remote NIXL agents using status.workers[rank].nixlMetadata            │
│ 6. Execute RDMA transfers (get_zcopy)                                        │
│ 7. Run FP8 processing on received weights                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Components

#### 1. CRD Definition

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: modelmetadatas.modelexpress.nvidia.com
spec:
  group: modelexpress.nvidia.com
  versions:
    - name: v1alpha1
      served: true
      storage: true
      subresources:
        status: {}
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              required: [modelName]
              properties:
                modelName:
                  type: string
                expectedWorkers:
                  type: integer
            status:
              type: object
              properties:
                phase:
                  type: string
                  enum: [Pending, Initializing, Ready, Stale]
                workers:
                  type: array
                  items:
                    type: object
                    properties:
                      workerRank:
                        type: integer
                      nixlMetadata:
                        type: string
                        format: byte
                      tensorCount:
                        type: integer
                      ready:
                        type: boolean
                      stabilityVerified:
                        type: boolean
                      publishedAt:
                        type: string
                        format: date-time
                conditions:
                  type: array
                  items:
                    type: object
                    properties:
                      type:
                        type: string
                      status:
                        type: string
                      lastTransitionTime:
                        type: string
                        format: date-time
      additionalPrinterColumns:
        - name: Model
          type: string
          jsonPath: .spec.modelName
        - name: Phase
          type: string
          jsonPath: .status.phase
        - name: Workers
          type: integer
          jsonPath: .status.workers[*].workerRank
        - name: Age
          type: date
          jsonPath: .metadata.creationTimestamp
  scope: Namespaced
  names:
    plural: modelmetadatas
    singular: modelmetadata
    kind: ModelMetadata
    shortNames:
      - mxmeta
```

#### 2. Server Changes

The ModelExpress server will support both backends via configuration:

```rust
pub enum MetadataBackend {
    Redis { url: String },
    Kubernetes { namespace: String },
}

impl P2pStateManager {
    pub async fn publish_metadata(&self, model_name: &str, workers: Vec<WorkerMetadata>) {
        match &self.backend {
            MetadataBackend::Redis { url } => self.publish_to_redis(model_name, workers).await,
            MetadataBackend::Kubernetes { namespace } => {
                self.publish_to_crd(namespace, model_name, workers).await
            }
        }
    }
}
```

#### 3. Client Changes (Python)

```python
class MetadataClient:
    """Unified client for Redis or CRD backend."""
    
    def __init__(self):
        self.backend = os.environ.get("MX_METADATA_BACKEND", "redis")
        
    def wait_for_source_ready(self, model_name: str, worker_id: int) -> bool:
        if self.backend == "kubernetes":
            return self._wait_for_crd_ready(model_name, worker_id)
        else:
            return self._wait_for_redis_ready(model_name, worker_id)
    
    def _wait_for_crd_ready(self, model_name: str, worker_id: int) -> bool:
        """Watch ModelMetadata CR for readiness."""
        from kubernetes import client, watch
        
        api = client.CustomObjectsApi()
        w = watch.Watch()
        
        cr_name = self._sanitize_model_name(model_name)
        for event in w.stream(
            api.list_namespaced_custom_object,
            group="modelexpress.nvidia.com",
            version="v1alpha1",
            namespace=os.environ.get("POD_NAMESPACE", "default"),
            plural="modelmetadatas",
            field_selector=f"metadata.name={cr_name}",
            timeout_seconds=7200
        ):
            cr = event["object"]
            workers = cr.get("status", {}).get("workers", [])
            for w in workers:
                if w["workerRank"] == worker_id:
                    if w.get("ready") and w.get("stabilityVerified"):
                        return True
        return False
```

### Comparison: Redis vs CRD

| Aspect | Redis | CRD |
|--------|-------|-----|
| **External Dependency** | Requires Redis sidecar | None (uses K8s API) |
| **Observability** | `redis-cli KEYS "mx:*"` | `kubectl get mxmeta` |
| **Lifecycle** | TTL-based cleanup | Owner references |
| **Consistency** | Lua atomic scripts | K8s optimistic concurrency |
| **Large Data** | Native JSON | ConfigMaps (1MB limit) |
| **Scalability** | Redis replication | etcd (K8s control plane) |
| **Access Control** | Network/auth | RBAC |
| **Complexity** | Simple | Requires CRD + controller |

### Migration Path

1. **Phase 1**: Implement CRD backend alongside Redis (feature flag)
2. **Phase 2**: Validate CRD backend in staging
3. **Phase 3**: Default to CRD for new deployments
4. **Phase 4**: Deprecate Redis backend

### Configuration

```yaml
# Environment variables
MX_METADATA_BACKEND: "kubernetes"  # or "redis" (default)
MX_METADATA_NAMESPACE: "kavin"     # For CRD backend
MX_REDIS_HOST: "modelexpress-server"  # For Redis backend
MX_REDIS_PORT: "6379"
```

---

## Summary

The current Redis-based implementation provides a simple, functional solution for metadata coordination. The proposed CRD alternative offers native Kubernetes integration, better observability, and automatic lifecycle management. Both backends can coexist, allowing gradual migration based on deployment requirements.
