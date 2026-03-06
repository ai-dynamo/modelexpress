# Testing Context: PR #137 Metadata Backend Validation

## Branch
`kavink/p2p_nixl_transfers_metadata`

## What This PR Does
PR #137 adds a **layered metadata architecture** for ModelExpress P2P transfers:
- In-memory cache (always present, fast reads)
- Optional **write-through persistence** to Redis or Kubernetes CRDs
- Ready flags always in-memory (ephemeral by nature)
- Server hydrates cache from persistent backend on startup

## What's Been Done

### 1. Code Changes from Review Feedback (ALL COMPLETE)
All 17 code changes from nicolasnoble's review on PR #137 have been implemented:
- Default backend `redis` → `grpc` in `metadata_client.py`
- Removed non-existent `stability_verified` field from Python client
- `wait_for_ready()` returns `(False, None, None)` when backend unavailable
- Removed unused imports, added `re` for sanitize
- `_sanitize_model_name()` filters `[^a-z0-9\-.]` to match Rust
- `KubernetesBackend.publish_ready_signal()` uses `resourceVersion` + retries on 409
- Replaced `_log()` with `logger` in `vllm_loader.py`, added stdout StreamHandler
- Exception re-raised instead of swallowed in `_publish_metadata_to_server`
- Rust: CR create handles 409; status patch uses `resourceVersion` + retry loop
- Rust: `expected_workers` from `MX_EXPECTED_WORKERS` env or current worker count
- Rust: `addr`/`size` parsing returns errors instead of `unwrap_or(0)`
- Rust: ConfigMaps get `ownerReferences` to parent ModelMetadata CR
- Rust: `get_backend()` uses double-checked locking (read then write with recheck)
- YAMLs: namespace `kavin` → `default`, removed env-specific `storageClassName`

### 2. Images Built and Pushed
```
nvcr.io/nvidian/dynamo-dev/modelexpress-server:metadata-test
nvcr.io/nvidian/dynamo-dev/modelexpress-client:metadata-test
```
These contain ALL the code changes above.

### 3. In-Memory Backend Test (PASSED ✅)
- Namespace: `metadata`
- Model: `Qwen/Qwen2.5-0.5B` (small, TP=1)
- Server: `MX_METADATA_BACKEND=memory`
- Source published 170 tensors (0.99 GB) via gRPC, published ready
- Target received via RDMA: **87.4 Gbps**, 0.090s transfer, 17.96s total load
- Inference confirmed: "The capital of France is Paris..."

### 4. Redis Write-Through Test (IN PROGRESS)
**Status: Transfer verified, hydration test remaining**

#### What's deployed now (namespace `metadata`):
| Component | Image | Config |
|-----------|-------|--------|
| `modelexpress-server` | `...:metadata-test` | `MX_METADATA_BACKEND=redis`, Redis 7-alpine sidecar, `REDIS_URL=redis://localhost:6379` |
| `mx-source` | `...:metadata-test` | Qwen 0.5B, TP=1, `MX_SERVER_ADDRESS=modelexpress-server:8001`, gRPC backend (default), `rdma/ib: 1` |
| `mx-target` | `...:metadata-test` | Same model, `--load-format mx-target`, gRPC backend, `rdma/ib: 1` |

#### YAMLs used:
- `examples/p2p_transfer_k8s/deploy/test-redis-server.yaml` — MX server with Redis sidecar
- `examples/p2p_transfer_k8s/deploy/test-source.yaml` — Qwen 0.5B source
- `examples/p2p_transfer_k8s/deploy/test-target.yaml` — Qwen 0.5B target

#### Results so far:
- ✅ Server started with `LayeredRedis { url: "redis://localhost:6379" }`
- ✅ Server logged: `Published metadata for model 'Qwen/Qwen2.5-0.5B'` to BOTH memory and Redis
- ✅ Redis contains: `mx:models` and `mx:model:Qwen/Qwen2.5-0.5B` (170 tensors, 10529-byte NIXL blob)
- ✅ Target completed RDMA transfer: 0.075s, total load 17.17s
- ✅ Target inference works
- ⏳ **Hydration test NOT done** — need to kill MX server container (not pod!) so it restarts and re-reads from Redis

#### Hydration test steps remaining:
```bash
# 1. Kill MX server process (Redis sidecar stays alive with data)
microk8s kubectl exec -n metadata deployment/modelexpress-server -c modelexpress-server -- /bin/sh -c "kill 1"

# 2. Wait for container restart
sleep 25

# 3. Check server logs — should say "Hydrated 1 model(s)" or "Loaded metadata for model"
#    instead of "No existing metadata found"
microk8s kubectl logs -n metadata deployment/modelexpress-server -c modelexpress-server --tail=25

# 4. Confirm Redis still has data
microk8s kubectl exec -n metadata deployment/modelexpress-server -c redis -- redis-cli KEYS "mx:*"
```

### 5. CRD Write-Through Test (NOT STARTED)
After Redis hydration passes, test the Kubernetes CRD backend:

```bash
# 1. Install CRD
microk8s kubectl apply -f examples/p2p_transfer_k8s/deploy/persistence/crd-modelmetadata.yaml

# 2. Install RBAC in metadata namespace (edit namespace: metadata in the file first)
microk8s kubectl apply -f examples/p2p_transfer_k8s/deploy/persistence/rbac-modelmetadata.yaml -n metadata

# 3. Tear down current server, redeploy with MX_METADATA_BACKEND=kubernetes
#    Need a test-crd-server.yaml similar to test-redis-server.yaml but with:
#      MX_METADATA_BACKEND: "kubernetes"
#      MX_METADATA_NAMESPACE from downward API (metadata.namespace)
#      serviceAccountName: modelexpress

# 4. Restart source → verify ModelMetadata CR + ConfigMaps created:
microk8s kubectl get modelmetadatas -n metadata
microk8s kubectl get configmaps -n metadata -l modelexpress.nvidia.com/model

# 5. Start target → verify RDMA transfer + inference
# 6. Kill MX server container → verify hydration from CRD
```

## Key Files
| File | Purpose |
|------|---------|
| `modelexpress_client/python/modelexpress/metadata_client.py` | Python client: backend abstraction (gRPC, Redis, K8s) |
| `modelexpress_client/python/modelexpress/vllm_loader.py` | vLLM custom loaders (MxSource/MxTarget) |
| `modelexpress_client/python/modelexpress/client.py` | gRPC client wrapper (MxClient) |
| `modelexpress_server/src/state.rs` | Server state manager (ready flags + backend dispatch) |
| `modelexpress_server/src/metadata_backend.rs` | Backend trait + BackendConfig + create_backend() |
| `modelexpress_server/src/metadata_backend/memory.rs` | In-memory backend |
| `modelexpress_server/src/metadata_backend/redis.rs` | Redis backend |
| `modelexpress_server/src/metadata_backend/kubernetes.rs` | K8s CRD backend |
| `modelexpress_server/src/metadata_backend/layered.rs` | Layered backend (cache + persistent) |
| `modelexpress_server/src/p2p_service.rs` | gRPC service impl |
| `modelexpress_common/proto/p2p.proto` | Protobuf: PublishMetadata, GetMetadata, PublishReady, GetReady |
| `examples/p2p_transfer_k8s/deploy/persistence/` | Persistence YAMLs (Redis, CRD, RBAC) |
| `examples/p2p_transfer_k8s/deploy/test-redis-server.yaml` | Test YAML: MX server with Redis sidecar |
| `examples/p2p_transfer_k8s/deploy/test-source.yaml` | Test YAML: Qwen 0.5B source |
| `examples/p2p_transfer_k8s/deploy/test-target.yaml` | Test YAML: Qwen 0.5B target |
| `docs/metadata.md` | Full architecture doc with debugging section |
| `docs/pr137_feedback.md` | Review feedback tracking |

## Important Notes
- Source YAML requires `rdma/ib: 1` in resource limits/requests or NIXL/UCX fails with `NIXL_ERR_BACKEND`
- The `LIBFABRIC` plugin warning is expected (not installed in image, not needed — UCX is used)
- Ready flags are ALWAYS in-memory on the server regardless of backend mode
- `rollout restart` on the server deployment kills the Redis sidecar too (data lost) — to test hydration, kill only the MX server container process
- Source publishes ready via `metadata_client.get_backend().publish_ready()` which uses gRPC by default (no `MX_METADATA_BACKEND` set on source/target means gRPC)
- Secrets needed in `metadata` namespace: `hf-token-secret`, `nvcr-imagepullsecret`
