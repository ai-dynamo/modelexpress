# Operator & LLM Integration Improvements for ModelExpress P2P

Lessons learned from integrating ModelExpress P2P (NIXL-based weight transfer) with TRT-LLM via the Dynamo operator on GCP GB200.

---

## Context

ModelExpress P2P enables live GPU-to-GPU weight transfer between a **source** (model loaded from disk, weights registered with NIXL) and a **target** (receives weights via RDMA, skipping disk I/O). The source and target run as separate pods managed by the Dynamo operator via `DynamoGraphDeployment` CRDs.

With TRT-LLM's PyTorch backend and TP>1, the `LLM()` wrapper spawns MPI worker processes internally. The orchestrator process holds a `GenerationExecutorProxy` — it has no access to the actual torch model. This architecture mismatch caused multiple issues during integration.

---

## P0: Critical — Would have prevented all bugs encountered

### 1. MX Server: `PublishMetadata` should merge workers, not replace

**Problem:** When each MPI worker calls `PublishMetadata` independently with its own rank, the MX server's Redis backend replaces the entire model record. Last writer wins — only 1 of 4 ranks persists.

**Current workaround:** Gather all workers on MPI rank 0, publish once.

**Proposed fix:** The Redis backend should atomically merge workers by rank:

```lua
-- Lua script for atomic worker merge
local existing = redis.call('GET', key)
local record = existing and cjson.decode(existing) or {model_name=model_name, workers={}}
for _, new_worker in ipairs(new_workers) do
    local found = false
    for i, w in ipairs(record.workers) do
        if w.worker_rank == new_worker.worker_rank then
            record.workers[i] = new_worker
            found = true
            break
        end
    end
    if not found then table.insert(record.workers, new_worker) end
end
redis.call('SET', key, cjson.encode(record))
```

This would make per-worker publish safe without requiring MPI coordination, and would support heterogeneous deployments where ranks start at different times.

### 2. TRT-LLM: Expose a post-init hook for worker processes

**Problem:** `LLM()` spawns MPI workers internally via `MpiPoolSession`. There is no official API to run code inside each worker after `setup_engine()` completes. We had to patch `worker_main()` in `tensorrt_llm/executor/worker.py` directly.

**Proposed fix:** Add a `worker_post_init_fn` parameter to `LLM()`:

```python
def my_post_init(worker):
    # Called in each worker process after setup_engine()
    from modelexpress.trtllm_live_transfer import publish_from_worker
    publish_from_worker(worker)

llm = LLM(
    model="meta-llama/Llama-3.1-70B",
    worker_post_init_fn=my_post_init,
)
```

**Alternative:** Support `collective_rpc` for the MPI/IPC executor (currently only `RayExecutor` implements it). This would let the orchestrator trigger actions on all workers without patching internals.

---

## P1: High — Prevents common deployment errors

### 3. Operator: Auto-inject ModelExpress env vars

**Problem:** When `--model-express-role source` is set, workers need `MODEL_EXPRESS_SOURCE=1`, `WORLD_SIZE`, `MODEL_EXPRESS_URL`, and `MODEL_NAME` in their environment. These must be in the pod spec (YAML) because MPI workers inherit env at spawn time — `os.environ` changes in the orchestrator after `import tensorrt_llm` don't propagate.

Users have to manually add these env vars, which is error-prone and undiscoverable.

**Proposed fix:** The operator should auto-inject env vars when it sees `--model-express-role`:

```yaml
# User writes:
args:
  - --model-express-role
  - source
  - --model-express-url
  - modelexpress-server:8001
  - --model-path
  - meta-llama/Llama-3.1-70B

# Operator auto-injects into pod env:
env:
  - name: MODEL_EXPRESS_SOURCE
    value: "1"
  - name: MODEL_EXPRESS_URL
    value: "modelexpress-server:8001"
  - name: MODEL_NAME
    value: "meta-llama/Llama-3.1-70B"
  - name: WORLD_SIZE
    value: "4"  # derived from resources.limits.gpu
```

### 4. Operator: Use accessible image registry for kube-rbac-proxy

**Problem:** The operator deployment uses `gcr.io/kubebuilder/kube-rbac-proxy:v0.15.0` as a sidecar. This image is inaccessible from GCP clusters, causing `ErrImagePull` on every operator pod restart. When the operator is down, its webhook is unavailable, blocking ALL DGD operations cluster-wide.

We had to manually run `kubectl set image` to switch to `registry.k8s.io/kubebuilder/kube-rbac-proxy:v0.15.0` every time the operator restarted.

**Proposed fix:** Change the operator Helm chart / deployment to use `registry.k8s.io/kubebuilder/kube-rbac-proxy:v0.15.0` (the official Kubernetes mirror) as the default.

---

## P2: Medium — Improves observability and resilience

### 5. Operator: Separate liveness from readiness for long-loading models

**Problem:** The source pod takes ~75 minutes to load a 162 GB model (Kimi K2.5 NVFP4, EP=4). During this time, `/live` returns 503. The startup probe needs `failureThreshold: 120 × periodSeconds: 60 = 2 hours` to avoid killing the pod. This is fragile and provides no visibility into loading progress.

**Proposed fix:** Add distinct endpoints:
- `/healthz` — process is alive (always 200 after init)
- `/live` — model is loaded and engine is ready
- `/ready` — model is published to MX server and can serve inference

The operator could use `/healthz` for liveness and `/ready` for readiness, with the startup probe on `/live`.

### 6. Operator: Webhook failure policy should be `Ignore`, not `Fail`

**Problem:** When the operator's webhook service is unavailable (e.g., sidecar image pull failure), all DGD create/update/delete operations fail with `no endpoints available for service`. This locks out all users in the cluster.

**Proposed fix:** Set the validating webhook's `failurePolicy: Ignore` so operations proceed when the webhook is down. Alternatively, make the webhook optional and deployable separately.

---

## P3: Nice to have — Cleaner architecture

### 7. Operator: Support source-only mode without full Dynamo stack

**Problem:** The source only needs to load weights and publish them via NIXL. It doesn't serve inference requests and doesn't need NATS, etcd, or a frontend pod. But `DynamoGraphDeployment` requires the full platform (etcd, NATS) and creates a frontend pod.

**Proposed fix:** Support a `source-only` component type that:
- Skips frontend pod creation
- Doesn't require NATS/etcd
- Only runs the TRT-LLM engine with `--model-express-role source`
- Exits or sleeps after publishing (holding GPU memory for RDMA)

### 8. Operator: Document plain Deployment equivalents

**Problem:** When the operator is broken, there's no way to deploy workloads. Users need a fallback.

**Proposed fix:** For each DGD pattern, provide a documented plain Kubernetes Deployment equivalent. We created `kimi-source-deploy.yaml` as a manual fallback — this pattern should be formalized.

---

## Appendix: Architecture mismatch summary

| Aspect | Standalone MPI source | DGD (Dynamo engine) |
|--------|----------------------|---------------------|
| Who loads model | Script directly | TRT-LLM MPI worker processes |
| Orchestrator has model? | N/A | No — only `GenerationExecutorProxy` |
| MPI setup | `mpirun -np 4` | `MpiPoolSession` spawns workers |
| Env propagation | All ranks inherit from `mpirun` | Must be in pod spec (YAML) |
| Publish from | Rank 0 after `comm.gather` | Must be from workers, not orchestrator |
| gRPC publish | 1 call with all workers | Requires MPI gather in workers (or MX merge) |

The standalone MPI source works because the script IS the worker — it has direct access to the model, MPI communicator, and NIXL. The DGD source adds an orchestrator layer (`LLM()` → `GenerationExecutorProxy` → MPI workers) that prevents direct model access from the process that manages the lifecycle.

The two P0 items (MX server merge + TRT-LLM worker hook) would eliminate this mismatch entirely.
