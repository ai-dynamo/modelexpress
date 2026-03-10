# Planner + ModelExpress P2P Scaling Plan

**Status**: Phase 1 (manual DGDSA scaling) VALIDATED  
**Date**: March 10, 2026  
**Prerequisite**: Kimi K2.5 P2P transfer validated — **369 Gbps RoCE, 648 GB in 3.51s** (Phase 2.5b)

---

## 1. Goal

Validate a workflow where the **Dynamo SLA Planner** automatically scales TRT-LLM worker replicas, and new replicas load weights via **ModelExpress NIXL RDMA** (seconds) instead of from disk (~75 minutes for Kimi K2.5).

---

## 2. Planner Architecture

The Dynamo SLA Planner is an autoscaling controller that runs as a service inside a DGD deployment.

### Scaling Modes

| Mode | Trigger Interval | Data Source | Use Case |
|------|-----------------|-------------|----------|
| Throughput-based | 60-180s | Prometheus (TTFT, ITL, request rate, ISL, OSL) | Stable workloads |
| Load-based | 5-10s | Router `/metrics` (active prefill tokens, KV blocks) | Bursty traffic |

### Flow

```
Prometheus metrics → Planner → correction factors → load prediction
→ replica calculation → connector.set_component_replicas()
→ DGDSA Scale or DGD patch → operator reconciles → new pods
```

### Connectors

| Connector | Scaling mechanism |
|-----------|------------------|
| KubernetesConnector | DGDSA Scale subresource (preferred) or DGD patch |
| GlobalPlannerConnector | Delegates to remote GlobalPlanner |
| VirtualConnector | Writes to runtime; external system executes |

---

## 3. Current Scaling (Disk Load)

When the planner decides to scale up:

1. Planner calls `set_component_replicas(N)` via KubernetesConnector
2. Connector patches DGDSA (if enabled) or DGD `replicas` field
3. DGD operator reconciles → creates new pod(s)
4. New pod starts → loads model from disk (PVC/HF cache)
5. **For Kimi K2.5: ~75 minutes** until the pod is ready to serve

This latency makes autoscaling impractical for large models.

---

## 4. P2P Scaling via ModelExpress

With ModelExpress, the scale-up path becomes:

1. Planner calls `set_component_replicas(N)` (same as before)
2. Connector patches DGDSA/DGD (same)
3. DGD operator creates new pod with `--model-express-role target` in spec
4. New pod starts → queries ModelExpress server → NIXL RDMA from source
5. **For Kimi K2.5: ~3.5 seconds transfer + ~30s autotuning** (validated March 10)

**No planner code changes are required.** The load path is determined by the DGD worker spec — if it includes `--model-express-url` and `--model-express-role target`, the new pod uses P2P automatically.

---

## 5. Architecture

### Deployment Topology

```
┌─────────────────────────────────────┐
│ ModelExpress Infrastructure          │
│  • modelexpress-server (gRPC)        │
│  • Redis (metadata store)            │
│  • ComputeDomain (IMEX channels)     │
└──────────────┬──────────────────────┘
               │ gRPC
┌──────────────┴──────────────────────┐
│ MX Source (dedicated deployment)      │
│  • Loads Kimi K2.5 from disk (~75m)   │
│  • Publishes via NIXL                 │
│  • Holds GPU memory for RDMA reads    │
│  • --model-express-role source        │
└──────────────┬──────────────────────┘
               │ NIXL RDMA (RoCE)
┌──────────────┴──────────────────────┐
│ DGD Workers (scaled by planner)       │
│  • --model-express-role target        │
│  • Load via MxLiveCheckpointLoader    │
│  • Ready in ~30s                      │
│  • Serve inference via Dynamo         │
│  • Replicas: 1..N (planner-managed)   │
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│ Planner                               │
│  • Monitors Prometheus metrics         │
│  • Adjusts worker replicas             │
│  • connector → DGDSA → DGD → pods     │
└─────────────────────────────────────┘
```

### Source vs Target Roles

| Aspect | MX Source | DGD Workers (Targets) |
|--------|----------|----------------------|
| Deployment | Separate (plain Deployment or DGD) | Part of the DGD scaled by planner |
| Load path | Disk → GPU (~75 min) | RDMA from source (~30s) |
| Serves inference | Optional (can serve while holding weights) | Yes |
| Scaled by planner | No (always 1 replica) | Yes (1..N replicas) |
| Role flag | `--model-express-role source` | `--model-express-role target` |

---

## 6. DGD Configuration for Planner + MX

### Worker service spec (target)

```yaml
agg:  # or TRTLLMPrefillWorker/TRTLLMDecodeWorker for disagg
  componentType: worker
  scalingAdapter:
    enabled: true    # enables DGDSA for planner scaling
  extraPodSpec:
    mainContainer:
      args:
        - --model-path
        - baseten-admin/Kimi-2.5-text-nvfp4-v3
        - --model-express-url
        - modelexpress-server:8001
        - --model-express-role
        - target
      env:
        - name: MODEL_EXPRESS_URL
          value: "modelexpress-server:8001"
        - name: MODEL_NAME
          value: "baseten-admin/Kimi-2.5-text-nvfp4-v3"
        - name: WORLD_SIZE
          value: "4"
        - name: UCX_TLS
          value: "rc_v,rc_x,rc,dc_x,dc,cuda_copy,tcp"  # NO cuda_ipc
      securityContext:
        privileged: true
      volumeMounts:
        - mountPath: /dev/shm
          name: shm
        - mountPath: /dev/infiniband
          name: infiniband
    volumes:
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 100Gi
      - name: infiniband
        hostPath:
          path: /dev/infiniband
    resourceClaims:
      - name: compute-domain-channel
        resourceClaimTemplateName: kavin-compute-domain-channel
  replicas: 1          # initial; planner adjusts dynamically
  resources:
    limits:
      gpu: "4"
    claims:
      - name: compute-domain-channel
```

### Planner service spec

The agg planner uses **load-based scaling only** (throughput-based is not supported in agg mode).
It pulls per-worker metrics from the frontend's built-in `/metrics` endpoint — **no external Prometheus deployment required**.

The worker must have `--publish-events-and-metrics` and `--request-plane nats` to push KV/load events
via NATS. The frontend must also have `--request-plane nats` to subscribe.

```yaml
Planner:
  componentType: planner
  extraPodSpec:
    mainContainer:
      command: [python3]
      args:
        - -m
        - dynamo.planner
        - --config
        - '{"environment": "kubernetes", "backend": "trtllm", "mode": "agg",
           "enable_throughput_scaling": false, "enable_load_scaling": true,
           "throughput_adjustment_interval": 30, "ttft": 3000, "itl": 30}'
```

Worker must include (in addition to MX target flags):

```yaml
args:
  - --publish-events-and-metrics
  - --request-plane
  - nats
```

Worker service must have `subComponentType: decode` for planner validation.

---

## 7. Validation Steps

### Phase 1: Manual scaling test — VALIDATED (March 10, 2026)

**YAML**: `deploy/gcp/kimi-planner-dgd.yaml`

**Results:**

| Event | Timestamp | Duration |
|-------|-----------|----------|
| DGDSA patched (replicas: 1→2) | 04:58:59Z | — |
| Second worker pod created | ~04:59:20Z | ~20s (scheduling) |
| P2P transfer complete (648 GB) | 05:00:48Z | **3.3-3.5s** |
| Second worker 1/1 Ready | ~05:06:30Z | ~6 min (autotuning + CUDA graphs) |
| **Total: scale → ready** | | **~7.5 min** |

**Transfer speeds (second worker, concurrent with first):**

| Rank | Data | Time | Speed |
|------|------|------|-------|
| 0 | 162.09 GB | 3.32s | 390.5 Gbps |
| 1 | 162.09 GB | 3.32s | 390.3 Gbps |
| 2 | 162.09 GB | 3.48s | 372.5 Gbps |
| 3 | 162.09 GB | 3.50s | 371.0 Gbps |

Both workers serve inference through the DGD frontend (round-robin routing).
Compare to **~75 min** loading from disk — **10x faster total, 180x faster transfer**.

### Phase 2: Planner-driven scaling

1. Send sustained load via `genai-perf` or `curl` loop
2. Observe planner scaling up workers (check planner pod logs)
3. Verify new workers load via P2P (check per-rank transfer logs)
4. Measure: end-to-end latency from planner decision → worker ready → serving traffic
5. Expected: < 2 min total (planner interval 30s + P2P 3.5s + autotune 30s)

### Phase 3: Scale-down behavior

1. Reduce load
2. Observe planner scaling down workers
3. Verify graceful shutdown (in-flight requests complete, `terminationGracePeriodSeconds: 600`)
4. Verify MX source remains stable (GPU memory held)

### Deployment commands

```bash
# 1. Ensure source is published
kubectl -n kavin get pods -l app=kimi-source-deploy
kubectl -n kavin exec deploy/redis -- redis-cli GET 'mx:model:baseten-admin/Kimi-2.5-text-nvfp4-v3'

# 2. Deploy planner DGD
kubectl -n kavin apply -f deploy/gcp/kimi-planner-dgd.yaml

# 3. Watch pods come up
watch kubectl -n kavin get pods -l app.kubernetes.io/part-of=kimi-planner

# 4. Check worker P2P transfer logs
kubectl -n kavin logs -l app.kubernetes.io/part-of=kimi-planner,nvidia.com/dynamo-component=KimiWorker | grep -i transfer

# 5. Test inference
FRONTEND=$(kubectl -n kavin get pod -l nvidia.com/dynamo-component=Frontend -o jsonpath='{.items[0].status.podIP}')
curl http://$FRONTEND:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"baseten-admin/Kimi-2.5-text-nvfp4-v3","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'

# 6. Manual scale test (Phase 1)
kubectl -n kavin scale dgdsa/kimi-planner-kimiworker --replicas=2

# 7. Check planner decisions
kubectl -n kavin logs -l nvidia.com/dynamo-component=Planner --tail=50
```

---

## 8. Key Metrics to Track

| Metric | Disk Load | P2P Load (Measured) | DGDSA Scale Test |
|--------|-----------|---------------------|------------------|
| Transfer time (Kimi K2.5 TP=4) | N/A | **3.51s** | **3.3-3.5s** (concurrent) |
| Time to ready (incl. autotuning) | ~75 min | ~7.5 min (autotune dominates) | **7.5 min** |
| Transfer bandwidth | N/A (disk) | **369 Gbps** (RoCE) | **371-390 Gbps** |
| Scale command → ready | N/A | — | **7.5 min** |
| GPU memory overhead (source) | 0 | 648 GB (4x 162 GB) | Same |
| Concurrent P2P from same source | N/A | — | **Works** |

---

## 9. Open Questions

1. **Source lifecycle**: Should the MX source be part of the DGD or a separate deployment? If separate, who manages it?
2. **Source redundancy**: What happens if the source pod dies? Should we have 2 sources for HA?
3. **Multi-model**: Can one MX server coordinate P2P for multiple models simultaneously?
4. **Scale-to-zero**: If planner scales targets to 0, the source still holds GPU memory. How to handle this?
5. **Planner awareness**: Should the planner know about P2P load time (faster scaling decisions)?

---

## 10. Prerequisites

- [x] ModelExpress P2P transfer validated (Qwen TP=2 at 25-33 Gbps RoCE)
- [x] **Kimi K2.5 TP=4 P2P transfer validated — 369 Gbps, 648 GB in 3.51s**
- [x] **Inference confirmed via Dynamo frontend**
- [x] DGD operator webhook working (`dynamo-system` namespace, kube-rbac-proxy fix applied)
- [x] **DGDSA scaling adapter tested — scale 1→2 with P2P, both workers serving**
- [x] Planner DGD YAML created: `deploy/gcp/kimi-planner-dgd.yaml`
- [x] Planner initializes, validates deployment, discovers frontend metrics URL
- [ ] Planner per-worker load metrics flowing (blocked — see §11)
- [ ] Planner-driven auto-scaling end-to-end (Phase 2)
- [ ] genai-perf or load generator for sustained traffic test

---

## 11. Bugs Found During Planner Integration

### 11.1 `AggPlanner` crash: `component_type` not set (FIXED)

`BasePlanner.__init__()` accesses `self.component_type` before `AggPlanner` sets it.
`PrefillPlanner`/`DecodePlanner` define it as a class attribute; `AggPlanner` creates a
bare `BasePlanner` and sets it post-init — too late.

**Fix** (pushed to `kavink/trtllm-p2p`): Add `component_type` parameter to `BasePlanner.__init__()`,
pass `SubComponentType.DECODE` from `AggPlanner`. See `planner_core.py` and `agg_planner.py`.

**Note**: `agg_planner.py` is main-only — does not exist in `release/0.9.1`.

### 11.2 Planner `validate_deployment` requires `subComponentType: decode`

The `KubernetesConnector.validate_deployment()` looks for a DGD service with
`subComponentType: decode`. Our worker must set `subComponentType: decode` even in
aggregated mode.

### 11.3 Per-worker load metrics not flowing

The agg planner's `DirectRouterMetricsClient` fetches from the frontend `/metrics` endpoint.
The frontend exposes per-worker metrics (`worker_active_decode_blocks`, etc.) via `KvWorkerMonitor`,
which subscribes to NATS KV events from workers. **No external Prometheus is needed.**

Requirements for metrics to flow:
- Worker: `--publish-events-and-metrics` + `--request-plane nats`
- Frontend: `--request-plane nats`
- Both connected to the same NATS server

**Status**: NATS EventPublisher registered on worker, EventSubscriber registered on frontend,
but per-worker gauges still not appearing on `/metrics` after inference traffic. Needs debugging
with planner team — may be a timing or event format issue in the agg path.

### 11.4 `wait_for_graph_deployment_ready` stuck after planner restart

After deleting and recreating the planner pod, the `wait_for_graph_deployment_ready` loop
sees `planner: desired=1, updated=2` due to stale PodCliqueSet revision. Blocks the planner
from entering the load loop. Resolves eventually but can take 30+ minutes.

### 11.5 `/dev/shm` duplicate mount in DGD

The DGD operator auto-injects `/dev/shm` emptyDir volume. If the extraPodSpec also defines
`/dev/shm`, the PodCliqueSet creation fails with `Duplicate value: {"mountPath":"/dev/shm"}`.
Worker DGD specs must NOT include `/dev/shm` — the operator handles it.
