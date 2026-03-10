# Planner + ModelExpress P2P Scaling Plan

**Status**: Planning  
**Date**: March 9, 2026  
**Prerequisite**: Kimi K2.5 P2P transfer validated via Dynamo engine (Phase 2.5b)

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
5. **For Kimi K2.5: ~30 seconds at RoCE speeds** (estimated from Qwen results)

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

```yaml
Planner:
  componentType: planner
  extraPodSpec:
    mainContainer:
      args:
        - --sla-planner-config
        - /config/planner.yaml
      env:
        - name: PROMETHEUS_URL
          value: "http://prometheus:9090"
```

### Planner config

```yaml
# planner.yaml
prefill_ttft_sla_ms: 3000
decode_itl_sla_ms: 30
throughput_adjustment_interval: 30    # shorter with P2P (no 75 min wait)
load_adjustment_interval: 5
min_prefill_replicas: 1
max_prefill_replicas: 8
min_decode_replicas: 1
max_decode_replicas: 8
prediction_method: constant
scaling_connector: kubernetes
```

---

## 7. Validation Steps

### Phase 1: Manual scaling test

1. Deploy MX source (publishes Kimi K2.5 weights)
2. Deploy DGD with 1 target worker (loads via P2P)
3. Verify worker loads in ~30s via RoCE
4. Manually patch DGD to `replicas: 2`
5. Verify second worker also loads via P2P in ~30s
6. Measure: time from replica increase → worker ready

### Phase 2: Planner-driven scaling

1. Deploy full DGD with planner + MX source + 1 target worker
2. Send load (e.g. via `genai-perf` or `curl`)
3. Observe planner scaling up workers
4. Verify new workers load via P2P (check per-rank logs)
5. Measure: end-to-end latency from planner decision → worker ready → serving traffic

### Phase 3: Scale-down behavior

1. Reduce load
2. Observe planner scaling down workers
3. Verify graceful shutdown (in-flight requests complete)
4. Verify MX source remains stable (GPU memory held)

---

## 8. Key Metrics to Track

| Metric | Disk Load | P2P Load | Target |
|--------|-----------|----------|--------|
| Time to ready (Kimi K2.5 TP=4) | ~75 min | ~30s (est.) | < 60s |
| Scale-up responsiveness | Impractical | Planner interval + P2P | < 2 min total |
| Transfer bandwidth | N/A (disk) | 25-33 Gbps (RoCE) | > 20 Gbps |
| GPU memory overhead (source) | 0 | 648 GB (4x 162 GB) | Acceptable |

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
- [ ] Kimi K2.5 TP=4 P2P transfer validated at scale
- [ ] DGD operator webhook fixed (kube-rbac-proxy image)
- [ ] DGDSA scaling adapter tested
- [ ] Planner example with ModelExpress args
- [ ] Prometheus metrics collection verified
- [ ] genai-perf or load generator available for testing
