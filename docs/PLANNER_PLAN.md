# Planner + ModelExpress P2P Scaling Plan

**Status**: Ready to validate  
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

**YAML**: `deploy/gcp/kimi-planner-dgd.yaml`

1. Ensure MX source is deployed and published (`kimi-source-deploy.yaml`)
2. Deploy DGD: `kubectl -n kavin apply -f kimi-planner-dgd.yaml`
3. Verify first worker loads via P2P (~3.5s transfer + ~30s autotuning)
4. Test inference: `curl <frontend>:8000/v1/chat/completions`
5. Manually scale via DGDSA: `kubectl -n kavin scale dgdsa/kimi-planner-kimiworker --replicas=2`
6. Verify second worker also loads via P2P in ~35s
7. Measure: time from replica increase → worker ready

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

| Metric | Disk Load | P2P Load (Measured) | Target |
|--------|-----------|---------------------|--------|
| Transfer time (Kimi K2.5 TP=4) | N/A | **3.51s** | < 10s |
| Time to ready (incl. autotuning) | ~75 min | **~35s** (3.5s transfer + ~30s autotune) | < 60s |
| Scale-up responsiveness | Impractical | Planner interval + P2P | < 2 min total |
| Transfer bandwidth | N/A (disk) | **369 Gbps** (RoCE rc_mlx5) | > 200 Gbps |
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
- [x] **Kimi K2.5 TP=4 P2P transfer validated — 369 Gbps, 648 GB in 3.51s**
- [x] **Inference confirmed via Dynamo frontend**
- [ ] DGD operator webhook fixed (kube-rbac-proxy image)
- [ ] DGDSA scaling adapter tested
- [x] Planner DGD YAML created: `deploy/gcp/kimi-planner-dgd.yaml`
- [ ] Prometheus metrics collection verified
- [ ] genai-perf or load generator available for testing
