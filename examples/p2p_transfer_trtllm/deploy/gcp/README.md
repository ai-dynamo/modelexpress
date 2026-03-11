# Kimi K2.5 P2P Weight Transfer on GCP GB200

Fast GPU-to-GPU weight loading for Kimi K2.5 via ModelExpress NIXL RDMA.
New workers load weights in **1.6-3.5 seconds at 370-610 Gbps** instead of ~75 min from disk.
End-to-end disaggregated inference validated with mixed TP (prefill TP=4 + decode TP=8).

## Validated Results

| Model | Mode | TP | Transfer | Speed | Inference | Transport |
|-------|------|-----|----------|-------|-----------|-----------|
| Qwen 0.5B | Aggregated | 2 | 1.26 GB | 25-33 Gbps | Works | RoCE |
| Kimi K2.5 | Aggregated | 4 | 648 GB | **369 Gbps** | Works | RoCE |
| Kimi K2.5 | Agg + DGDSA scale | 4 | 648 GB | **371-390 Gbps** | Works | RoCE |
| Kimi K2.5 | Disagg same-TP | 4+4 | 1.3 TB | 234-538 Gbps | P2P only | RoCE |
| Kimi K2.5 | **Mixed TP disagg** | **4+8** | **1.4 TB** | **345-610 Gbps** | **Works** | **RoCE** |
| Kimi K2.5 | Mixed TP (inference) | 4+8 | — | — | TTFT 3.4s | NIXL KV |

---

## Deployment Modes

### 1. Simple P2P (source + target)

Standalone source holds weights, target receives via RDMA.

```bash
# Deploy infra
kubectl -n kavin apply -f mx-infra.yaml

# Deploy source (~75 min to load)
kubectl -n kavin exec deploy/redis -- redis-cli FLUSHALL
kubectl -n kavin apply -f kimi-source-deploy.yaml
# Watch for: "published ALL 4 workers to MX server"

# Deploy target (after source publishes)
kubectl -n kavin apply -f kimi-target-deploy.yaml
# Watch for: "Transfer complete: 1754 tensors, 162.09 GB in 3.51s (369 Gbps)"
```

### 2. Aggregated + Planner (DGD with autoscaling)

Single worker type (both prefill and decode in one engine), scaled by planner via DGDSA.

```bash
# Source must be running and published first (see above)

# Deploy aggregated DGD with planner
kubectl -n kavin apply -f kimi-planner-dgd.yaml
# Creates: Frontend + Planner + KimiWorker (MX target)

# Worker loads via P2P automatically, planner monitors load metrics

# Manual scale test
kubectl -n kavin scale dgdsa/kimi-planner-kimiworker --replicas=2
# Second worker loads via P2P in ~3.5s, ready in ~7.5 min
```

### 3. Disaggregated + Planner (prefill/decode split)

Separate prefill and decode workers, each loading via P2P, independently scaled.

```bash
# Source must be running and published first

# Deploy disagg DGD (same TP=4 for both)
kubectl -n kavin apply -f kimi-disagg-mx-dgd.yaml
# Creates: Frontend (KV router) + Planner + prefill (MX target) + decode (MX target)

# Both workers load from the same source concurrently
# Watch for transfer logs on each:
kubectl -n kavin logs -l nvidia.com/dynamo-component=prefill | grep "Transfer complete"
kubectl -n kavin logs -l nvidia.com/dynamo-component=decode  | grep "Transfer complete"

# Scale prefill independently
kubectl -n kavin scale dgdsa/kimi-disagg-prefill --replicas=2

# Scale decode independently
kubectl -n kavin scale dgdsa/kimi-disagg-decode --replicas=2
```

### 4. Mixed TP Disagg (Phase 2 — in progress)

Prefill TP=4 + Decode TP=8 with separate MX sources per TP size.

```bash
# Deploy dual MX infrastructure
kubectl -n kavin apply -f mx-infra.yaml          # prefill MX server
kubectl -n kavin apply -f mx-infra-decode.yaml    # decode MX server

# Deploy sources (both load in parallel, ~75 min each)
kubectl -n kavin apply -f kimi-source-deploy.yaml            # TP=4 prefill source
kubectl -n kavin apply -f kimi-source-decode-dgd.yaml        # TP=8 decode source (multinode)

# After both sources publish, deploy disagg DGD
kubectl -n kavin apply -f kimi-disagg-phase2-dgd.yaml
# Prefill reads from modelexpress-server (TP=4 source)
# Decode reads from modelexpress-server-decode (TP=8 source)

# Scale decode (TP=8, multinode: 2 nodes per replica)
kubectl -n kavin scale dgdsa/kimi-disagg-p2-decode --replicas=2
```

---

## File Reference

### Infrastructure

| File | Purpose |
|------|---------|
| `mx-infra.yaml` | ModelExpress server + Redis (prefill source) |
| `mx-infra-decode.yaml` | Second MX server + Redis (decode source, Phase 2) |

### Sources (hold weights for RDMA)

| File | TP | Type | Notes |
|------|-----|------|-------|
| `kimi-source-deploy.yaml` | 4 | Deployment | Single node, proven |
| `kimi-source-dgd.yaml` | 4 | DGD | With frontend, for operator |
| `kimi-source-decode-dgd.yaml` | 8 | DGD (multinode: 2) | Phase 2, validated |
| `kimi-source.yaml` | 4 | Standalone MPI | Legacy, no Dynamo engine |

### Targets / Workers (receive weights via P2P)

| File | Mode | Workers | Planner | Scaling |
|------|------|---------|---------|---------|
| `kimi-target-deploy.yaml` | Simple | 1 aggregated | No | Manual |
| `kimi-planner-dgd.yaml` | Aggregated DGD | 1 aggregated | Yes (agg) | DGDSA |
| `kimi-disagg-mx-dgd.yaml` | Disagg DGD (same TP) | prefill + decode (TP=4) | Yes (disagg) | 2x DGDSA |
| `kimi-disagg-phase2-dgd.yaml` | Disagg DGD (mixed TP) | prefill TP=4 + decode TP=8 | Yes (disagg) | 2x DGDSA |

### Testing

| File | Purpose |
|------|---------|
| `qwen-source-deploy.yaml` | Qwen 0.5B TP=2 source (fast iteration, loads in 30s) |
| `qwen-target-deploy.yaml` | Qwen 0.5B TP=2 target |

---

## How Scaling Works

```
Source (separate Deployment)           DGD (scaled by planner/DGDSA)
┌──────────────────────┐              ┌───────────────────────────┐
│ kimi-source-deploy   │              │ Frontend (KV router)      │
│ --model-express-role │  NIXL RDMA   │ Planner (load-based)      │
│   source             │◄────────────│ Worker (--model-express-   │
│ Holds 648 GB in GPU  │              │   role target)            │
│ Publishes to MX srv  │              │ DGDSA scales replicas     │
└──────────────────────┘              └───────────────────────────┘

kubectl scale dgdsa/<name> --replicas=2
  → Operator creates new pod with same spec (target role baked in)
  → New pod queries MX server → NIXL RDMA from source → 3.5s
  → Autotuning → ready to serve
  → Frontend discovers new worker via NATS
```

The source and workers are independent resources. The source is always 1 replica
holding weights. Workers scale 1..N, each loading via P2P from the same source.

For disagg mode, prefill and decode have separate DGDSAs and scale independently.

---

## Required Pod Config for GB200

```yaml
securityContext:
  privileged: true                    # for /dev/infiniband access

env:
  UCX_TLS: "rc_v,rc_x,rc,dc_x,dc,cuda_copy,tcp"   # NO cuda_ipc
  OMPI_MCA_pml: "ob1"                               # avoid UCX UD timeout
  OMPI_MCA_btl: "tcp,self,vader"                     # MPI over TCP+shmem

volumes:
  # /dev/shm auto-injected by DGD operator — do NOT add in DGD specs
  # For plain Deployments: emptyDir (100Gi, Memory)
  /dev/infiniband: hostPath                          # RoCE devices

resourceClaims:
  compute-domain-channel                             # GPU allocation via DRA

affinity:
  topologyKey: nvidia.com/gpu.clique                 # same NVLink/RoCE domain
```

## Key Findings

- **No `cuda_ipc` in UCX_TLS** — `cuIpcOpenMemHandle` fails on GB200. Remove to use host-staged RoCE.
- **`/dev/shm`** — NCCL needs >64MB. DGD operator auto-injects it; plain Deployments must add it manually.
- **`ob1` PML** — Required for TP>=4 with privileged on GB200. UCX UD times out during MPI bootstrap.
- **ComputeDomain** — Required for IMEX channels. Without it, NIXL `loadRemoteMD` fails.
- **DGD `/dev/shm` conflict** — Do NOT define `/dev/shm` in DGD extraPodSpec (operator injects it, causes duplicate mount error).
- **Multinode SSH** — DGD multinode worker needs `HOME=/root` if image has non-root user, or sshd can't find host keys.

## Prerequisites

```bash
# 1. Namespace with Dynamo platform
kubectl -n kavin get pods  # etcd + nats running

# 2. Secrets
kubectl -n kavin get secret hf-token-secret
kubectl -n kavin get secret nvcr-imagepullsecret

# 3. ComputeDomain
cat <<EOF | kubectl -n kavin apply -f -
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: kavin-compute-domain
spec:
  numNodes: 0
  channel:
    resourceClaimTemplate:
      name: kavin-compute-domain-channel
EOF

# 4. Teleport auth
tsh kube login dynamo-gcp-dev-01
```

## Performance Summary

### Weight Loading: Disk vs P2P

| Metric | From Disk (PVC) | P2P RDMA | Speedup |
|--------|----------------|----------|---------|
| Prefill TP=4 weight load | ~68 min | **3.4s** | 1,200x |
| Decode TP=8 weight load | ~31 min | **1.6s** | 1,160x |
| Total startup (incl. autotuning) | ~68 min | **~90s** | 45x |

### Inference (Mixed TP Disagg)

| Metric | Value |
|--------|-------|
| TTFT (time to first token) | 3.4s |
| Total time (8 tokens) | 4.1s |
| KV cache backend | NIXL |
| Prefill worker | TP=4 (single node) |
| Decode worker | TP=8 (2 nodes, multinode MPI) |

### Key Fix: `safe_allgather`

The `safe_allgather` patch in TRT-LLM's `communicator.py` chunks MPI allgather
messages into 64KB pieces, fixing `MPI_ERR_TRUNCATE` with `ob1` TCP BTL on GB200.
Applied at build time via `trtllm_patches/v1.3.0rc5/patch_tp_allgather.py`.

## Image

```
nvcr.io/nvidian/dynamo-dev/kavink:dynamo-trtllm-mx-v1.6.0
```

## Branches

- **modelexpress:** `kavink/trtllm` on `github.com:ai-dynamo/modelexpress`
- **dynamo:** `kavink/trtllm-p2p` on `github.com:ai-dynamo/dynamo`

## Docs

- [TRTLLM_DYNAMO_PHASE2_5.md](../../../docs/TRTLLM_DYNAMO_PHASE2_5.md) — Phase 2.5 aggregated P2P
- [TRTLLM_PHASE_3.md](../../../docs/TRTLLM_PHASE_3.md) — Phase 3 disagg + mixed TP
- [PLANNER_PLAN.md](../../../docs/PLANNER_PLAN.md) — Planner integration and scaling
- [disagg_trtllm.md](../../../docs/disagg_trtllm.md) — Disagg architecture and mixed TP design
- [disagg_inference_issues.md](../../../docs/disagg_inference_issues.md) — Multinode inference issues + fixes
