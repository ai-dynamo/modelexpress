# ModelExpress P2P for TRT-LLM on GCP GB200

GPU-to-GPU weight loading for Kimi K2.5 via ModelExpress NIXL RDMA.
Targets load weights in seconds instead of 15-24 minutes from PVC.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │     ModelExpress Server + Redis      │
                    │   (gRPC metadata, NIXL descriptors)  │
                    └────────┬────────────────┬────────────┘
                  publish    │                │   query
                  metadata   │                │   metadata
                             │                │
  ┌──────────────────────────┴───┐  ┌─────────┴───────────────────────┐
  │  MX Source (DGD, TP=8)       │  │  MX Targets (DGD)               │
  │                              │  │                                 │
  │  1. Load weights from PVC    │  │  1. Query MX server for source  │
  │  2. model.load_weights()     │  │  2. NIXL RDMA into param bufs   │
  │  3. ── PUBLISH HERE ──       │  │  3. post_load_weights()         │
  │  4. post_load_weights()      │  │  4. NCCL init + serve           │
  │  5. Serve (holds GPU mem)    │  │                                 │
  │                              │  │  Source publishes BEFORE step 4 │
  │  Node A ┌──┐┌──┐┌──┐┌──┐     │  │  so targets run same xforms     │
  │         │R0││R1││R2││R3│     │  │                                 │
  │  Node B ┌──┐┌──┐┌──┐┌──┐     │  │  ┌───────────────────────────┐  │
  │         │R4││R5││R6││R7│     │  │  │ Prefill  (TP=8, 2 nodes)  │  │
  │                              │  │  │ Decode   (TP=8, 2 nodes)  │  │
  └──────────────────────────────┘  │  │ Frontend (KV router)      │  │
               │                    │  └───────────────────────────┘  │
               │    NIXL RDMA       │                                 │
               └────────────────────►  400-457 Gbps RoCE per rank     │
                90.75 GB/rank        ─────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

```bash
# 1. Teleport auth
tsh kube login dynamo-gcp-dev-01

# 2. Dynamo platform (etcd + NATS) must be running
kubectl -n kavin get pods  # verify etcd-0, nats-0

# 3. Secrets
kubectl -n kavin get secret hf-token-secret
kubectl -n kavin get secret nvcr-imagepullsecret

# 4. ComputeDomain (creates IMEX channels for GPU allocation)
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

# 5. Shared PVC with model files
kubectl -n kavin get pvc shared-model-cache
```

---

## Aggregated Inference (P2P)

Single worker type — serves both prefill and decode. Fastest to validate.

### Step 1: Deploy infrastructure

```bash
kubectl -n kavin apply -f mx-infra-decode.yaml
# Creates: modelexpress-server-decode + redis-decode
```

### Step 2: Deploy source (loads from PVC, publishes via RDMA)

```bash
kubectl -n kavin apply -f kimi-source-decode-dgd.yaml
# TP=8 across 2 nodes, loads ~15 min from PVC
# Watch: kubectl -n kavin logs -f <source-leader-pod> | grep "published ALL"
```

### Step 3: Deploy target (receives weights via P2P)

```bash
# After source publishes:
kubectl -n kavin apply -f kimi-target-agg-tp8-dgd.yaml
# TP=8, loads via RDMA in ~2 seconds
# Watch: kubectl -n kavin logs -f <target-leader-pod> | grep "Gbps"
```

### Step 4: Test inference

```bash
FRONTEND=$(kubectl -n kavin get pod -l app.kubernetes.io/part-of=kimi-target-agg-tp8 \
  -l nvidia.com/dynamo-component=frontend -o name | head -1)
kubectl -n kavin exec $FRONTEND -- curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"baseten-admin/Kimi-2.5-text-nvfp4-v3",
       "messages":[{"role":"user","content":"What is the capital of France?"}],
       "max_tokens":50}'
```

---

## Disaggregated Inference (P2P)

Separate prefill and decode workers with KV cache transfer. Each loads
weights via P2P from the same MX source.

### Step 1: Deploy infrastructure + source

```bash
kubectl -n kavin apply -f mx-infra-decode.yaml
kubectl -n kavin apply -f kimi-source-decode-dgd.yaml
# Wait for source to publish (~15 min)
```

### Step 2: Deploy disagg targets

```bash
kubectl -n kavin apply -f kimi-disagg-mx-tp8-dgd.yaml
# Creates: Frontend (KV router) + Prefill (MX target) + Decode (MX target)
# Both load via P2P concurrently
```

### Step 3: Test inference

```bash
FRONTEND_IP=$(kubectl -n kavin get pod -l app.kubernetes.io/part-of=kimi-disagg-mx-tp8 \
  -l nvidia.com/dynamo-component=frontend -o jsonpath='{.items[0].status.podIP}')
kubectl -n kavin exec <any-worker-pod> -- curl -s http://${FRONTEND_IP}:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"baseten-admin/Kimi-2.5-text-nvfp4-v3",
       "messages":[{"role":"user","content":"What is the capital of France?"}],
       "max_tokens":50}'
```

### Disagg without MX (PVC baseline)

To validate disagg works without P2P (both workers load from PVC):

```bash
kubectl -n kavin apply -f kimi-disagg-baseline-dgd.yaml
# No MX source needed — loads directly from shared-model-cache PVC
```

---

## File Reference

### Infrastructure

| File | Purpose |
|------|---------|
| `mx-infra.yaml` | ModelExpress server + Redis (TP=4 source) |
| `mx-infra-decode.yaml` | ModelExpress server + Redis (TP=8 source) |

### Sources (load from PVC, publish for RDMA)

| File | TP | Nodes | Notes |
|------|-----|-------|-------|
| `kimi-source-decode-dgd.yaml` | 8 | 2 | Primary source for TP=8 targets |
| `kimi-source-deploy.yaml` | 4 | 1 | Legacy TP=4 source (Deployment) |
| `kimi-source-dgd.yaml` | 4 | 1 | TP=4 source (DGD format) |

### Targets (receive weights via P2P)

| File | Mode | TP | MX Source | Notes |
|------|------|-----|-----------|-------|
| `kimi-target-agg-tp8-dgd.yaml` | Aggregated | 8 | decode server | Simplest P2P test |
| `kimi-disagg-mx-tp8-dgd.yaml` | Disagg | 8+8 | decode server | Prefill + decode via P2P |
| `kimi-disagg-baseline-dgd.yaml` | Disagg | 8+8 | PVC (no MX) | Baseline for comparison |
| `kimi-disagg-mx-dgd.yaml` | Disagg | 4+4 | MX server | Legacy TP=4 disagg |
| `kimi-disagg-phase2-dgd.yaml` | Disagg | 4+8 | dual MX | Mixed TP (needs 2 sources) |
| `kimi-target-pvc-tp8-dgd.yaml` | Aggregated | 8 | PVC (no MX) | Ground truth baseline |

### Testing

| File | Purpose |
|------|---------|
| `qwen-source-deploy.yaml` | Qwen 0.5B TP=2 source (fast iteration) |
| `qwen-target-deploy.yaml` | Qwen 0.5B TP=2 target |
| `qwen-baseline-test.yaml` | Qwen baseline without MX |

---

## Required Configuration for GCP GB200

All worker pods need these settings. See `docs/disagg_trtllm.md` for details.

```yaml
securityContext:
  privileged: true              # RDMA memory registration
  runAsUser: 0                  # SSH key path fix for multinode

env:
  HOME: /root                   # Fix SSH host key path mismatch
  UCX_TLS: "self,sm,rc,cuda_copy,gdr_copy,tcp"
  UCX_IB_GID_INDEX: "3"        # GCP RoCEv2 GID selection
  TRTLLM_UCX_INTERFACE: eth0    # Prevent 169.254.x.x binding
  OMPI_MCA_pml: ob1             # Bypass UCX UD for MPI
  OMPI_MCA_btl: "tcp,self,vader"
  NATS_SERVER: "nats://dynamo-platform-nats.<ns>.svc.cluster.local:4222"
  ETCD_ENDPOINTS: "dynamo-platform-etcd.<ns>.svc.cluster.local:2379"

# Engine config (extra-engine-args YAML):
  cache_transceiver_config:
    backend: DEFAULT            # Bypass UCX UD for KV cache transfer
  enable_autotuner: false       # Avoid warmup MPI desync
```

---

## Cleanup

```bash
# Delete all workloads
kubectl -n kavin delete dgd --all

# Delete compute domain (releases IMEX channels)
kubectl -n kavin delete computedomain kavin-compute-domain

# Keep infra running for next deployment
# kubectl -n kavin delete -f mx-infra-decode.yaml  # if needed
```

---

## Building the Image

The image combines three repos into a single ARM64 container layered on the
Dynamo TRT-LLM base image.

### Repos and branches

| Repo | Branch | What it provides |
|------|--------|-----------------|
| `modelexpress` | `kavink/trtllm` | MX client, NIXL transfer, TRT-LLM patches |
| `dynamo` | `kavink/trtllm-p2p` | Engine P2P hooks (`--model-express-url`) |
| `TensorRT-LLM` | `kavink/presharded-weight-loading` | `LoadFormat.PRESHARDED` (applied via patches) |

### Directory layout

```
~/work/github/
├── modelexpress/   (kavink/trtllm branch)
└── dynamo/         (kavink/trtllm-p2p branch)
```

### Build command

```bash
cd ~/work/github/modelexpress

docker buildx build --platform linux/arm64 --no-cache \
    -f examples/p2p_transfer_trtllm/Dockerfile.ph3-gcp-gb200 \
    --build-context dynamo=../dynamo \
    -t nvcr.io/nvidian/dynamo-dev/kavink:dynamo-trtllm-mx-v1.8.0 \
    --push .
```

The Dockerfile (`examples/p2p_transfer_trtllm/Dockerfile.ph3-gcp-gb200`):
1. Starts from `karenc:dynamo-trtllm-v1.0.0-a9b6f95` (TRT-LLM 1.3.0rc5 + NIXL, ARM64)
2. Installs ModelExpress Python client (gRPC + NIXL transfer)
3. Copies Dynamo engine/worker files from `dynamo` repo via `--build-context`
4. Applies TRT-LLM patches: `PRESHARDED` LoadFormat, source publish hook, MPI allgather fix

### Building the base image from dynamo

If you don't have access to `karenc:dynamo-trtllm-v1.0.0-a9b6f95`, build the
base image from the `dynamo` repo using its rendered Dockerfile:

```bash
cd ~/work/github/dynamo

docker buildx build --platform linux/arm64 --no-cache \
    -f container/trtllm-runtime-cuda13.1-arm64-rendered.Dockerfile \
    --build-arg ARCH=arm64 \
    --build-arg ARCH_ALT=aarch64 \
    -t my-registry/dynamo-trtllm-base:latest \
    --push .
```

Then update the `FROM` line in `Dockerfile.ph3-gcp-gb200` to point to your
base image instead of `karenc:dynamo-trtllm-v1.0.0-a9b6f95`.

### Current image

```
nvcr.io/nvidian/dynamo-dev/kavink:dynamo-trtllm-mx-v1.8.0
```
