# ModelExpress P2P for TRT-LLM on GCP GB200

GPU-to-GPU weight loading for TRT-LLM via ModelExpress NIXL RDMA.
Targets load weights in ~2 seconds instead of 15-20 minutes from disk.

**Validated:** Kimi K2.5 TP=8, 16 target ranks × 90.75 GB at 363–506 Gbps (RoCE).

## How It Works

```
                    ┌─────────────────────────────────────┐
                    │     ModelExpress Server + Redis      │
                    │   (gRPC metadata, NIXL descriptors)  │
                    └────────┬────────────────┬────────────┘
                  publish    │                │   query
                  metadata   │                │   metadata
                             │                │
  ┌──────────────────────────┴───┐  ┌─────────┴───────────────────────┐
  │  Source (TP=8, 2 nodes)      │  │  Targets (prefill + decode)     │
  │                              │  │                                 │
  │  1. checkpoint_format="MX"   │  │  1. checkpoint_format="MX"      │
  │  2. No MX sources found      │  │  2. MX sources detected         │
  │  3. Fall back to disk load   │  │  3. MxLiveWeightLoader RDMA     │
  │  4. publish_as_source()      │  │  4. _weights_presharded=True    │
  │  5. post_load_weights()      │  │  5. post_load_weights()         │
  │  6. Serve (holds GPU mem)    │  │  6. Serve                       │
  └──────────────────────────────┘  └─────────────────────────────────┘
               │                                    ▲
               └────── NIXL RDMA (RoCE) ────────────┘
                  ~2s, 363-506 Gbps per rank
```

The integration uses TRT-LLM's `@register_checkpoint_loader("MX")` architecture
([TRT-LLM PR #13531](https://github.com/NVIDIA/TensorRT-LLM/pull/13531)):
- **Source** auto-detects no existing MX sources → loads from disk → publishes via `publish_model_params()`
- **Target** auto-detects existing sources → `MxLiveWeightLoader` does RDMA → marks modules `_weights_presharded`

## Quick Start

### Prerequisites

- GKE cluster with GB200 nodes (ARM64) and CPU nodes
- Dynamo platform (etcd + NATS) running in your namespace
- `nvcr-imagepullsecret` and `hf-token-secret` in your namespace
- `shared-model-cache` PVC with the model downloaded
- ComputeDomain created for your namespace

### Step 1: Deploy MX infrastructure

```bash
kubectl -n <namespace> apply -f mx-infra-decode.yaml
# Creates: modelexpress-server-decode + redis-decode
# Verify: kubectl -n <namespace> get pods -l 'app in (redis-decode, modelexpress-server-decode)'
```

### Step 2: Deploy source (loads from disk, publishes weights)

```bash
kubectl -n <namespace> apply -f kimi-source-decode-dgd.yaml
# TP=8 across 2 nodes, loads ~15-20 min from disk
```

Wait for all 8 workers to publish:
```bash
kubectl exec -n <namespace> deploy/redis-decode -- redis-cli KEYS 'mx:source:*:*' | wc -l
# Should output: 8
```

### Step 3: Deploy targets (receive weights via RDMA)

```bash
kubectl -n <namespace> apply -f kimi-disagg-mx-tp8-dgd.yaml
# Creates: Frontend + Prefill (TP=8) + Decode (TP=8)
# Both load via RDMA concurrently (~2s per rank)
```

### Step 4: Verify RDMA transfer

Check per-rank transfer logs:
```bash
kubectl exec -n <namespace> <target-worker-pod> -- cat /tmp/mx_logs/rank0.log
# Look for: "Transfer complete: 1815 tensors, 90.75 GB in 1.97s (369.1 Gbps)"
```

Or check main logs:
```bash
kubectl logs -n <namespace> <target-leader-pod> | grep "MX P2P weight transfer succeeded"
```

### Step 5: Cleanup

```bash
kubectl delete dgd -n <namespace> --all
# MX infra can stay running for future deployments
```

## Building the Image

Two Dockerfiles are provided for different base images:

| Dockerfile | Base Image | TRT-LLM Version |
|------------|-----------|-----------------|
| `Dockerfile.ph3-gcp-gb200` | `karenc:dynamo-trtllm-v1.0.0-a9b6f95` | 1.3.0rc5 |
| `Dockerfile.dynamo-runtime` | `tensorrtllm-runtime:1.1.0-dev.3` | 1.3.0rc11 |

### Build (recommended: dynamo-runtime base)

```bash
cd <modelexpress-repo-root>

docker buildx build --platform linux/arm64 --no-cache \
    -f examples/p2p_transfer_k8s/client/trtllm/Dockerfile.dynamo-runtime \
    --build-context trtllm=../TensorRT-LLM \
    -t nvcr.io/nvidian/dynamo-dev/<user>:dynamo-trtllm-mx-<tag> \
    --push .
```

Requires the TRT-LLM repo checked out alongside with the MX checkpoint loader:
```
~/work/github/
├── modelexpress/     (branch: kavink/trtllm_clean)
└── TensorRT-LLM/    (branch: kavink/mx-compat-fixes)
```

### What the Dockerfile does

1. Installs `modelexpress` Python client (gRPC + NIXL transfer)
2. Patches Dynamo with `--model-express-url` support (`patch_dynamo_mx.py`)
3. Copies `MXCheckpointLoader` from TRT-LLM fork
4. Patches base `model_loader.py` with P2P hooks (`patch_mx_loader.py`)
5. Symlinks `modelexpress` into the venv for import

### Pre-built images

```
nvcr.io/nvidian/dynamo-dev/kavink:dynamo-trtllm-mx-v2.9.0   # old base, E2E validated
nvcr.io/nvidian/dynamo-dev/kavink:dynamo-trtllm-mx-v3.2.0   # new base, E2E validated
```

## File Reference

| File | Purpose |
|------|---------|
| `Dockerfile.dynamo-runtime` | Docker build for `tensorrtllm-runtime` base images |
| `Dockerfile.ph3-gcp-gb200` | Docker build for older `karenc` base image |
| `mx-infra-decode.yaml` | ModelExpress server + Redis deployment |
| `kimi-source-decode-dgd.yaml` | Source DGD (TP=8, loads from disk, publishes) |
| `kimi-disagg-mx-tp8-dgd.yaml` | Target DGD (prefill + decode + frontend, loads via RDMA) |

Patch scripts (in `trtllm_patches/`):

| File | Purpose |
|------|---------|
| `dynamo/patch_dynamo_mx.py` | Adds `--model-express-url` to Dynamo engine/worker |
| `v1.3.0rc5/patch_mx_loader.py` | Adds P2P hooks to TRT-LLM model_loader.py |

## Companion PRs

| PR | Repo | Status |
|----|------|--------|
| [#13531](https://github.com/NVIDIA/TensorRT-LLM/pull/13531) | TRT-LLM | MX-only checkpoint loader (ready for review) |
| [#8037](https://github.com/ai-dynamo/dynamo/pull/8037) | Dynamo | `--model-express-url` engine integration |
| [#218](https://github.com/ai-dynamo/modelexpress/pull/218) | ModelExpress | This directory (Dockerfiles, yamls, patches) |
| [#202](https://github.com/ai-dynamo/modelexpress/pull/202) | ModelExpress | MX client (`MxLiveWeightLoader`, merged) |

## Customization

### Different model

Update in the DGD yamls:
- `--model-path` and `--served-model-name` args
- `MODEL_NAME` env var
- `model_kwargs.num_hidden_layers` in the ConfigMap
- Ensure the model is on the `shared-model-cache` PVC

### Different namespace

Update FQDN references for services:
- `modelexpress-server-decode.<namespace>.svc.cluster.local`
- `dynamo-platform-nats.<namespace>.svc.cluster.local`
- `dynamo-platform-etcd.<namespace>.svc.cluster.local`
- `resourceClaimTemplateName` for compute domain

### Different cluster / node pools

Update `nodeSelector` and `nodeAffinity` in the DGD yamls:
- `cloud.google.com/gke-nodepool` values

## GCP GB200 Required Config

All worker pods need these environment variables:

```yaml
env:
  HOME: /root
  UCX_TLS: "cuda_ipc,cuda_copy,rc"
  UCX_IB_GID_INDEX: "3"
  TRTLLM_UCX_INTERFACE: eth0
  OMPI_MCA_pml: ob1
  OMPI_MCA_btl: "tcp,self,vader"
  NCCL_CUMEM_ENABLE: "1"
  NCCL_NVLS_ENABLE: "1"
```
