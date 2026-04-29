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

### Step 1 — Check out the companion repos

The Dockerfile copies the `MXCheckpointLoader` source from the TRT-LLM
fork referenced in [PR #13531](https://github.com/NVIDIA/TensorRT-LLM/pull/13531).
Until that PR merges into `NVIDIA/TensorRT-LLM:main`, check out the
PR branch alongside the modelexpress repo:

```bash
mkdir -p ~/work/github && cd ~/work/github

# This repo (the modelexpress branch with the MX P2P client + this examples/ directory)
git clone https://github.com/ai-dynamo/modelexpress.git
cd modelexpress
git checkout <pr-branch-or-main-after-merge>      # e.g. kavink/trtllm_clean (PR #218)
cd ..

# TRT-LLM with MXCheckpointLoader
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout <pr-branch-or-main-after-merge>      # e.g. PR #13531 head, or main once merged
cd ..
```

After both PRs merge upstream you only need `main` of each repo.

The directory layout the build context expects:
```
~/work/github/
├── modelexpress/    <-- you build from here
└── TensorRT-LLM/    <-- referenced via --build-context trtllm=../TensorRT-LLM
```

### Step 2 — Build and push

```bash
cd ~/work/github/modelexpress

docker buildx build --platform linux/arm64 --no-cache \
    -f examples/p2p_transfer_k8s/client/trtllm/Dockerfile.dynamo-runtime \
    --build-context trtllm=../TensorRT-LLM \
    -t <YOUR_REGISTRY>/<YOUR_NAME>:<YOUR_TAG> \
    --push .
```

Substitute:
- `<YOUR_REGISTRY>` — your container registry host/org (e.g. `nvcr.io/<org>/<repo>`)
- `<YOUR_NAME>` — image name
- `<YOUR_TAG>` — version tag

For x86 builds, use `--platform linux/amd64` and a base image variant
that supports it. The `tensorrtllm-runtime` image is published for both
arm64 and amd64.

### What the Dockerfile does

1. Installs the `modelexpress` Python client (gRPC + NIXL transfer)
   from the local `modelexpress_client/python` source tree
2. Compiles the protobuf and symlinks `modelexpress` into the venv
3. Patches Dynamo with `--model-express-url` CLI support and engine
   integration (`trtllm_patches/dynamo/patch_dynamo_mx.py`)
4. Copies `MXCheckpointLoader` from the `TensorRT-LLM` build context
5. Patches the base `model_loader.py` with P2P hooks
   (`trtllm_patches/v1.3.0rc5/patch_mx_loader.py`)
6. Verifies every patch applied (build fails fast if any pattern
   doesn't match — protects against upstream API drift)

## File Reference

| File | Purpose |
|------|---------|
| `Dockerfile.dynamo-runtime` | Docker build for `tensorrtllm-runtime` base images |
| `Dockerfile.ph3-gcp-gb200` | Docker build for older `karenc` base image |
| `mx-infra-decode.yaml` | ModelExpress server + Redis deployment |
| `kimi-source-decode-dgd.yaml` | Source DGD (TP=8, loads from disk, publishes) |
| `kimi-disagg-mx-tp8-dgd.yaml` | Target DGD (prefill + decode + frontend, loads via RDMA) |
| [`hpa/`](hpa/README.md) | HPA-driven autoscale demo (single-replica → multi-replica with RDMA) |

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

The yamls in this directory use placeholders for cluster-specific
values. Before applying, replace the following:

| Placeholder | Replace with | Where |
|-------------|--------------|-------|
| `<REGISTRY>/<NAME>:<TAG>` | Your container registry path + tag | All DGD yamls (`image:` field) |
| `<NAMESPACE>` | The Kubernetes namespace where Dynamo platform runs | `NATS_SERVER`, `ETCD_ENDPOINTS`, `MODEL_EXPRESS_URL`, `resourceClaimTemplateName` |
| `<GPU_NODE_POOL>` | Your GPU node pool name | `cloud.google.com/gke-nodepool` in worker yamls |
| `<CPU_NODE_POOL>` | Your CPU node pool name | `cloud.google.com/gke-nodepool` in `mx-infra-decode.yaml` |

If you don't want to build, you can validate the deployment path with
any image that has the same layered components. Build the image with
the instructions above and substitute its full URI into the DGD yamls.

### Different model

Update in the DGD yamls:
- `--model-path` and `--served-model-name` args
- `MODEL_NAME` env var
- `model_kwargs.num_hidden_layers` in the ConfigMap
- Ensure the model is on the `shared-model-cache` PVC

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
