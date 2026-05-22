<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ModelExpress Deployment Guide

User-facing guide for configuring and deploying ModelExpress. For architecture details, see [`ARCHITECTURE.md`](ARCHITECTURE.md). For development setup, see [`../CONTRIBUTING.md`](../CONTRIBUTING.md).

## Server Configuration

ModelExpress uses a layered configuration system. Sources are applied in order of precedence:

1. **Command line arguments** (highest priority)
2. **Environment variables** (`MODEL_EXPRESS_*` prefix)
3. **Configuration file** (YAML)
4. **Default values** (lowest priority)

### Generating a Configuration File

```bash
cargo run --bin config_gen -- --output model-express.yaml
```

The generated file contains all options with their defaults:

```yaml
server:
  host: "0.0.0.0"
  port: 8001

cache:
  directory: "./cache"
  max_size_bytes: null
  eviction:
    enabled: true
    policy:
      type: lru
      unused_threshold: "7d"
      max_models: null
      min_free_space_bytes: null
    check_interval: "1h"

logging:
  level: info
  format: pretty
  file: null
  structured: false
```

### Starting the Server

The server requires `MX_METADATA_BACKEND` (`redis` or `kubernetes`) plus the connection
env vars for the chosen backend — the server refuses to start without them. See
[Distributed backend selection](#distributed-backend-selection) below for the full env
contract.

```bash
# Redis backend
export MX_METADATA_BACKEND=redis
export REDIS_URL=redis://localhost:6379
cargo run --bin modelexpress-server

# Kubernetes backend (typically only useful in-cluster)
export MX_METADATA_BACKEND=kubernetes
export POD_NAMESPACE=default   # or MX_METADATA_NAMESPACE
cargo run --bin modelexpress-server

# With a configuration file (backend env vars still required)
MX_METADATA_BACKEND=redis REDIS_URL=redis://localhost:6379 \
  cargo run --bin modelexpress-server -- --config model-express.yaml

# With CLI overrides
MX_METADATA_BACKEND=redis REDIS_URL=redis://localhost:6379 \
  cargo run --bin modelexpress-server -- --port 8080 --log-level debug

# Validate config without starting (backend env vars still required — the validator
# parses the full startup path including MX_METADATA_BACKEND)
MX_METADATA_BACKEND=redis REDIS_URL=redis://localhost:6379 \
  cargo run --bin modelexpress-server -- --config model-express.yaml --validate-config
```

### Configuration Options

#### Server Settings

| Option | CLI Flag | Env Var | Default | Description |
|--------|----------|---------|---------|-------------|
| host | `--host` | `MODEL_EXPRESS_SERVER_HOST` | `0.0.0.0` | Bind address |
| port | `--port`, `-p` | `MODEL_EXPRESS_SERVER_PORT` | `8001` | gRPC port |

#### Distributed backend selection

Model lifecycle state (download status, LRU timestamps) and P2P worker metadata are both
persisted to a distributed backend. The server fails fast at startup if no backend is
reachable.

| Env var | Values | Required | Notes |
|---------|--------|----------|-------|
| `MX_METADATA_BACKEND` | `redis` \| `kubernetes` | yes | Selects the backend for both the P2P metadata store and the model registry |
| `REDIS_URL` | e.g. `redis://host:6379` | when Redis | Redis connection (or set `MX_REDIS_HOST` / `MX_REDIS_PORT`) |
| `POD_NAMESPACE` / `MX_METADATA_NAMESPACE` | e.g. `default` | when Kubernetes | Namespace for the `ModelMetadata` and `ModelCacheEntry` CRDs |

To use the Kubernetes backend, apply `examples/crds.yaml` at cluster install time (installs both the `ModelMetadata` P2P CRD and the `ModelCacheEntry` registry CRD), then either enable `serviceAccount.rbac.enabled=true` on the Helm chart or apply `examples/p2p_transfer_k8s/server/kubernetes_backend/rbac-modelmetadata.yaml`.

#### Storage access modes

MX has one configurable filesystem path, the model weights cache (`MODEL_EXPRESS_CACHE_DIRECTORY`, default `./cache`). Its access-mode requirement depends on deployment topology, not on MX itself:

| Mode | Cache volume | Notes |
|------|-------------|-------|
| Single-replica MX, all pods on one node, RWO cache | RWO | Simplest option |
| Multi-container sharing the cache (e.g. vLLM worker on a different node) | RWX | Operator choice; MX doesn't force it |
| Multi-replica MX with `MODEL_EXPRESS_NO_SHARED_STORAGE=true` on clients (gRPC streaming) | RWO per replica OR ephemeral | Needs an MX-aware init container in the client pod; no ready-made vLLM recipe today (tracked MX-290) |
| ModelStreamer object storage on clients | none | Clients stream from object storage directly |
| P2P RDMA receivers | none on receiver (sender still needs disk) | Weights land in GPU HBM |

For new multi-replica deployments, prefer the no-shared-storage row: each MX replica can use its own RWO or ephemeral cache while Redis or Kubernetes coordinates lifecycle state. The RWX row is mainly for existing shared-cache topologies, and the single-replica row is a local/dev simplification.

#### Cache Settings

| Option | CLI Flag | Env Var | Default | Description |
|--------|----------|---------|---------|-------------|
| directory | `--cache-directory` | `MODEL_EXPRESS_CACHE_DIRECTORY` | `./cache` | Model cache directory |
| max_size_bytes | - | - | null (unlimited) | Max cache size in bytes |
| eviction.enabled | `--cache-eviction-enabled` | `MODEL_EXPRESS_CACHE_EVICTION_ENABLED` | `true` | Enable LRU eviction |

Eviction policy settings (in config file only):
- `eviction.policy.unused_threshold` - Evict models unused for this duration (default: 7 days)
- `eviction.policy.max_models` - Max models to keep (default: unlimited)
- `eviction.check_interval` - How often to check for eviction (default: 1 hour)

#### Logging Settings

| Option | CLI Flag | Env Var | Default | Description |
|--------|----------|---------|---------|-------------|
| level | `--log-level`, `-l` | `MODEL_EXPRESS_LOG_LEVEL` | `info` | trace, debug, info, warn, error |
| format | `--log-format` | `MODEL_EXPRESS_LOG_FORMAT` | `pretty` | json, pretty, compact |
| file | - | - | null (stdout) | Log file path |
| structured | - | - | `false` | Structured logging |

### Environment Variable Examples

```bash
export MODEL_EXPRESS_SERVER_HOST="127.0.0.1"
export MODEL_EXPRESS_SERVER_PORT=8080
export MODEL_EXPRESS_CACHE_DIRECTORY="/data/cache"
export MODEL_EXPRESS_CACHE_EVICTION_ENABLED=true
export MODEL_EXPRESS_LOG_LEVEL=debug
export MODEL_EXPRESS_LOG_FORMAT=json
export MX_METADATA_BACKEND=redis
export REDIS_URL=redis://redis:6379
```

## Client Configuration

The CLI client also uses layered configuration: CLI args > env vars > config file > defaults.

| Env Var | Default | Description |
|---------|---------|-------------|
| `MODEL_EXPRESS_ENDPOINT` | `http://localhost:8001` | Server endpoint |
| `MODEL_EXPRESS_TIMEOUT` | `30` | Request timeout (seconds) |
| `MODEL_EXPRESS_CACHE_DIRECTORY` | (auto) | Cache path override |
| `MODEL_EXPRESS_MAX_RETRIES` | (none) | Max retry attempts |
| `MODEL_EXPRESS_NO_SHARED_STORAGE` | `false` | Use gRPC streaming instead of shared storage |
| `MODEL_EXPRESS_TRANSFER_CHUNK_SIZE` | `32768` | Transfer chunk size (bytes) |

Cache directory resolution for HuggingFace: `MODEL_EXPRESS_CACHE_DIRECTORY` -> `HF_HUB_CACHE` -> `~/.cache/huggingface/hub`.

Cache directory resolution for NGC: `MODEL_EXPRESS_CACHE_DIRECTORY` -> `~/.cache/ngc`.

GCS uses the configured/default ModelExpress cache root; `MODEL_EXPRESS_CACHE_DIRECTORY` overrides it. Cached GCS models are stored under `<cache>/gcs/<bucket>/<object-prefix>`. See [`GCS_PROVIDER.md`](GCS_PROVIDER.md) for provider internals.

See [`CLI.md`](CLI.md) for full CLI usage documentation.

## Docker

### Production Image

The multi-stage Dockerfile builds all binaries (server, CLI, test tools):

```bash
docker build -t model-express .
docker run -p 8001:8001 model-express
```

### Docker Compose

Single-service setup for local development:

```bash
docker-compose up --build
```

### Custom Client Image (P2P Transfers)

For GPU-to-GPU weight transfers with vLLM:

```bash
docker build -f examples/p2p_transfer_k8s/client/vllm/Dockerfile \
  -t your-registry/mx-client:TAG .
docker push your-registry/mx-client:TAG
```

For SGLang:

```bash
docker build -f examples/p2p_transfer_k8s/client/sglang/Dockerfile \
  -t your-registry/sglang-modelexpress:TAG .
docker push your-registry/sglang-modelexpress:TAG
```

The Dynamo examples use their own runtime image Dockerfile:
`examples/dynamo_p2p_transfer_k8s/Dockerfile`.

## Kubernetes

### Standalone Deployment

Deploy the server using one of the example manifests under `examples/`:

- **With Redis backend**: `examples/p2p_transfer_k8s/server/redis_backend/modelexpress-server-redis.yaml`
- **With Kubernetes CRD backend**: `examples/p2p_transfer_k8s/server/kubernetes_backend/modelexpress-server-kubernetes.yaml`
- **Dynamo model cache**: `examples/dynamo_model_cache_k8s/agg.yaml`

### HuggingFace Token

Most deployments need a HuggingFace token for model downloads:

```bash
export HF_TOKEN=your_hf_token
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

### NGC API Key

To download models from NVIDIA NGC, set an NGC API key. The server resolves it in this order:

1. `NGC_API_KEY` environment variable
2. `NGC_CLI_API_KEY` environment variable
3. `~/.ngc/config` (written by `ngc config set`)

```bash
export NGC_API_KEY=your_ngc_api_key
kubectl create secret generic ngc-api-key-secret \
  --from-literal=NGC_API_KEY=${NGC_API_KEY} \
  -n ${NAMESPACE}
```

Pass it to the server pod via `envFrom` or individual `env` entries in your deployment manifest.

### Google Cloud Storage Credentials

To download models from Google Cloud Storage with the direct `gcs` provider, use a full `gs://<bucket>/<object-prefix>` model name and configure Google Application Default Credentials for the process doing the download. The identity needs permission to list and read objects under the model prefix, for example `storage.objects.list` and `storage.objects.get`.

Common credential options:

1. Set `GOOGLE_APPLICATION_CREDENTIALS` to a mounted service account JSON key.
2. Use `gcloud auth application-default login` for local development.
3. Use GKE Workload Identity or another platform-provided ADC source in Kubernetes.

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/var/secrets/google/service-account.json
kubectl create secret generic gcs-service-account-key \
  --from-file=service-account.json=/path/to/service-account.json \
  -n ${NAMESPACE}
```

Mount the secret into the server or client pod and set `GOOGLE_APPLICATION_CREDENTIALS` to the mounted file path. When using Workload Identity, no key secret is needed. For cache layout, manifest behavior, and failure modes, see [`GCS_PROVIDER.md`](GCS_PROVIDER.md).

### Helm Chart

The `helm/` directory provides a full Helm chart with configurable replicas, PVC, ingress, and resource limits.

```bash
# Deploy with defaults (1 replica, 10Gi PVC)
helm/deploy.sh --namespace my-ns

# Development (debug logging, 512Mi memory)
helm/deploy.sh --namespace my-ns --values helm/values-development.yaml

# Production (3 replicas, 2Gi memory, ingress, pod anti-affinity)
helm/deploy.sh --namespace my-ns --values helm/values-production.yaml

# Local testing (no PVC, emptyDir)
helm/deploy.sh --namespace my-ns --values helm/values-local-storage.yaml
```

See [`../helm/README.md`](../helm/README.md) for the full parameter reference and installation guide.

### Dynamo Model Cache Deployment

For deploying ModelExpress alongside Dynamo with a vLLM worker:

```bash
kubectl apply -f examples/dynamo_model_cache_k8s/agg.yaml
```

See [`../examples/dynamo_model_cache_k8s/README.md`](../examples/dynamo_model_cache_k8s/README.md) for the full guide.

## P2P GPU Weight Transfers

ModelExpress supports GPU-to-GPU model weight transfers between supported inference instances using NVIDIA NIXL over RDMA. vLLM uses `--load-format mx`, which auto-detects whether to load from disk or receive via RDMA. SGLang uses `remote_instance` with the `modelexpress` backend; see [SGLang Clients](#sglang-clients).

### Choosing a Metadata Backend

Pick based on workload, not operational preference. The choice has structural consequences for what the system can do.

| Workload shape                                                         | Backend          | Why                                                                                                                                            |
|------------------------------------------------------------------------|------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| Stable-weight inference. Weights fixed at pod startup, no mid-life refit. Simple K8s deployment. | `k8s-service`    | Lowest deployment footprint. No server, no Redis, no CRDs. Matches the homogeneous pool assumption that Service-routing requires.             |
| RL rollouts. Training loop updates weights every step, all inference pods refit in-place, repeat. | `redis` or `kubernetes` | Central store tracks each worker's state individually by `worker_id`. Targets can fetch "worker W as it exists right now" instead of random-sampling a pool. Live refits stay consistent at the per-worker level. |
| Live fine-tune broadcasts. New checkpoint produced outside training, pushed to all replicas, hot-swapped in place. | `redis` or `kubernetes` | Same reason as RL. The k8s-service backend can't swap a live pod's source_id without restarting the pod.                                      |
| Mixed-version fleet. Multiple revisions serving concurrently, callers dispatch by revision. | `redis` or `kubernetes` | Central store indexes by `mx_source_id`, so multiple identities coexist cleanly. k8s-service requires one Service pool per identity.          |
| Heterogeneous hardware. Some sources on H100, some on B200, callers match on topology. | `redis` or `kubernetes` | Central store carries per-worker metadata including identity fields; k8s-service's pool assumption requires all pods to be interchangeable.   |
| Multiple checkpoints in parallel (base + LoRA, fp16 + nvfp4, etc.).   | Either           | Different `SourceIdentity` produces different `mx_source_id`. Each identity gets its own Service (k8s-service) or its own source records (central). Both work. |

The central-coordinator backends (`redis`, `kubernetes`) are the default. Reach for `k8s-service` specifically when the deployment meets three criteria: (1) weights stay fixed for each pod's lifetime, (2) every pod behind a given Service serves the exact same checkpoint, and (3) dropping the `modelexpress-server` / Redis / CRD components is a material simplification.

See [`K8S_SERVICE_BACKEND.md`](K8S_SERVICE_BACKEND.md) for the design rationale, limitations, and the structural reasons these backend families differ.

### P2P Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MX_METADATA_BACKEND` | (required on server; `""` on client) | Server: `redis` or `kubernetes`. Client: `""`/`server`/`redis`/`kubernetes` (central server) or `k8s-service` (decentralized via K8s Service routing). |
| `MODEL_EXPRESS_URL` | `localhost:8001` | gRPC server address (ignored when client uses `k8s-service` backend) |
| `MX_SERVER_ADDRESS` | `localhost:8001` | Backward-compat alias for `MODEL_EXPRESS_URL` |
| `MX_REGISTER_LOADERS` | `1` | Auto-register the mx loader with vLLM |
| `MX_POOL_REG` | `0` | Allocation-level NIXL registration via `cuMemGetAddressRange`. Registers each unique cudaMalloc block instead of each tensor, typically 80-99% fewer registrations, without changing transfer semantics. `MX_VMM_ARENA=1` uses direct arena registration and does not require pool-reg. |
| `MX_VMM_ARENA` | `0` | Route weight allocations into a CUDA VMM arena via PyTorch's `CUDAPluggableAllocator`, then register the used arena range as one NIXL MR with dmabuf at end-of-load. Reserves 16.0 TiB of VA by default, with no physical commit until allocations are mapped. Requires the `modelexpress.vmm._alloc_ext` C extension to have built at install time; if it did not, this flag is a no-op with a warning and the loader falls back to the pool-reg path. See [VMM Arena](#vmm-arena-single-mr-registration). |
| `UCX_CUDA_COPY_REG_WHOLE_ALLOC` | (UCX default) | Set to `off` with `MX_VMM_ARENA=1` until the upstream UCX `cuda_copy_md` length-truncation fix ships. |
| `MX_NIXL_BACKEND` | `UCX` | NIXL backend for GPU-to-GPU RDMA. `UCX` (default) for InfiniBand / RoCE. `LIBFABRIC` for AWS EFA — see [NIXL Backend Selection](#nixl-backend-selection). |
| `MX_RDMA_NIC_PIN` | (unset) | Per-rank IB NIC pinning. `auto` runs a topology probe; comma-separated NIC list is an explicit override. Workaround for openucx/ucx#11259. |
| `MX_RDMA_NIC_PIN_MIN_RATE_GBPS` | (auto, max-rate filter) | Override the auto-detect rate filter with an explicit lower bound (Gb/s). |
| `MODEL_EXPRESS_LOG_LEVEL` | (inherits vLLM) | Override log level for `modelexpress.*` loggers. `DEBUG` enables per-tensor checksums and adopted tensor details |
| `MX_P2P_METADATA` | `0` | Enable P2P metadata exchange (source workers only). Opt-in on central-coordinator backends. Auto-enabled (and this env var ignored) on backends that declare themselves decentralized, currently `k8s-service`. |
| `MX_METADATA_PORT` | `5555` | Base NIXL listen port; effective port is `MX_METADATA_PORT + device_id` |
| `MX_WORKER_GRPC_PORT` | `0` | Worker gRPC port for P2P tensor manifest serving |
| `MX_WORKER_HOST` | (auto-detect) | Override worker IP/hostname for P2P endpoints |
| `MX_MODEL_REVISION` | (from vLLM config) | Override for `SourceIdentity.revision`. Pin to the exact HF commit SHA / checkpoint version so `mx_source_id` is content-addressed. Required for decentralized backends where no central coordinator tracks versions. |
| `MX_K8S_SERVICE_PATTERN` | `mx-sources` | DNS template for the `k8s-service` backend. `{rank}` is substituted with the worker's own rank. If the resolved pattern has no `:port`, the client auto-appends `:{MX_WORKER_GRPC_PORT + rank}` (multi-GPU-per-pod shape); if it has an explicit port, that port is used verbatim (1-GPU-per-pod shape). |
| `MX_K8S_SOURCE_RETRIES` | `5` | `k8s-service` backend: max retries on `FAILED_PRECONDITION` (revision mismatch during rolling updates). Each retry opens a fresh gRPC channel so kube-proxy re-picks a backend. |
| `MX_K8S_SOURCE_BACKOFF_SECONDS` | `0.5` | `k8s-service` backend: sleep between retry attempts. |
| `MX_STATUS_TTL_SECS` | `3600` | TTL for Redis metadata keys (seconds) |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL (Redis backend only) |
| `MX_METADATA_NAMESPACE` | `default` | K8s namespace for CRD backend |
| `VLLM_RPC_TIMEOUT` | `7200000` | vLLM RPC timeout in ms (2 hours for large models) |
| `VLLM_PLUGINS` | - | Set to `modelexpress` to register the mx loader |

Each GPU worker publishes independently using its global rank (`torch.distributed.get_rank()`). No inter-worker coordination or barriers required.

### NIXL Backend Selection

`MX_NIXL_BACKEND` selects the NIXL plugin used for GPU-to-GPU RDMA.
The default `UCX` covers InfiniBand and RoCE clusters. Set
`MX_NIXL_BACKEND=LIBFABRIC` on AWS EFA, where the UCX backend can
silently fall back to TCP depending on the libibverbs / EFA installer
combination on the host.

Both source and target workers must use the same backend — backends
do not interoperate. Confirm via worker logs:

```
NIXL agent 'mx-auto-worker0-...' created on device 0 (backend=LIBFABRIC)
```

### NIC Pinning (UCX Workaround)

`MX_RDMA_NIC_PIN=auto` works around
[openucx/ucx#11259](https://github.com/openucx/ucx/issues/11259), where
UCX may pick a NIC on a different NUMA node from a worker's GPU when
the IB device pool spans multiple NUMA domains; the resulting CUDA
RDMA traffic crosses the CPU interconnect and loses bandwidth.

The probe runs at worker startup, walks PCIe sysfs, and sets
`UCX_NET_DEVICES` to a single NUMA-local NIC per worker before the
NIXL agent is constructed. Same affinity metric as
`nvidia-smi topo -m` (PIX > PXB > NODE > SYS).

Recommended on multi-GPU hosts where the IB pool spans NUMA. Leave
unset on single-NUMA hosts or when you manage `UCX_NET_DEVICES` per
rank externally. Once the upstream UCX fix lands and a patched UCX
is deployed, drop this env var.

`MX_RDMA_NIC_PIN_MIN_RATE_GBPS` overrides the default max-rate filter
for clusters with multiple rate tiers in the compute fabric.
`MX_RDMA_NIC_PIN` also accepts a comma-separated NIC list indexed by
`device_id` for unusual topologies where the auto-probe can't infer
the mapping.

### VMM Arena (Single-MR Registration)

`MX_VMM_ARENA=1` installs a `CUDAPluggableAllocator` that routes weight
allocations issued during `initialize_model`, `load_weights`, and
`process_weights_after_loading` into a CUDA VMM arena. The arena reserves
16.0 TiB of virtual address space at startup with `cuMemAddressReserve`.
That reservation only consumes VA. It does not commit VRAM until an
allocation is mapped with `cuMemMap`, so the large default is safe on CUDA
systems with a 49-bit device VA space.

Each allocation from PyTorch maps its own physical VMM handle at the next
arena address. Frees unmap and release that handle, so replacement tensors
created during post-processing can return physical memory before the final
registration step. At end-of-load, ModelExpress registers the used arena
range once through dmabuf and publishes all tensor descriptors against
that single MR.

Recommended source-worker setting:

```bash
MX_VMM_ARENA=1
UCX_CUDA_COPY_REG_WHOLE_ALLOC=off
```

`MX_POOL_REG=1` is not required for the arena path. Pool-reg still helps
non-arena deployments by deduplicating normal cudaMalloc allocations, but
arena registration bypasses the pool-reg path and calls `register_arena`
directly. The arena produces one MR for the used range regardless of the
pool-reg setting.

Set `UCX_CUDA_COPY_REG_WHOLE_ALLOC=off` until the upstream UCX
`cuda_copy_md` length-truncation fix ships. Without it, UCX can truncate
a multi-handle VMM registration to the first physical handle, and RDMA
operations that cross into later handles fail. See the reproducer and
fix notes in this gist:
<https://gist.github.com/nicolasnoble/e0e57eb5a1b902057ae3d1df59c039cf>.

### P2P Metadata Exchange (Opt-In)

`MX_P2P_METADATA=1` makes source workers expose their own per-worker gRPC `WorkerService` (the `WorkerGrpcServer` on `MX_WORKER_GRPC_PORT`) and their NIXL agent metadata directly on the worker's NIXL listen thread (`MX_METADATA_PORT`). Targets fetch both directly from the source worker rather than pulling them through the central store. The division of responsibility depends on which metadata backend is in use:

- **Central-coordinator backends (`redis`, `kubernetes`):** opt-in via `MX_P2P_METADATA=1`. By default the source publishes full tensor metadata (NIXL blobs + tensor descriptors) to the central server, and targets fetch the full blob from the server. With the env var set, the source publishes only a lightweight pointer (its `worker_grpc_endpoint` and NIXL listen address) to the central server, and targets use that pointer to connect directly to the source for the MB-scale data. Targets auto-detect which mode a source is using based on whether `worker_grpc_endpoint` is populated in the server's metadata; no configuration needed on the target side.
- **`k8s-service` backend:** auto-enabled. The backend declares itself decentralized (via a class attribute `REQUIRES_P2P_METADATA = True`), so the client forces the P2P path regardless of the env var. Deployers don't need to set `MX_P2P_METADATA` themselves. If the env var is explicitly set to `0` alongside this backend, the client logs a warning that the setting is ignored but otherwise proceeds correctly.

Set `MX_METADATA_PORT` and `MX_WORKER_GRPC_PORT` to fixed ports when running in K8s (port 0 picks an ephemeral port). Set `MX_WORKER_HOST` if the pod IP auto-detection doesn't produce a routable address.

### ModelStreamer (Object Storage & Local Disk)

ModelStreamer streams safetensors directly to GPU memory via `runai-model-streamer`. Supports S3, GCS, Azure Blob Storage, and local filesystem (PVC) paths. This is a storage-loading path and does not require P2P by itself. If the same deployment also enables ModelExpress P2P metadata and RDMA resources, later replicas can receive weights from an already-loaded source instead of streaming from storage again.

All storage backends (S3, GCS, Azure) are included as core dependencies — no extra install step needed. The strategy activates when `MX_MODEL_URI` is set. See [`../examples/model_streamer_k8s/`](../examples/model_streamer_k8s/) for Kubernetes examples, including the Azure Blob recipe.

**General configuration:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MX_MODEL_URI` | (none) | Model location. Must be set to enable ModelStreamer. Accepts: remote URI (`s3://bucket/model`, `gs://...`, `az://...`), absolute local path (`/models/deepseek-ai/DeepSeek-V3`), or HuggingFace model ID (`deepseek-ai/DeepSeek-V3` — resolved via `HF_HUB_CACHE`). |
| `MX_MS_DISTRIBUTED` | `0` | Enable distributed streaming (streams directly to each worker's GPU). Requires tensor parallelism > 1 and a CUDA-capable platform. Set to `1` to activate. |
| `RUNAI_STREAMER_CONCURRENCY` | `8` | Number of concurrent read threads |
| `RUNAI_STREAMER_MEMORY_LIMIT` | (none) | CPU staging buffer size in bytes. `0` reuses a single-tensor buffer (most memory efficient). See [runai-model-streamer docs](https://github.com/run-ai/model-streamer). |

**S3 / S3-compatible:**

| Variable | Description |
|----------|-------------|
| `AWS_ACCESS_KEY_ID` | S3 credentials (auto-detected by boto3) |
| `AWS_SECRET_ACCESS_KEY` | S3 credentials |
| `AWS_SESSION_TOKEN` | Required for temporary credentials (SSO/IRSA) |
| `AWS_DEFAULT_REGION` | AWS region |
| `AWS_ENDPOINT_URL` | Custom endpoint for S3-compatible storage (MinIO, Ceph) |

**Google Cloud Storage:**

| Variable | Description |
|----------|-------------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON key file |

Also supports GKE Workload Identity and Application Default Credentials (ADC) — no env vars needed when running on GKE with a properly configured service account.

**Azure Blob Storage:**

| Variable | Description |
|----------|-------------|
| `AZURE_STORAGE_ACCOUNT_NAME` | Storage account name |
| `AZURE_CLIENT_ID` | Service principal or workload identity client ID |
| `AZURE_CLIENT_SECRET` | Service principal client secret |
| `AZURE_TENANT_ID` | Azure tenant ID |

Use service principal auth, Azure Managed Identity, or AKS workload identity through `DefaultAzureCredential`. The identity needs `Storage Blob Data Reader` on the storage account or container.

Credentials are auto-detected by the underlying cloud SDKs. No credentials flow through the MX server or gRPC.

### UCX/NIXL Tuning

| Variable | Recommended | Description |
|----------|-------------|-------------|
| `UCX_TLS` | `rc_x,rc,dc_x,dc,cuda_copy` | Transport layers for InfiniBand |
| `UCX_RNDV_SCHEME` | `get_zcopy` | Zero-copy RDMA reads |
| `UCX_RNDV_THRESH` | `0` | Force rendezvous for all transfers |
| `NIXL_LOG_LEVEL` | `INFO` | NIXL logging (DEBUG for troubleshooting) |
| `UCX_LOG_LEVEL` | `WARN` | UCX logging (DEBUG for troubleshooting) |

### P2P Kubernetes Deployment

Deploy multiple identical instances - the first one loads from disk and subsequent ones receive via RDMA.

#### Redis Backend

```bash
NAMESPACE=my-namespace

# Deploy server with Redis sidecar
kubectl -n $NAMESPACE apply -f examples/p2p_transfer_k8s/server/redis_backend/modelexpress-server-redis.yaml

# Deploy single-node vLLM (TP=8, 1 node)
kubectl -n $NAMESPACE apply -f examples/p2p_transfer_k8s/client/vllm/vllm-single-node.yaml
```

#### Kubernetes CRD Backend

```bash
# Install CRDs and RBAC
kubectl apply -f examples/crds.yaml
kubectl -n $NAMESPACE apply -f examples/p2p_transfer_k8s/server/kubernetes_backend/rbac-modelmetadata.yaml

# Deploy server with CRD backend
kubectl -n $NAMESPACE apply -f examples/p2p_transfer_k8s/server/kubernetes_backend/modelexpress-server-kubernetes.yaml

# Deploy multi-node vLLM (TP=8, PP=2, 2 nodes)
kubectl -n $NAMESPACE apply -f examples/p2p_transfer_k8s/client/vllm/vllm-multi-node.yaml
```

See [`../examples/p2p_transfer_k8s/README.md`](../examples/p2p_transfer_k8s/README.md) for the full P2P transfer guide including architecture, prerequisites, and performance expectations.

#### K8s-Service-Routed Backend

No `modelexpress-server`, no Redis, no CRDs. Source pods sit behind a Kubernetes Service; clients hit the Service DNS and kube-proxy load-balances. See [`K8S_SERVICE_BACKEND.md`](K8S_SERVICE_BACKEND.md) for when to use this backend and when to prefer the central-coordinator alternatives.

Two deployment topologies are supported; pick based on your TP parallelism needs:

1. **Multi-GPU-per-pod** (TP ranks share NVLink inside one pod). One Service with N named ports. Default pattern: `MX_K8S_SERVICE_PATTERN=mx-sources`, client auto-computes `:{MX_WORKER_GRPC_PORT + rank}`.
2. **1-GPU-per-pod** (one rank per pod; rank partitioning via labels). N Services with rank selectors. Pattern: `MX_K8S_SERVICE_PATTERN=mx-sources-rank-{rank}:6555`.

**Deploy the multi-GPU-per-pod shape (the common case for TP inference):**

```bash
# 1. Create the HF token secret (once per namespace).
export HF_TOKEN=your_hf_token
kubectl -n $NAMESPACE create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN}

# 2. Apply the source pool: one Service with N named ports + one
#    Deployment with multi-GPU pods.
kubectl -n $NAMESPACE apply -f examples/k8s_service_sources/sources-tp2-single-pod.yaml

# 3. Wait for the first replica to finish loading (can take minutes for
#    large models). Readiness probe flips when the WorkerGrpcServer is
#    serving.
kubectl -n $NAMESPACE wait --for=condition=Ready pod -l app=mx-sources --timeout=15m

# 4. Verify the Service has live endpoints.
kubectl -n $NAMESPACE get svc mx-sources
kubectl -n $NAMESPACE get endpoints mx-sources

# 5. Scale up. New replicas will pull weights via P2P RDMA from the
#    existing ready pods rather than re-downloading from storage.
kubectl -n $NAMESPACE scale deployment mx-sources --replicas=4
```

**Deploy the 1-GPU-per-pod shape:**

```bash
kubectl -n $NAMESPACE apply -f examples/k8s_service_sources/sources-tp2.yaml
kubectl -n $NAMESPACE wait --for=condition=Ready pod -l app=mx-sources --timeout=15m
kubectl -n $NAMESPACE apply -f examples/k8s_service_sources/target.yaml
```

**Minimal inline YAML for the single-Service / multi-port shape:**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mx-sources
spec:
  selector: { app: mx-sources }
  ports:
    - { name: rank-0, port: 6555, targetPort: 6555 }
    - { name: rank-1, port: 6556, targetPort: 6556 }
    # ... one port per rank, port = 6555 + rank
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mx-sources
spec:
  replicas: 2
  selector: { matchLabels: { app: mx-sources } }
  template:
    metadata: { labels: { app: mx-sources } }
    spec:
      containers:
        - name: vllm
          image: your-registry/modelexpress-client:TAG
          env:
            - { name: MX_METADATA_BACKEND, value: "k8s-service" }
            - { name: MX_MODEL_REVISION,   value: "<pinned-commit-sha>" }
            - { name: MX_WORKER_GRPC_PORT, value: "6555" }
            # MX_K8S_SERVICE_PATTERN defaults to `mx-sources`; omit unless overriding.
          args: ["--model", "$(MODEL_NAME)", "--load-format", "mx", "--tensor-parallel-size", "2"]
          resources: { limits: { nvidia.com/gpu: 2 } }
```

**Common operations:**

```bash
# Check which rank a pod's workers are serving.
kubectl -n $NAMESPACE logs deploy/mx-sources -c vllm | grep -i "worker_rank"

# Inspect the Service's port -> backend mapping.
kubectl -n $NAMESPACE describe svc mx-sources

# Rolling update to a new model revision. Update MX_MODEL_REVISION env
# in the Deployment; K8s rolls pods one by one; during the transition
# kube-proxy may route to either version. The client handshake returns
# FAILED_PRECONDITION on mismatch, and targets retry on a fresh channel.
kubectl -n $NAMESPACE set env deployment/mx-sources MX_MODEL_REVISION=<new-sha>
kubectl -n $NAMESPACE rollout status deployment/mx-sources

# Tear down.
kubectl -n $NAMESPACE delete -f examples/k8s_service_sources/sources-tp2-single-pod.yaml
```

See [`../examples/k8s_service_sources/README.md`](../examples/k8s_service_sources/README.md) for the annotated manifests and [`K8S_SERVICE_BACKEND.md`](K8S_SERVICE_BACKEND.md) for the design rationale.

#### SGLang Clients

ModelExpress also works as the remote-instance weight loader for SGLang via
upstream [sgl-project/sglang#24723](https://github.com/sgl-project/sglang/pull/24723),
supporting both Mooncake TransferEngine and NIXL transports. See
[`SGLANG.md`](SGLANG.md) for the user-facing guide.

## Debugging

```bash
# Stream server logs
kubectl -n $NAMESPACE logs -f deploy/modelexpress-server

# Stream vLLM instance logs
kubectl -n $NAMESPACE logs -f deploy/mx-vllm

# Check Redis state (P2P metadata)
kubectl -n $NAMESPACE exec deploy/modelexpress-server -c redis -- redis-cli KEYS 'mx:source:*'

# Inspect a source index (identity + worker list)
kubectl -n $NAMESPACE exec deploy/modelexpress-server -c redis -- redis-cli HGETALL 'mx:source:<source_id>'

# Flush Redis (clear stale metadata - do this on redeploy)
kubectl -n $NAMESPACE exec deploy/modelexpress-server -c redis -- redis-cli FLUSHALL

# Check Kubernetes CRD state (P2P worker metadata + model registry)
kubectl -n $NAMESPACE get modelmetadatas
kubectl -n $NAMESPACE get modelcacheentries   # model registry (lifecycle state, LRU)

# Test inference
kubectl -n $NAMESPACE exec deploy/mx-vllm -- curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-ai/DeepSeek-V3", "prompt": "Hello", "max_tokens": 10}'
```

## Performance Reference

| Model | Total Data | Transfer Time | Per-Worker Speed |
|-------|-----------|---------------|------------------|
| DeepSeek-V3 (671B, FP8) | 681 GB (8 GPUs) | ~15 seconds | ~45 Gbps |
| Llama 3.3 70B | 140 GB (8 GPUs) | ~5 seconds | ~28 Gbps |
