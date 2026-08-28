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

> Most environment variables the code reads — including their defaults and fallback chains — are defined in one place per language: `modelexpress_common/src/envs.rs` (Rust server/client) and `modelexpress_client/python/modelexpress/envs.py` (Python client). Treat those modules as the canonical inventory; the tables below mirror them, and the documented exceptions are called out where needed.

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
| metrics_port | `--metrics-port` | `MODEL_EXPRESS_SERVER_METRICS_PORT` | `9401` | Prometheus `/metrics` port. `0` disables the listener. Deliberately not the gRPC port: tonic serves HTTP/2 only, so a scrape aimed at `--port` can never complete. See [METRICS.md](METRICS.md). |

> The env var above reaches the server **only** through the `--metrics-port` clap
> override. The layered config loader reads environment variables as
> `Environment::with_prefix("MODEL_EXPRESS").separator("_")`, so
> `MODEL_EXPRESS_SERVER_METRICS_PORT` resolves to the key path
> `server.metrics.port`, matches no field, and is dropped by serde without a
> warning. In a config file the field is `server.metrics_port`.

#### Distributed backend selection

Model lifecycle state (download status, LRU timestamps) and P2P worker metadata are both
persisted to a distributed backend. The server fails fast at startup if no backend is
reachable.

| Env var | Values | Required | Notes |
|---------|--------|----------|-------|
| `MX_METADATA_BACKEND` | `redis` \| `kubernetes` | yes | Selects the backend for both the P2P metadata store and the model registry |
| `REDIS_URL` | e.g. `redis://host:6379` | when Redis | Redis connection (or set `MX_REDIS_HOST` / `MX_REDIS_PORT`) |
| `POD_NAMESPACE` / `MX_METADATA_NAMESPACE` | e.g. `default` | when Kubernetes | Namespace for the `ModelMetadata` and `ModelCacheEntry` CRDs |

To use the Kubernetes backend, apply `examples/crds.yaml` at cluster install time
(installs both the `ModelMetadata` P2P CRD and the `ModelCacheEntry` registry CRD),
then either enable `serviceAccount.rbac.enabled=true` on the Helm chart or apply
`examples/p2p_transfer_k8s/server/kubernetes_backend/rbac-modelmetadata.yaml`.
The chart creates a `ClusterRole` and `ClusterRoleBinding`, allowing the server
to run in a dedicated namespace while accessing metadata resources in another
namespace.

For automatic cleanup of P2P metadata, expose the client Pod identity through
the Kubernetes Downward API. The checked-in vLLM, SGLang, and Dynamo manifests
already include these fields:

```yaml
env:
  - name: POD_NAME
    valueFrom:
      fieldRef:
        fieldPath: metadata.name
  - name: POD_UID
    valueFrom:
      fieldRef:
        fieldPath: metadata.uid
  - name: POD_NAMESPACE
    valueFrom:
      fieldRef:
        fieldPath: metadata.namespace
```

With the Kubernetes metadata backend, a complete same-namespace identity makes
the Pod the owner of every weight or artifact `ModelMetadata` CR it publishes.
Kubernetes then garbage-collects those records when the Pod is deleted. Missing,
partial, or cross-namespace identity does not block publication; ModelExpress
omits the owner reference and retains the previous reaper-based lifecycle. Redis
and non-Kubernetes deployments ignore these fields.

#### Storage access modes

MX has one configurable filesystem path, the model weights cache (`MODEL_EXPRESS_CACHE_DIRECTORY`, default `./cache`). Its access-mode requirement depends on deployment topology, not on MX itself:

| Mode | Cache volume | Notes |
|------|-------------|-------|
| Single-replica MX, all pods on one node, RWO cache | RWO | Simplest option |
| Multi-container sharing the cache (e.g. vLLM worker on a different node) | RWX | Operator choice; MX doesn't force it |
| Multi-replica MX with `MODEL_EXPRESS_NO_SHARED_STORAGE=true` on clients (gRPC streaming) | RWO per replica OR ephemeral | Needs an MX-aware init container in the client pod; no ready-made vLLM recipe today (tracked MX-290) |
| ModelStreamer from object storage on clients | none | Clients stream through a bounded CPU staging buffer without landing the checkpoint on local disk |
| ModelStreamer from a local path on clients | Existing local/PVC path | Reads the configured local checkpoint through the pipelined ModelStreamer path |
| P2P RDMA receivers, weights only | none on receiver | Weights land in GPU HBM; the source may have bootstrapped through InstantTensor, ModelStreamer, GDS, or the native loader |
| P2P RDMA receivers, weights and artifacts | Writable local staging and runtime cache paths | Weights land in GPU HBM. File-backed artifacts are staged locally, verified, and installed into the target engine's filesystem caches. |

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
| `MODEL_EXPRESS_NO_SHARED_STORAGE` | `false` | Use gRPC streaming instead of shared storage |
| `MODEL_EXPRESS_TRANSFER_CHUNK_SIZE` | `32768` | Transfer chunk size (bytes) |

Cache directory resolution for HuggingFace: `MODEL_EXPRESS_CACHE_DIRECTORY` -> `HF_HUB_CACHE` -> `~/.cache/huggingface/hub`.

Cache directory resolution for NGC: `MODEL_EXPRESS_CACHE_DIRECTORY` -> `~/.cache/ngc`.

GCS uses the configured/default ModelExpress cache root; `MODEL_EXPRESS_CACHE_DIRECTORY` overrides it. Cached GCS models are stored under `<cache>/gcs/<bucket>/<object-prefix>`. See [`GCS_PROVIDER.md`](GCS_PROVIDER.md) for provider internals.

See [`CLI.md`](CLI.md) for full CLI usage documentation.

## ServiceAccount Authentication

Optional, off by default. When enabled, the server authenticates every gRPC caller
(except health checks) against a Kubernetes ServiceAccount token and authorizes them
against an exact-match allowlist. No sidecar or service mesh is required: the server
calls the Kubernetes `TokenReview` API in-process.

- **AuthN**: the caller presents a projected ServiceAccount token; the server verifies it
  via `TokenReview` and extracts `system:serviceaccount:<namespace>:<serviceaccount>`.
- **AuthZ**: that `<namespace>:<serviceaccount>` must exactly match a configured allowlist
  entry.

> **Warning: only enable auth over an encrypted transport.** The server and clients
> speak plaintext gRPC; neither terminates TLS itself. Without encryption the bearer
> token crosses the wire in cleartext and anyone who can sniff the traffic can replay
> it until it expires. Run enforce mode only where the transport is encrypted, e.g. a
> service mesh providing mTLS (Istio/Linkerd sidecars) or a TLS-terminating proxy in
> front of the server.

Other properties to be aware of:

- Auth is enforced when each RPC starts. A long-lived streaming RPC that was accepted
  keeps flowing even if the token expires or the ServiceAccount is revoked mid-stream;
  revocation takes effect on the next RPC (bounded by the cache TTL below).
- Definitive rejections are cached per token; backend errors (e.g. an unreachable
  apiserver) return `UNAVAILABLE` and are not cached. A caller cycling unique invalid
  tokens sends one `TokenReview` to the Kubernetes API server per token. The gRPC port
  should not be reachable from untrusted networks.

### Modes

| Mode | Behavior |
|------|----------|
| `off` (default) | No auth. Tokens are ignored. |
| `enforce` | Verify every call and reject unauthenticated or non-allowlisted callers. |

### Server configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `MODEL_EXPRESS_SECURITY_MODE` | `off` | `off` \| `enforce` |
| `MODEL_EXPRESS_SECURITY_TOKEN_AUDIENCES` | (none) | Comma-separated audiences the token must carry. Required for `enforce`. |
| `MODEL_EXPRESS_SECURITY_ALLOWED_SERVICE_ACCOUNTS` | (none) | Comma-separated `<namespace>:<serviceaccount>` allowlist. Required for `enforce`. |
| `MODEL_EXPRESS_SECURITY_CACHE_TTL_SECS` | `60` | TTL for the verified-token and rejection caches. |

`enforce` fails config validation if either the audience list or the allowlist is empty,
so an omitted list can't silently deny-all or accept-any-audience. Configured values
still need to be correct.

The server's ServiceAccount needs permission to create `TokenReview`s (a cluster-scoped
subresource), via a `ClusterRoleBinding` to the built-in `system:auth-delegator` role.
The Helm chart creates this automatically when it also creates the ServiceAccount
(`serviceAccount.create=true`, the default) and `security.enabled=true`; with an
existing ServiceAccount, create the equivalent binding separately:

```yaml
security:
  enabled: true
  mode: enforce
  tokenAudiences: ["modelexpress"]
  allowedServiceAccounts:
    - "vllm:worker"
    - "vllm:router"
```

### Client configuration

Clients (Rust and Python) attach the token automatically when a projected token file is
present, and send nothing when it is absent (so the same client works against an `off`
server, including off-cluster). Mount a projected token into each worker pod with an
audience that matches the server's, then point the client at it:

```yaml
volumes:
  - name: mx-token
    projected:
      sources:
        - serviceAccountToken:
            path: modelexpress
            audience: modelexpress
            expirationSeconds: 3600
# mounted at /var/run/secrets/tokens/modelexpress
```

| Env Var | Default | Description |
|---------|---------|-------------|
| `MX_AUTH_TOKEN_PATH` | `/var/run/secrets/tokens/modelexpress` | Projected token file path |
| `MX_AUTH_TOKEN_TTL_SECONDS` | `60` | How often to re-read the token (rotation is also picked up on mtime change) |

## Docker

### Production Image

The multi-stage Dockerfile builds all binaries (server, CLI, test tools):

```bash
docker build -f docker/Dockerfile -t model-express .
docker run -p 8001:8001 model-express
```

### Docker Compose

Local development setup. Brings up the server plus a Redis metadata backend:

```bash
docker compose -f docker/docker-compose.yml up --build
```

### Python Client Distributions

`docker/Dockerfile.client-wheel` builds the publishable artifacts for the
Python client on top of `quay.io/pypa/manylinux_2_28_${arch}`. Per target
platform it produces:

- `manylinux_2_28_x86_64` or `manylinux_2_28_aarch64` wheels for cp310,
  cp311, cp312, cp313 (each compiles the `modelexpress.vmm._alloc_ext` shim
  against the matching CPython ABI and is hardened with `auditwheel repair`)
- `py3-none-any` pure-Python wheel built with `MX_SKIP_EXT=1` (no compiled
  extension; runtime falls back to the pool-reg path)
- sdist tarball

The `manylinux_2_28` base images and `auditwheel` are pinned (by digest and
version respectively) so published artifacts are reproducible. Refresh the
base-image digests deliberately with
`docker buildx imagetools inspect quay.io/pypa/manylinux_2_28_<arch>`.

The Dockerfile is multi-arch. Pick the target with buildx `--platform`:

```bash
# x86_64 only -> ./dist/*.whl, ./dist/*.tar.gz
docker buildx build --platform linux/amd64 \
  -f docker/Dockerfile.client-wheel \
  --target export --output type=local,dest=./dist .

# arm64 only -> ./dist/*.whl, ./dist/*.tar.gz
docker buildx build --platform linux/arm64 \
  -f docker/Dockerfile.client-wheel \
  --target export --output type=local,dest=./dist .

# Both at once -> ./dist/linux_amd64/* and ./dist/linux_arm64/*
docker buildx build --platform linux/amd64,linux/arm64 \
  -f docker/Dockerfile.client-wheel \
  --target export --output type=local,dest=./dist .
```

Cross-platform builds need QEMU emulation registered with buildx
(`docker run --privileged --rm tonistiigi/binfmt --install all` once per
host). Native builds on the matching arch run without emulation.

Without buildx (single arch, matches the host):

```bash
docker build -f docker/Dockerfile.client-wheel --target builder -t mx-wheel-builder .
docker run --rm -v "$PWD/dist:/out" mx-wheel-builder bash -lc 'cp -r /dist/. /out/'

#### CI uploads to Artifactory

`.github/workflows/build-wheels.yml` runs this Dockerfile on every PR
(via `copy-pr-bot` mirroring into `pull-request/<pr_id>` branches) and
every push to `main` / `release/**`, building both archs in parallel on
velonix self-hosted runners and uploading the artifacts to NV Artifactory.

Destination layout under `${ARTIFACTORY_PYPI_REPO_NAME}`:

| Event | Subpath |
|---|---|
| `push` to `pull-request/<pr_id>` (copy-pr-bot mirror) | `pr/<pr_id>/<commit_sha>/<run_id>/<run_attempt>/<arch>/` |
| `push` to `main`, `release/**` | `post-merge/<commit_sha>/<run_id>/<run_attempt>/<arch>/` |

Each path contains the 6 artifacts from one arch: 4 manylinux wheels
(cp310-cp313), 1 `py3-none-any` wheel, and 1 sdist. The upload step is
gated on the `automated-release` GitHub environment, which holds three
secrets: `ARTIFACTORY_URL`, `ARTIFACTORY_TOKEN` (JFrog identity token),
and `ARTIFACTORY_PYPI_REPO_NAME`.

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

The SGLang example image starts from the known-good release image
`lmsysorg/sglang:v0.5.13.post1` and installs the ModelExpress Python package
with `--no-deps` so the base image's CUDA/NIXL/Torch dependency stack stays
intact.

For the Mooncake TransferEngine SGLang example, add
`--build-arg INSTALL_MOONCAKE=true` to install
`mooncake-transfer-engine-cuda13`.

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

ModelExpress supports GPU-to-GPU model weight transfers between supported inference instances using NVIDIA NIXL over RDMA. vLLM 0.23.0 and newer recognize `--load-format modelexpress` natively, which runs the priority chain P2P RDMA -> InstantTensor -> ModelStreamer -> GDS -> native loader; the ModelExpress Python package must still be installed, and `mx` remains a backward-compatible alias. SGLang uses `remote_instance` with the `modelexpress` backend; see [SGLang Clients](#sglang-clients).

### Cross-Vendor (CUDA/XPU) Compatibility

ModelExpress supports cross-family weight transfer between `cuda` and `xpu` only for unquantized model weights.

A weight identity is considered unquantized when:

- `SourceIdentity.quantization` is empty or `none`
- `SourceIdentity.dtype` is on the plain-dtype allowlist (`float16`, `bfloat16`,
  `float32`, and their aliases). The check is an allowlist, not a denylist: any
  dtype not on it (for example `int8`, `int4`, `qint8`, `auto`, or an unknown
  future quantized dtype) fails closed and is treated as quantized, so a
  hardware-specific layout is never silently admitted cross-vendor.

Quantized models are not transferred across accelerator families. This includes models using quantization methods or storage formats such as:

- `fp8`
- `fp4`
- `nvfp4`
- `mxfp4`
- `mxfp8`
- `awq`
- `gptq`
- other non-empty quantization methods

The reason is that ModelExpress transfers post-processed in-memory tensor bytes, not raw checkpoint files. For quantized models, the post-processed layout can depend on the accelerator family, GPU architecture, selected kernel, and framework quantization backend. For example, FP8 scale packing or FP4/NVFP4 swizzling may differ between CUDA and XPU kernels. Copying those bytes across vendors can make the transfer succeed while causing silent inference corruption.

If a target finds only cross-family sources for a quantized model, it skips P2P RDMA and falls through to the next load strategy, such as GDS or disk loading. Expect a slower cold start instead of an RDMA receive in mixed CUDA/XPU fleets serving quantized weights.

Same-family transfer is unaffected. Quantized transfer remains allowed for:

- `cuda` to `cuda`
- `xpu` to `xpu`

because source and target are expected to use the same accelerator-family post-processing layout.

Cross-family transfers also require the source manifest and the target's
registered tensors to name the exact same set. A tensor-name mismatch (a
local-only or source-only name) or a zero-match transfer fails closed with a
`ManifestMismatchError` rather than transferring a subset. This prevents a
vendor-specific hidden or derived tensor from leaving part of the target at
dummy values while RDMA reports success. Same-family transfers keep tolerating
subset transfers (unmatched names are warned, not fatal).

Future work may validate specific quantized CUDA/XPU combinations on hardware. If a specific pair is proven inference-correct after RDMA, ModelExpress may add an explicit allowlist for that quantization/layout pair. ModelExpress does not currently dequantize, requantize, or convert post-processed tensor layouts during transfer. See the accelerator-compatibility rule in [`ARCHITECTURE.md`](ARCHITECTURE.md).

### Choosing a Metadata Backend

Pick based on workload, not operational preference. The choice has structural consequences for what the system can do.

| Workload shape                                                         | Backend          | Why                                                                                                                                            |
|------------------------------------------------------------------------|------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| Stable-weight inference. Weights fixed at pod startup, no mid-life refit. Simple K8s deployment. | `k8s-service`    | Lowest deployment footprint. No server, no Redis, no CRDs. Matches the homogeneous pool assumption that Service-routing requires.             |
| Future RL refit workflows (under development). Training updates weights every step and rollout workers need per-worker sources. | `redis` or `kubernetes` | The central store provides the per-worker addressability required by the planned receiver-driven refit workflow. Selecting this backend does not enable end-to-end live refit today. |
| Future live fine-tune broadcasts (under development). New checkpoints are pushed to running replicas. | `redis` or `kubernetes` | These workflows require the same per-worker addressability. The `k8s-service` backend cannot swap a live pod's `mx_source_id` without restarting the pod. |
| Mixed-version fleet. Multiple revisions serving concurrently, callers dispatch by revision. | `redis` or `kubernetes` | Central store indexes by `mx_source_id`, so multiple identities coexist cleanly. k8s-service requires one Service pool per identity.          |
| Heterogeneous hardware. Some sources on H100, some on B200, callers match on topology. | `redis` or `kubernetes` | Central store carries per-worker metadata including identity fields; k8s-service's pool assumption requires all pods to be interchangeable.   |
| Multiple checkpoints in parallel (base + LoRA, fp16 + nvfp4, etc.).   | Either           | Different `SourceIdentity` produces different `mx_source_id`. Each identity gets its own Service (k8s-service) or its own source records (central). Both work. |

The central-coordinator backends (`redis`, `kubernetes`) are the default. Reach for `k8s-service` specifically when the deployment meets three criteria: (1) weights stay fixed for each pod's lifetime, (2) every pod behind a given Service serves the exact same checkpoint, and (3) dropping the `modelexpress-server` / Redis / CRD components is a material simplification.

Receiver-driven RL refit and live fine-tune broadcast are under development. The table identifies the metadata backend required by those future workflows; it does not describe a currently supported end-to-end refit path.

See [`K8S_SERVICE_BACKEND.md`](K8S_SERVICE_BACKEND.md) for the design rationale, limitations, and the structural reasons these backend families differ.

### P2P Environment Variables

`MX_SERVER_ADDRESS` is the variable ModelExpress is standardizing on for the client's gRPC server address; `MODEL_EXPRESS_URL` is deprecated and will be removed in a future release. During the transition, set both to the same value: some client paths (notably the TRT-LLM live-transfer integration) currently read only `MODEL_EXPRESS_URL`, and when both are set `MODEL_EXPRESS_URL` takes precedence. Do not drop `MODEL_EXPRESS_URL` yet.

| Variable | Default | Description |
|----------|---------|-------------|
| `MX_METADATA_BACKEND` | (required on server; `""` on client) | Server: `redis` or `kubernetes`. Client: `""`/`server`/`redis`/`kubernetes` (central server) or `k8s-service` (decentralized via K8s Service routing). |
| `MX_SERVER_ADDRESS` | `localhost:8001` | Client's gRPC server address (recommended; ignored when client uses `k8s-service` backend) |
| `MODEL_EXPRESS_URL` | `localhost:8001` | Deprecated in favor of `MX_SERVER_ADDRESS`. Still read by all client paths and still takes precedence when both are set, because the TRT-LLM live-transfer integration reads only this name. It is removed once that path reads `MX_SERVER_ADDRESS`; until then set both to the same value. |
| `MX_DISABLE_PATCHES` | `0` | Emergency escape hatch that skips all runtime compatibility patches. Set to `1`, `true`, `yes`, or `on` if a patch is incompatible with the installed engine. |
| `MX_P2P_SOURCE_SELECTOR` | `random` | P2P source-ordering policy for the RDMA load path. `random` (behavior-preserving default; local-RNG shuffle) or `rendezvous_hash` (stateless deterministic spreading via HRW hashing; stable across restarts and minimally disrupted by source-set changes). Unknown values log a warning and fall back to `random`. Ordering only — the `MAX_SOURCE_RETRIES=3` retry budget is unchanged. |
| `MX_METRICS_ENABLED` | `0` | Opt-in Prometheus metrics collector for the client. `1` enables the collectors (requires the `metrics` extra, `prometheus-client`, which the vLLM, SGLang and TensorRT-LLM engine images already provide). Off by default; selection signals are always emitted as structured logs regardless. See [METRICS.md](METRICS.md). |
| `PROMETHEUS_MULTIPROC_DIR` | (unset) | Pod-local directory shared by every rank, so one endpoint serves the merged union of all of them. **Required on any pod with more than one worker process**, and it must be set in the pod manifest: `prometheus_client` latches its value class at import, so assigning it from Python produces zero data with no error. Wipe it at the container entrypoint with `python -m modelexpress.metrics --reset`, never from worker code. |
| `MX_METRICS_PORT` | (unset) | With metrics enabled, serve a pull `/metrics` endpoint on this port. One rank per pod wins the bind and serves every rank's data; the rest re-attempt periodically so the endpoint migrates if the winner exits. |
| `MX_METRICS_PUSHGATEWAY` | (unset) | With metrics enabled, push to this Pushgateway host:port. An escape hatch for pure-batch pods; mutually exclusive with `MX_METRICS_PORT`, enforced in code, because running both double-counts every series. One push per pod, keyed on the pod UID. |
| `MX_METRICS_BIND_RETRY_SECS` | `15` | How often a rank that lost the `/metrics` bind re-attempts it, so endpoint ownership migrates when the current owner exits. |
| `MX_METRICS_SOURCE_ID_LABEL` | `0` | Restore the per-peer `source_worker_id` label on `mx_p2p_source_selections_total`, for comparing source-selection policies. **Benchmark runs only:** the id is `uuid4().hex[:8]` minted per process, so its label domain grows with process count over time rather than with cluster size. Point such a run at a throwaway Prometheus. |
| `MX_METRICS_SCHEME` | `""` | Optional run/scheme label, so multiple runs compare on one dashboard. On the client it is a label on `mx_build_info` **and** on every `mx_p2p_*` family; on the server it is on `mx_build_info` only. |
| `MX_POOL_REG` | `0` | Allocation-level NIXL registration via `cuMemGetAddressRange`. Registers each unique cudaMalloc block instead of each tensor, typically 80-99% fewer registrations, without changing transfer semantics. `MX_VMM_ARENA=1` uses direct arena registration and does not require pool-reg. |
| `MX_VMM_ARENA` | `0` | Route weight allocations into a CUDA VMM arena via PyTorch's `CUDAPluggableAllocator`, then register the used arena range as one NIXL MR with dmabuf at end-of-load. Reserves 16.0 TiB of VA by default, with no physical commit until allocations are mapped. Requires the `modelexpress.vmm._alloc_ext` C extension to have built at install time; if it did not, this flag is a no-op with a warning and the loader falls back to the pool-reg path. See [VMM Arena](#vmm-arena-single-mr-registration). |
| `MX_ARENA_SINGLE_MR` | `0` | Keep single-MR arena registration even when the arena spans several `cuMemCreate` handles. Only safe on transports that can register across handles (dmabuf/IB); cuda_ipc cannot, so the default falls back to per-tensor registration. See [VMM Arena](#vmm-arena-single-mr-registration). |
| `UCX_CUDA_COPY_REG_WHOLE_ALLOC` | (UCX default) | Set to `off` with `MX_VMM_ARENA=1` on any UCX predating the `cuda_copy_md` length-truncation fix (openucx/ucx#11461). Scoped to the `cuda_copy` transport; it does not affect `cuda_ipc`. |
| `MX_NIXL_BACKEND` | `UCX` | NIXL backend for GPU-to-GPU RDMA. `UCX` (default) for InfiniBand / RoCE. `LIBFABRIC` for AWS EFA — see [NIXL Backend Selection](#nixl-backend-selection). |
| `MX_RDMA_NIC_PIN` | (unset) | Per-rank IB NIC pinning. `auto` runs a topology probe; comma-separated NIC list is an explicit override. Workaround for openucx/ucx#11259. |
| `MX_RDMA_NIC_PIN_MIN_RATE_GBPS` | (auto, max-rate filter) | Override the auto-detect rate filter with an explicit lower bound (Gb/s). |
| `MODEL_EXPRESS_LOG_LEVEL` | (inherits vLLM) | Override log level for `modelexpress.*` loggers. `DEBUG` enables per-tensor checksums and adopted tensor details |
| `MX_P2P_METADATA` | `1` | Enable P2P metadata exchange (source workers only). Set to `0` to publish full metadata through a central-coordinator backend. This setting is ignored on backends that require P2P metadata, currently `k8s-service`. |
| `MX_METADATA_PORT` | `5555` | Base NIXL listen port; effective port is `MX_METADATA_PORT + device_id` |
| `MX_REFIT_METADATA_PORT` | `7555` | Base NIXL listen port for an RL generator's refit client; effective port is `MX_REFIT_METADATA_PORT + device_id`, separate from a boot-time loader manager |
| `MX_WORKER_GRPC_PORT` | `6555` | Base worker gRPC port for P2P tensor and artifact manifest serving |
| `MX_WORKER_HOST` | (auto-detect) | Override worker IP/hostname for P2P endpoints |
| `MX_ARTIFACT_TRANSFER` | `0` | Opt in to cache artifact transfer. The vLLM loader uses it for torch compile, Triton, DeepGEMM, TileLang, CuTe DSL, and FlashInfer JIT caches, including persistent autotune files when supported by vLLM. The SGLang NIXL loader uses the same artifact path for compatible torch compile, Triton, TVM-FFI, DeepGEMM, TileLang, CuTe DSL, and FlashInfer caches. Requires the P2P metadata path; if `MX_P2P_METADATA=0`, the loader logs a warning and skips artifact transfer. |
| `MX_ARTIFACT_TRANSFER_CHUNK_SIZE` | `67108864` | Artifact transfer chunk size in bytes. Default is 64 MiB; maximum is 4 GiB. Larger values reduce manifest/RPC overhead but increase registered DRAM buffer memory, approximately `chunk_size * max_inflight_chunks` per source and target worker. |
| `MX_ARTIFACT_BUNDLE_ROOT` | `$TMPDIR/modelexpress-artifacts` | Staging root for tarred cache artifact bundles. |
| `MX_ARTIFACT_READY_URL` | Framework default | Readiness endpoint polled before source workers publish weight metadata or prepare and publish cache artifact bundles. Defaults to `http://127.0.0.1:8000/health` for vLLM and `http://127.0.0.1:30000/health` for SGLang. On the non-head nodes of a multi-node engine a loopback host is rewritten onto the head's address, preserving the configured port and path; a non-loopback host is used verbatim. See [Multi-node readiness](#multi-node-readiness). |
| `MX_ARTIFACT_READY_TIMEOUT_SECS` | `1800` | Maximum time to wait for readiness and successful artifact publication before giving up. |
| `MX_ARTIFACT_COMPILE_CONFIG_DIGEST` | `""` (unset) | Adds compile configuration as a partitioning dimension for the torch compile cache artifact source pool. Workers that share a value discover each other's caches; workers with different values do not. Unset removes **only this dimension** — the pool is still partitioned by every other `SourceIdentity` field (model, tensor/pipeline/expert parallel size, dtype, quantization, revision, vLLM/torch/CUDA/Triton versions, GPU arch), so workers matching on all of those share one pool even when their compile configurations differ. See [Pairing workers by compile configuration](#pairing-workers-by-compile-configuration). |
| `MX_MODEL_REVISION` | (from vLLM config) | Override for `SourceIdentity.revision`. Pin to the exact HF commit SHA / checkpoint version so `mx_source_id` is content-addressed. Required for decentralized backends where no central coordinator tracks versions. |
| `MX_K8S_SERVICE_PATTERN` | `mx-sources` | DNS template for the `k8s-service` backend. `{rank}` is substituted with the worker's own rank. If the resolved pattern has no `:port`, the client auto-appends `:{MX_WORKER_GRPC_PORT + rank}` (multi-GPU-per-pod shape); if it has an explicit port, that port is used verbatim (1-GPU-per-pod shape). |
| `MX_K8S_SOURCE_RETRIES` | `5` | `k8s-service` backend: max retries on `FAILED_PRECONDITION` (revision mismatch during rolling updates). Each retry opens a fresh gRPC channel so kube-proxy re-picks a backend. |
| `MX_K8S_SOURCE_BACKOFF_SECONDS` | `0.5` | `k8s-service` backend: sleep between retry attempts. |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL (Redis backend only) |
| `MX_METADATA_NAMESPACE` | `default` | K8s namespace for CRD backend |
| `VLLM_RPC_TIMEOUT` | `7200000` | vLLM RPC timeout in ms (2 hours for large models) |
| `VLLM_PLUGINS` | - | For vLLM versions older than 0.23.0, set to `modelexpress` to register the `modelexpress` and `mx` loaders. vLLM 0.23.0 and newer recognize the load format natively. |

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
that single MR — but only when the arena is backed by a single physical
allocation. An arena that spans several `cuMemCreate` handles falls back to
per-tensor registration; see [Multi-handle arenas](#multi-handle-arenas) below.

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

Set `UCX_CUDA_COPY_REG_WHOLE_ALLOC=off` on any UCX predating the
`cuda_copy_md` length-truncation fix (openucx/ucx#11461). Without it, UCX
can truncate a multi-handle VMM registration to the first physical
handle, and RDMA operations that cross into later handles fail. See the
reproducer and fix notes in this gist:
<https://gist.github.com/nicolasnoble/e0e57eb5a1b902057ae3d1df59c039cf>.

That knob covers the `cuda_copy` transport only. It has no effect on
`cuda_ipc`, which has its own multi-handle limitation described in
[Multi-handle arenas](#multi-handle-arenas) below.

#### Multi-handle arenas

A CUDA fabric/IPC handle names exactly one `cuMemCreate` allocation. UCX
cuda_ipc resolves a registered region with `cuMemRetainAllocationHandle` and
`cuMemGetAddressRange`, both of which report the allocation holding the base
pointer rather than the whole reserve. Registering a multi-allocation arena as
one MR therefore publishes an rkey covering only its first chunk, and the peer
reads past what it mapped — measured on GB200 MNNVL as a segfault in
`cuMemcpyDtoDAsync_v2` with an arena spanning 1019 chunks.

ModelExpress detects this (`live_allocation_count > 1`) and falls back to
per-tensor registration, which is correct because each arena allocation is one
handle, so every tensor lies wholly inside one. The log line names the count:

```text
register_arena: arena spans 1019 physical allocations; a single MR would publish
an rkey covering only the first, which cuda_ipc cannot address. Falling back to
per-tensor registration ...
```

`UCX_CUDA_COPY_REG_WHOLE_ALLOC=off` does not cover this case: it applies to
cuda_copy, while the truncation above is in cuda_ipc, and UCX 1.21 has no
cuda_ipc equivalent. Upstream fixes are in flight for both sides —
[openucx/ucx#11283](https://github.com/openucx/ucx/pull/11283) for cuda_ipc and
[openucx/ucx#11461](https://github.com/openucx/ucx/pull/11461) for the
cuda_copy/dmabuf length truncation.

Set `MX_ARENA_SINGLE_MR=1` to keep the single-MR path on deployments where it
was validated, i.e. dmabuf/IB, where `ibv_reg_dmabuf_mr` does span several
handles.

### P2P Metadata Exchange

P2P metadata exchange is enabled by default. Source workers expose their own per-worker gRPC `WorkerService` (the `WorkerGrpcServer` on `MX_WORKER_GRPC_PORT`) and their NIXL agent metadata directly on the worker's NIXL listen thread (`MX_METADATA_PORT`). Targets fetch tensor manifests or artifact manifests directly from the source worker rather than pulling them through the central store. For file-backed cache artifacts, targets also call `PrepareArtifactChunk` and `ReleaseArtifactChunk` on this worker service while bytes move through NIXL into target-local staging, then install the staged artifact into the runtime cache directory. The division of responsibility depends on which metadata backend is in use:

- **Central-coordinator backends (`redis`, `kubernetes`):** the source publishes only a lightweight pointer (its `worker_grpc_endpoint` and NIXL listen address) to the central server, and targets use that pointer to connect directly to the source for the MB-scale data. Set `MX_P2P_METADATA=0` to publish full tensor metadata (NIXL blobs + tensor descriptors) to the central server instead. Targets auto-detect which mode a source is using based on whether `worker_grpc_endpoint` is populated in the server's metadata; no configuration is needed on the target side.
- **`k8s-service` backend:** auto-enabled for tensor metadata. The backend declares itself decentralized (via a class attribute `REQUIRES_P2P_METADATA = True`), so the client forces the P2P tensor path regardless of the env var. Deployers don't need to set `MX_P2P_METADATA` themselves. If the env var is explicitly set to `0` alongside this backend, the client logs a warning that the setting is ignored but otherwise proceeds correctly. File-backed artifact transfer currently requires a central-coordinator backend (`redis` or `kubernetes`) because `k8s-service` does not yet publish `artifact_source` discovery metadata.

Set `MX_METADATA_PORT` and `MX_WORKER_GRPC_PORT` to fixed ports when running in K8s (port 0 picks an ephemeral port). Set `MX_WORKER_HOST` if the pod IP auto-detection doesn't produce a routable address.

For cache artifact transfer, set `MX_ARTIFACT_TRANSFER=1` on source and target workers. The default P2P metadata path is also required; if it was disabled, set `MX_P2P_METADATA=1`. The vLLM and SGLang NIXL loaders install compatible artifacts before model initialization, then schedule publisher threads after successful load. Each publisher waits for readiness before publishing local cache directories and waits for their file count, total size, and max mtime to settle before sealing the artifact.

vLLM publishes torch compile (`VLLM_CACHE_ROOT/torch_compile_cache`), Triton (`TRITON_CACHE_DIR`, or `~/.triton/cache`), DeepGEMM (`DG_JIT_CACHE_DIR`, or `VLLM_CACHE_ROOT/deep_gemm`), TileLang (`TILELANG_CACHE_DIR`, or `~/.tilelang/cache`), CuTe DSL (`CUTE_DSL_CACHE_DIR`, or `$TMPDIR/<user>/cutlass_python_cache`), and FlashInfer (`FLASHINFER_WORKSPACE_BASE/.cache/flashinfer`, or `~/.cache/flashinfer`) caches. The FlashInfer artifact also includes vLLM's persistent autotune directory from `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`, or `VLLM_CACHE_ROOT/flashinfer_autotune_cache` when unset; ModelExpress does not change either path.

SGLang's NIXL loader publishes torch compile (`TORCHINDUCTOR_CACHE_DIR`, or PyTorch Inductor's runtime `cache_dir()`), Triton (`TRITON_CACHE_DIR`, or `~/.triton/cache`), TVM-FFI (`TVM_FFI_CACHE_DIR`, or `~/.cache/tvm-ffi`), DeepGEMM (`SGLANG_DG_CACHE_DIR`, or `~/.cache/deep_gemm`), TileLang (`TILELANG_CACHE_DIR`, or `~/.tilelang/cache`), CuTe DSL (`CUTE_DSL_CACHE_DIR`, or `$TMPDIR/<user>/cutlass_python_cache`), and FlashInfer (`FLASHINFER_WORKSPACE_BASE/.cache/flashinfer`, or `~/.cache/flashinfer`) caches. The FlashInfer artifact also includes SGLang's persistent autotune directory from `SGLANG_CACHE_DIR/flashinfer/autotune`, or `~/.cache/sglang/flashinfer/autotune` when unset. SGLang runs that autotuner only for eligible FlashInfer MoE or FP4 backends; DeepGEMM + DeepEP does not produce an autotune cache. SGLang TransferEngine transport currently remains weight-only for ModelExpress artifact transfer because cache artifact bytes move through the NIXL artifact path.

Artifacts are sealed as tar archives holding regular files and directories
only. Symlinks in a cache directory are left out of the archive rather than
followed or copied verbatim, because engines treat them as derived state and
rebuild them on demand: FlashInfer, for example, relinks each
`trtllmGen_*_export` include path to its cubin directory on every JIT module
lookup, so a link carried across pods would be deleted and recreated anyway.
Skipping one costs the link entry alone -- neither the packaging walk nor tar
descends through a symlinked directory, so no subtree is lost. A link that
resolves inside the cache root is logged at debug (its target is archived under
its real path); a link that leaves the root or dangles is logged as a warning
naming the paths. Symlinks never fail an artifact publish.

#### Pairing workers by compile configuration

A torch compile cache is only reusable by a worker whose compile configuration
matches the one that produced it. vLLM enforces this itself: it derives its own
cache directory name from a hash that includes scheduler and compilation
settings, so a worker started with a different `--max-num-batched-tokens` or
`--max-model-len` looks in a different directory and recompiles from scratch.

ModelExpress cannot reproduce that hash at load time — part of it depends on
files traced by Dynamo, which are only known after compilation. Artifact
discovery therefore keys on `MX_ARTIFACT_COMPILE_CONFIG_DIGEST`, and **that
variable is unset by default**.

The source pool is still partitioned by every other `SourceIdentity` field, so
workers differing in model, parallelism sizes, dtype, quantization, revision,
vLLM/torch/CUDA/Triton version, or GPU architecture never see each other's
caches. What an unset digest removes is **only the compile-configuration
dimension**: workers matching on all of the above land in one pool even when
their compile settings differ, so one can be handed a cache built under a
different compile configuration. The bytes transfer and install successfully;
vLLM then ignores them and recompiles. The result is wasted transfer, not
incorrect output.

That single missing dimension is exactly the one that separates prefill from
decode in a disaggregated deployment — the two roles otherwise agree on every
field above.

Set the variable to a distinct value per compile configuration whenever a
deployment runs more than one — most commonly prefill and decode workers in a
disaggregated setup, which differ in `max_num_batched_tokens`:

```yaml
# prefill workers
- name: MX_ARTIFACT_COMPILE_CONFIG_DIGEST
  value: "prefill-mnbt8192"
# decode workers
- name: MX_ARTIFACT_COMPILE_CONFIG_DIGEST
  value: "decode-mnbt1024"
```

The value is opaque to ModelExpress; it only has to be equal across workers that
should share caches and different across those that should not. Note that this
partitions the pool but does not by itself make prefill and decode share a
cache — they cannot, because vLLM's own cache keys already differ.

To check whether a transferred cache was actually used, look for the
effectiveness line the vLLM loader emits once the engine is up:

```text
vLLM selected torch.compile cache directory <path>, which ModelExpress installed
```

A `WARNING` naming a different directory means the installed artifact was inert
and this worker recompiled — the signal that the pool needs partitioning. You
can confirm independently by comparing the `Using cache directory:` hash that
each worker logs.

In multi-node deployments, artifact metadata records the framework `node_rank`, so each target node selects the corresponding source node without making the artifact worker-specific. If artifact transfer is enabled while P2P metadata is disabled, the loader logs a warning and skips artifact transfer. Artifact discovery currently requires a central-coordinator backend (`redis` or `kubernetes`), and Kubernetes deployments must use the matching `ModelMetadata` CRD containing `status.worker.artifactSource.nodeRank`.

#### Multi-node readiness

Only the head node of a multi-node engine serves HTTP; the others run headless
(for vLLM, `--headless`, which starts the executor and no API server), so a
pod-local address such as `http://127.0.0.1:9090/health` can never be satisfied
there. Those nodes would never publish, leaving no source with `nodeRank=N` for
node N of a target replica to install.

The client therefore rewrites a **loopback** readiness host onto the head's
address, keeping the configured port and path; a configured **non-loopback**
host is used verbatim. The head address comes from the engine's own
distributed-init address (for vLLM, `parallel_config.master_addr`, populated
from `--master-addr` by the orchestrator), falling back to
`LWS_LEADER_ADDRESS`. Since the distributed process group is already
established through that address, no extra configuration is needed: the shipped
multi-node examples work unchanged with their loopback `MX_ARTIFACT_READY_URL`.

Cache artifacts may contain executable code. Transfer checksums detect corruption
but do not authenticate the publishing replica or attest the artifact. Enable
artifact transfer only within a trusted deployment, and network-isolate the MX
server and worker gRPC endpoints from untrusted clients. ModelExpress does not
currently sign cache artifacts.

RL refit has the same trusted-network requirement. Its trainer-local
`RefitWorkerService` serves exact-version manifests over plaintext gRPC; the
manifest digest detects corruption but does not authenticate the trainer.

Canonical S3/XOR trainers consume Hugging Face tensor buckets produced by the
training framework. Framework-native bucket settings remain the default.
Integrations may use `MX_REFIT_DELTA_BUCKET_BYTES` as an explicit override, or
its 512 MiB default when they have no native setting. `MX_REFIT_DELTA_WORKERS`
(default `min(32, CPU count)`) controls delta-processing concurrency.
Framework integrations read the bucket-size setting before constructing the
stream; ModelExpress preserves the supplied bucket boundaries.

### Server-Backed Model Cache (No Shared Storage)

For workers that cannot reach the Hugging Face Hub themselves, ModelExpress Server can act as the only route to the model. The worker asks the server for repository files; the server downloads the model once on a cold miss and serves every later worker from its own cache.

Weights and everything else are fetched at different times. Config, tokenizer, and index files arrive before the engine starts, because the engine resolves the model path (and fails offline) long before any weight loader runs. Weights stay behind the strategy chain, so a live P2P source is still the first choice and the server is only asked after `RdmaStrategy` finds nothing. Fetching metadata early does not weaken P2P-first — no weight moves on that path.

Enable it on the worker:

```yaml
MODEL_EXPRESS_URL: http://<model-express-service>:8001
MODEL_EXPRESS_NO_SHARED_STORAGE: "1"
MODEL_EXPRESS_CACHE_DIRECTORY: /home/dynamo/.cache/huggingface/hub
HF_HUB_CACHE: /home/dynamo/.cache/huggingface/hub
HF_HUB_OFFLINE: "1"
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_EXPRESS_NO_SHARED_STORAGE` | `0` | Fetch repository files from ModelExpress Server. When unset, nothing changes: no extra RPCs and no change to P2P or local loading. |
| `MODEL_EXPRESS_URL` | unset | ModelExpress Server address. Required together with the switch above; without an address the feature stays off. |
| `MX_SERVER_ADDRESS` | unset | Alternative spelling of the server address, accepted for parity with the P2P client. Either variable satisfies the requirement. |
| `MODEL_EXPRESS_CACHE_DIRECTORY` | `HF_HUB_CACHE` | Where the worker installs snapshots. Point it at the same path as `HF_HUB_CACHE` so the engine reads what ModelExpress wrote. |
| `MODEL_EXPRESS_TRANSFER_CHUNK_SIZE` | `1048576` | gRPC file-stream chunk size in bytes. Values outside 1..`MAX_CHUNK_SIZE` fall back to the default rather than failing startup. |

Requirements and limits:

- ModelExpress Server needs a writable cache directory, egress to Hugging Face, and `HF_TOKEN` for private repositories. The worker needs none of these.
- The server must be from a release newer than v0.5.0 — the first generation that keys registry entries on the weight mode. The metadata phase claims a metadata-only download; an older server records that claim against the model name alone, which marks the model complete, so the weight phase finds nothing left to fetch and the worker falls through to a native load that an offline pod cannot perform.
- Mount the cache path as a volume shared by every container that touches it. Without a volume the snapshot lands in the container's writable layer, invisible to other containers and lost on restart.
- A pinned revision is honoured. The engine's `revision` — a commit hash, a branch, or a tag — is what the server is asked for, and the snapshot is installed so that the engine's own offline lookup finds it: under `snapshots/<commit>/`, plus a `refs/<revision>` entry unless the requested revision is already the commit hash, which resolves by directory name. A pin for anything other than `main` leaves `refs/main` untouched, so it cannot misdirect a later unpinned resolution in this worker or the next one sharing the cache. A server that does not confirm the requested revision fails the install rather than quietly serving its default — one predating pinned-revision support reports no revision at all, and a commit hash that resolves to a different commit is refused outright. This requires `MODEL_EXPRESS_CACHE_DIRECTORY` and `HF_HUB_CACHE` to be the same path, as above.
- The weight phase pins to the commit its snapshot is named after, so a server whose default revision has moved past that snapshot serves the snapshot's own weights instead of failing the phase; when the pinned call fails with a `grpc.RpcError` (resolving a pin needs the Hub, and its failure takes this shape) it degrades to the unpinned request — a download failure the server reports through the status stream is raised, not retried. Weights whose commit differs from the local snapshot directory are refused rather than mixed in.
- `MX_MODEL_REVISION` does not pin anything. It labels the worker's P2P source identity and accepts any string, including one that names no Hugging Face revision at all. Pin through the engine's own revision setting instead.
- Reusing a local snapshot requires the server to name its revision. A server that already holds an unpinned model answers without naming one, so the worker restreams the metadata rather than assume the copy on disk is current. Metadata is small — well under a second — and the stream carries the commit, which makes restreaming the cheap way to stay correct.
- On a cold server the metadata phase waits only for the non-weight files. The weights are downloaded later, and only if P2P found no source. The server keys its registry entry on the weight mode, so the metadata-only request does not mark the model complete.
- The server dedups the upstream download but not the per-worker stream. Concurrent workers on a cold model all wait on one Hugging Face fetch, then each streams its own copy, so N replicas starting together cost N x model size in server egress. Size the server's network accordingly, or stagger large rollouts.
- An unreachable server costs about 20 seconds per worker before loading falls through to the next strategy. That is the TCP connect timeout; a shorter deadline would abort legitimate cold-cache downloads, which can take minutes.

### InstantTensor (Fast Local Safetensors)

InstantTensor loads the model's own safetensors directly onto CUDA using distributed loading, pipelined prefetching, and direct I/O, with GPUDirect Storage when the hardware supports it. It sits right after P2P RDMA in the loading chain: when no peer source is already serving, it is the fastest local-disk path before falling back to ModelStreamer, GDS, or the native loader. Unlike ModelStreamer it needs no `MX_MODEL_URI`; it reuses vLLM's built-in `--load-format instanttensor` path, so the engine resolves the model's weight files (downloading from the Hugging Face Hub into the local cache first if they are not already local).

The strategy is enabled by default. The `instanttensor` package is a core dependency on Linux (installed automatically alongside `runai-model-streamer`), so no extra install step is needed. The strategy activates on a CUDA device **when the engine adapter implements the InstantTensor capability**. Currently only the vLLM adapter implements it; on engines that do not (for example SGLang today), the strategy falls through even when `instanttensor` and a CUDA device are available. If the package is unavailable (for example on a non-Linux platform) the chain simply skips to the next strategy.

| Variable | Default | Description |
|----------|---------|-------------|
| `MX_INSTANT_TENSOR` | `1` | Enable the InstantTensor strategy. Set to `0` to disable it and fall through to ModelStreamer/GDS/native loading. |

InstantTensor also honors its own `INSTANTTENSOR_BACKEND` environment variable (`URING`, `AIO`, `CUFILE` for GDS, `MMAP`) for selecting the I/O backend; ModelExpress passes it through unchanged.

### ModelStreamer (Object Storage & Local Disk)

ModelStreamer reads safetensor ranges concurrently through a bounded CPU staging buffer and pipelines completed tensors into the inference engine while later reads continue. It supports S3, GCS, Azure Blob Storage, and local filesystem (PVC) paths. This storage-loading path does not require P2P by itself. If the same deployment also enables ModelExpress P2P metadata and RDMA resources, later replicas can receive weights from an already-loaded source instead of streaming from storage again.

All storage backends (S3, GCS, Azure) are included as core dependencies — no extra install step needed. The strategy activates when `MX_MODEL_URI` is set. See [`../examples/model_streamer_k8s/`](../examples/model_streamer_k8s/) for Kubernetes examples, including the Azure Blob recipe.

**General configuration:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MX_MODEL_URI` | (none) | Model location. Must be set to enable ModelStreamer. Accepts a remote URI (`s3://bucket/model`, `gs://...`, `az://...`) or absolute local path (`/models/deepseek-ai/DeepSeek-V4-Pro`). |
| `MX_MS_DISTRIBUTED` | `1` | Divide ModelStreamer reads across tensor-parallel ranks and share the results instead of having every rank read the full checkpoint. Requires tensor parallelism > 1 and a CUDA-capable platform; a no-op at TP1. On by default. Set to `0` to disable. |
| `RUNAI_STREAMER_CONCURRENCY` | `8` | Number of concurrent read threads |
| `RUNAI_STREAMER_MEMORY_LIMIT` | (none) | CPU staging buffer size in bytes. `0` reuses a single-tensor buffer (most memory efficient). See [runai-model-streamer docs](https://github.com/run-ai/runai-model-streamer). |

With vLLM, `MX_MODEL_URI` can also be a Hugging Face model ID. vLLM first downloads the safetensors into its local Hugging Face cache, then ModelStreamer reads those local files; ModelStreamer does not stream directly from the Hub.

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
          args: ["--model", "$(MODEL_NAME)", "--load-format", "modelexpress", "--tensor-parallel-size", "2"]
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
included in the known-good release image `lmsysorg/sglang:v0.5.13.post1`. The
integration supports both NIXL and Mooncake TransferEngine transports. See
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
  -d '{"model": "deepseek-ai/DeepSeek-V4-Pro", "prompt": "Hello", "max_tokens": 10}'
```

## Performance Reference

| Model | Total Data | Transfer Time | Per-Worker Speed |
|-------|-----------|---------------|------------------|
| DeepSeek-V3 (671B, FP8) | 681 GB (8 GPUs) | ~15 seconds | ~45 Gbps |
| Llama 3.3 70B | 140 GB (8 GPUs) | ~5 seconds | ~28 Gbps |
