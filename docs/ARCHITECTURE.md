<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ModelExpress architecture and internals

This is the implementation reference for the ModelExpress codebase. Start with the [documentation hub](README.md) or [Choose a ModelExpress path](guides/choose-a-path.md) for deployment questions. Use [Configuration](CONFIGURATION.md) for current defaults, [Integrations](integrations/README.md) for runtime setup, and [Troubleshooting](TROUBLESHOOTING.md) for operational symptoms. For contribution guidelines and dev setup, see [`CONTRIBUTING.md`](../CONTRIBUTING.md). For CLI usage, see [`CLI.md`](CLI.md). For GCS provider internals, see [`GCS_PROVIDER.md`](GCS_PROVIDER.md).

## Project Overview

ModelExpress is a Rust-based model cache management service and GPU-to-GPU model weight transfer system. It serves two roles:

- **Model Cache Service** - A sidecar alongside inference solutions (vLLM, SGLang, NVIDIA Dynamo) that accelerates model downloads from HuggingFace, NGC, and GCS. Model lifecycle state lives in a distributed registry — Redis or Kubernetes CRDs (`ModelCacheEntry`), selected via `MX_METADATA_BACKEND` — so multiple server replicas can coordinate without a shared-filesystem database. LRU cache eviction runs off the same registry.
- **P2P Weight Transfer** - GPU-to-GPU model weight transfers between inference replicas using NVIDIA NIXL over RDMA/InfiniBand, enabling ~15-second transfers for 681GB models. The Python client includes engine adapters for vLLM and SGLang.

### Current Status

| Model | Status | Transfer Time | Notes |
|-------|--------|---------------|-------|
| DeepSeek-V3 (671B, FP8) | Working | ~15s | 681GB across 8 GPUs @ ~45 Gbps per link |
| Llama 3.3 70B | Working | ~5s | 140GB across 8 GPUs @ ~28 Gbps per link |

## Architecture

```mermaid
graph TD
    subgraph "Model Cache Mode"
        C1[Client CLI / Library] -->|gRPC| S1[ModelExpress Server]
        S1 --> R1[(Redis / K8s CRD)]
        S1 --> HF[HuggingFace Hub]
        S1 --> NGC[NVIDIA NGC]
        S1 --> GCS[Google Cloud Storage]
        S1 --> Cache[Model Cache Dir]
    end

    subgraph "P2P Transfer Mode"
        subgraph "Node A"
            A[Inference Engine + MxModelLoader]
        end
        subgraph "Node B"
            B[Inference Engine + MxModelLoader]
        end
        A -->|gRPC metadata| S2[ModelExpress Server]
        B -->|gRPC metadata| S2
        S2 --> R[(Redis)]
        A -- "RDMA via NIXL" --> B
    end
```

### Components

| Component | Language | Location | Purpose |
|-----------|----------|----------|---------|
| Server | Rust | `modelexpress_server/` | gRPC server: model downloads, cache eviction, P2P coordination |
| Rust Client | Rust | `modelexpress_client/src/` | Client library and CLI tool |
| Python Client | Python | `modelexpress_client/python/` | Inference engine loaders, NIXL transfer manager, gRPC client |
| Common | Rust | `modelexpress_common/` | Protobuf definitions, shared types, provider trait, config |
| Workspace Tests | Rust | `workspace-tests/` | Integration tests and Criterion benchmarks |

## Repository Structure

```text
ModelExpress/
├── Cargo.toml                          # Workspace root (4 members)
├── Cargo.lock
├── docker/
│   ├── Dockerfile                      # Multi-stage production image
│   ├── Dockerfile.client-wheel         # Builds Python client wheels + sdist
│   └── docker-compose.yml              # Single-service dev setup
├── run_integration_tests.sh            # Integration test runner
├── test_client.sh                      # Client test script
├── test_grpc_transfer_k8s.sh           # K8s gRPC transfer test
├── deny.toml                           # cargo-deny config
├── rust-toolchain.toml                 # Rust version pinning
├── rustfmt.toml                        # Formatting config
├── modelexpress-cli-completion.bash    # Shell completions
│
├── modelexpress_server/
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs                     # Server startup, service registration
│       ├── lib.rs                      # Module exports
│       ├── config.rs                   # ServerConfig, layered loading, validation
│       ├── backend_config.rs           # Shared BackendConfig (Redis / K8s) + env parsing
│       ├── cache.rs                    # CacheEvictionService, LRU policy
│       ├── services.rs                 # Health, API, Model gRPC services + ModelDownloadTracker
│       ├── metrics.rs                   # Prometheus registry, mx_build_info, encoding
│       ├── metrics/
│       │   ├── exposition.rs            # HTTP/1.1 /metrics listener (separate port)
│       │   ├── buckets.rs               # Shared histogram bucket boundaries
│       │   ├── grpc.rs                  # Per-RPC tower layer; in-handler outcomes
│       │   ├── backend.rs               # Backend op counters and latency
│       │   ├── registry.rs              # Download claim, lease and status transitions
│       │   └── cache.rs                 # Eviction reasons and refreshed gauges
│       ├── p2p/
│       │   ├── state.rs                # P2pStateManager wrapper
│       │   ├── service.rs              # P2P gRPC service implementation
│       │   ├── source_identity.rs      # SHA256-based mx_source_id computation
│       │   ├── reaper.rs               # Server-side stale source detection and GC
│       │   ├── k8s_types.rs            # P2P CRD type definitions (ModelMetadata)
│       │   ├── backend.rs              # MetadataBackend trait + types
│       │   └── backend/
│       │       ├── redis.rs            # P2P Redis backend
│       │       ├── kubernetes.rs       # P2P Kubernetes CRD backend
│       │       └── instrumented.rs     # Metrics decorator over MetadataBackend
│       ├── refit/                        # Optional RL control plane; see docs/rl/
│       ├── registry/
│       │   ├── state.rs                # RegistryManager wrapper
│       │   ├── stats_refresh.rs        # Background gauge refresh (own interval)
│       │   ├── backend.rs              # RegistryBackend trait + ModelRecord
│       │   ├── entry_key.rs            # EntryKey: model + revision + weight mode
│       │   ├── k8s_types.rs            # ModelCacheEntry CRD type
│       │   └── backend/
│       │       ├── redis.rs            # Redis registry backend
│       │       ├── kubernetes.rs       # K8s ModelCacheEntry registry backend
│       │       └── instrumented.rs     # Metrics decorator over RegistryBackend
│       └── bin/
│           └── config_gen.rs           # Config file generator/migrator
│
├── modelexpress_client/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                      # Client struct, public API
│       └── bin/
│           ├── cli.rs                  # CLI entry point (modelexpress-cli)
│           ├── test_client.rs          # Concurrent/single download tests
│           ├── fallback_test.rs        # Provider fallback tests
│           └── modules/
│               ├── args.rs             # CLI args (Cli struct, embeds ClientArgs)
│               ├── handlers.rs         # CLI command handlers
│               ├── output.rs           # Output formatting (human, JSON, JSON-pretty)
│               └── payload.rs          # JSON payload reader (inline, file, stdin)
│
├── modelexpress_client/python/
│   ├── pyproject.toml                  # Python package config
│   ├── generate_proto.sh               # Proto stub generation script
│   └── modelexpress/
│       ├── __init__.py                 # Package init, vLLM loader auto-registration
│       ├── client.py                   # MxClient gRPC client
│       ├── nixl_transfer.py            # NixlTransferManager
│       ├── source_selection.py         # P2P source-ordering policies (random, rendezvous_hash)
│       ├── metrics.py                   # Opt-in Prometheus metrics collector (source-selection group today)
│       ├── gds_transfer.py             # GPUDirect Storage transfer support
│       ├── gds_loader.py               # GDS model loader
│       ├── adapter.py                  # EngineAdapter contract and strategy errors
│       ├── vllm_loader.py              # Compatibility shim for engines.vllm.loader
│       ├── metadata/                   # Metadata clients, publishing, heartbeat, worker manifest service
│       │   ├── __init__.py
│       │   ├── artifact_lifecycle.py   # Engine-agnostic cache-artifact lifecycle
│       │   ├── artifact_transfer.py    # Tar/NIXL artifact transfer primitives
│       │   ├── publish.py              # Source identity + metadata publication
│       │   ├── publisher.py            # Source publication and heartbeat signaling
│       │   ├── worker_server.py        # WorkerGrpcServer (P2P tensor/artifact manifests)
│       │   ├── source_id.py            # Python mx_source_id computation
│       │   ├── client_factory.py       # Selects central vs k8s-service metadata client
│       │   └── k8s_service_client.py   # Decentralized k8s-service metadata client
│       ├── load_strategy/              # Loading strategy chain
│       │   ├── __init__.py             # LoadStrategyChain.run()
│       │   ├── context.py              # LoadContext and LoadResult
│       │   ├── draft_gate.py           # Delay main publication until a draft pass completes
│       │   ├── base.py                 # LoadStrategy ABC and shared helpers
│       │   ├── rdma_strategy.py        # RdmaStrategy (P2P GPU transfer via NIXL)
│       │   ├── server_cache_strategy.py # ServerCacheStrategy (weights from MX Server)
│       │   ├── instant_tensor_strategy.py # InstantTensorStrategy (fast local safetensors)
│       │   ├── model_streamer_strategy.py # ModelStreamerStrategy (S3/GCS/Azure/local)
│       │   ├── gds_strategy.py         # GdsStrategy (GPUDirect Storage)
│       │   └── default_strategy.py     # DefaultStrategy (engine-native fallback)
│       ├── engines/
│       │   ├── vllm/                   # vLLM integration
│       │   │   ├── __init__.py         # vLLM loader registration
│       │   │   ├── adapter.py          # VllmAdapter and context builder
│       │   │   └── loader.py           # MxModelLoader implementation
│       │   └── sglang/                 # SGLang integration
│       │       ├── __init__.py
│       │       ├── adapter.py          # SglangAdapter; delegates cold reads to SGLang
│       │       ├── checkpoint_selection.py # Role-specific shard filtering before reads
│       │       ├── draft_weights.py    # Model-specific main/draft selection rules
│       │       ├── loader.py           # MxModelLoader for remote_instance backend
│       │       └── role_aware_streamer.py # Lazily imported SGLang shard-filtering loader
│       ├── tensor_utils.py             # Tensor collection, checksums, storage views
│       ├── transfer_safety.py          # Model feature detection for the P2P gate
│       ├── rank_utils.py               # Rank detection utilities
│       ├── vllm_worker.py              # Compatibility worker for older manual registration
│       ├── types.py                    # TensorDescriptor, WorkerMetadata dataclasses
│       ├── p2p_pb2.py                  # Generated protobuf stubs
│       └── p2p_pb2_grpc.py             # Generated gRPC stubs
│   └── modelexpress_rl/                 # Optional RL source package; see docs/rl/
│
├── modelexpress_common/
│   ├── Cargo.toml
│   ├── build.rs                        # tonic-build: compiles the shared gRPC definitions
│   ├── proto/
│   │   ├── health.proto                # HealthService
│   │   ├── api.proto                   # ApiService
│   │   ├── model.proto                 # ModelService
│   │   └── p2p.proto                   # P2pService
│   └── src/
│       ├── lib.rs                      # Module exports, gRPC stubs, type conversions
│       ├── cache.rs                    # CacheEvictionConfig, LruConfig, DurationConfig
│       ├── client_config.rs            # ClientConfig, ClientArgs (shared CLI args)
│       ├── config.rs                   # Config trait utilities
│       ├── download.rs                 # Download orchestration (smart-fallback, direct, server-only)
│       ├── models.rs                   # Status, ModelProvider, ModelStatus, ModelStatusResponse
│       ├── providers.rs                # ModelProviderTrait definition, re-exports
│       └── providers/
│           ├── gcs.rs                  # GcsProvider implementation
│           ├── gcs/                    # GCS manifest, cache layout, locking, download helpers
│           ├── huggingface.rs          # HuggingFaceProvider implementation
│           └── ngc.rs                  # NgcProvider implementation
│
├── workspace-tests/
│   ├── Cargo.toml
│   ├── tests/
│   │   └── integration_tests.rs        # Health, ping, download, fallback tests
│   └── benches/
│       └── performance.rs              # Criterion: DB ops, serialization benchmarks
│
├── helm/
│   ├── Chart.yaml                      # v0.2.2
│   ├── deploy.sh                       # Deploy script (microk8s/kubectl auto-detect)
│   ├── values.yaml                     # Default (1 replica, 10Gi PVC)
│   ├── values-development.yaml         # Dev (debug, 512Mi)
│   ├── values-production.yaml          # Prod (3 replicas, 2Gi, ingress)
│   ├── values-local-storage.yaml       # Test (no PVC, emptyDir)
│   └── templates/
│       ├── deployment.yaml
│       ├── service.yaml
│       ├── pvc.yaml
│       ├── ingress.yaml
│       └── serviceaccount.yaml
│
├── examples/
│   ├── p2p_transfer_k8s/               # GPU-to-GPU weight transfer example
│   │   ├── README.md
│   │   ├── model-download.yaml         # Model weights download job
│   │   ├── server/
│   │   │   ├── kubernetes_backend/     # CRD-based metadata (crd, rbac, server)
│   │   │   └── redis_backend/          # Redis-based metadata (redis, server)
│   │   └── client/
│   │       ├── vllm/
│   │       │   ├── Dockerfile          # vLLM + ModelExpress client image
│   │       │   ├── vllm-single-node.yaml  # TP-only (DeepSeek-V4-Pro)
│   │       │   └── vllm-multi-node.yaml   # TP+PP (DeepSeek-V4-Pro, 2 nodes)
│   │       └── sglang/
│   │           ├── Dockerfile          # SGLang + ModelExpress client image
│   │           └── sglang-single-node-p2p.yaml
│   ├── model_streamer_k8s/             # ModelStreamer startup examples
│   │   ├── README.md
│   │   └── client/
│   │       └── vllm/
│   │           ├── README.md
│   │           ├── vllm-single-node-streamer-azure.yaml
│   │           ├── vllm-single-node-streamer-s3.yaml
│   │           └── vllm-single-node-streamer-local.yaml
│   ├── crds.yaml                       # ModelMetadata + ModelCacheEntry CRDs (cluster-admin)
│   ├── dynamo_model_cache_k8s/         # Dynamo model-cache serving example
│   │   ├── README.md
│   │   └── agg.yaml
│   └── dynamo_p2p_transfer_k8s/        # Dynamo DGD with P2P weight transfer
│       ├── Dockerfile                   # dynamo vllm-runtime + MX client
│       ├── README.md
│       └── vllm/
│           ├── rbac-modelmetadata.yaml  # ServiceAccount + Role + RoleBinding
│           └── vllm-multi-node-aggregated.yaml  # DGD: MX server + Frontend + VllmWorker
│
├── docs/
│   ├── ARCHITECTURE.md                 # Architecture reference
│   ├── CLI.md                          # CLI tool documentation
│   ├── DEPLOYMENT.md                   # Deployment and configuration guide
│   └── metadata.md                     # Metadata storage and coordination protocol
│
├── .devcontainer/
│   ├── devcontainer.json               # VSCode config: rust-analyzer, port 8001
│   └── Dockerfile                      # Ubuntu 24.04 dev env
│
├── AGENTS.md                           # Always-on rules for all AI coding agents (Codex, Cursor, Copilot, Claude)
├── CLAUDE.md                           # Claude Code entry point: imports AGENTS.md
│
├── .agents/
│   └── skills/                         # Shared on-demand procedures (SKILL.md), read by Codex and Cursor
│       ├── add-cli-argument/
│       ├── add-grpc-service/
│       ├── bump-version/
│       └── dco/
│
├── .claude/
│   ├── settings.json                   # Claude Code permissions
│   └── skills/                         # Symlinks into .agents/skills/ so Claude Code sees the same skills
│
├── .github/
│   ├── copilot-instructions.md         # Copilot Chat pointer to AGENTS.md
│   ├── dco.yml                         # DCO enforcement
│   └── workflows/
│       ├── modelexpress-core-ci.yml     # CI pipeline
│       └── codeql.yml                  # Security scanning
│
├── .pre-commit-config.yaml             # Pre-commit hooks config
└── .coderabbit.yaml                    # CodeRabbit review config
```

## Workspace and Crate Structure

| Crate | Package Name | Type | Binary Targets |
|-------|-------------|------|----------------|
| `modelexpress_server` | `modelexpress-server` | lib + bin | `modelexpress-server`, `config_gen` |
| `modelexpress_client` | `modelexpress-client` | lib + bin | `modelexpress-cli`, `test_client`, `fallback_test` |
| `modelexpress_common` | `modelexpress-common` | lib | (none) |
| `workspace-tests` | `workspace-tests` | test + bench | (integration tests, criterion benchmarks) |

All cargo dependencies are declared in the root `Cargo.toml`. Sub-crates use workspace dependencies exclusively.

## gRPC Services

Six proto files define the server's gRPC services, all compiled via `tonic-build` in `modelexpress_common/build.rs`:

### health.proto - HealthService

| RPC | Request | Response | Purpose |
|-----|---------|----------|---------|
| `GetHealth` | `HealthRequest` | `HealthResponse` | Server version, status, uptime |

### api.proto - ApiService

| RPC | Request | Response | Purpose |
|-----|---------|----------|---------|
| `SendRequest` | `ApiRequest` | `ApiResponse` | Generic API (e.g., "ping" -> "pong") |

### model.proto - ModelService

| RPC | Request | Response | Purpose |
|-----|---------|----------|---------|
| `EnsureModelDownloaded` | `ModelDownloadRequest` | stream `ModelStatusUpdate` | Trigger download, stream progress |
| `StreamModelFiles` | `ModelFilesRequest` | stream `FileChunk` | Stream model file contents (1MB chunks) |
| `ListModelFiles` | `ModelFilesRequest` | `ModelFileList` | List files with sizes |
| `DeleteModel` | `DeleteModelRequest` | `DeleteModelResponse` | Remove a model's records from the registry (used by `model clear`) |

Key message types: `ModelProvider` (HuggingFace, NGC, GCS), `ModelStatus` (Downloading, Downloaded, Error), `ModelStatusUpdate`, `FileChunk`.

`EnsureModelDownloaded` verifies that a `DOWNLOADED` registry record still has its files on disk before honoring it as a cache hit. If the files are missing (for example after a `model clear` that only removed local storage), the stale record is deleted and the download is re-claimed so the model is actually re-fetched rather than returning a false success.

#### Pinned revisions

`ModelDownloadRequest` and `ModelFilesRequest` carry an optional `revision` (a branch, tag, or commit SHA); `ModelStatusUpdate` reports the `resolved_revision` the request landed on. Providers without a revision concept (NGC, GCS) reject a set `revision` and report none.

The server resolves the revision to an immutable commit **before** claiming the download lease, then pins every Hub file request to that commit, so a tag that moves mid-download cannot mix commits into one snapshot. A requested revision that does not exist fails the request; it never falls back to the default revision. An unpinned request still resolves the default revision to a commit SHA and reports it, so callers always learn the exact commit they got. If the Hub is unreachable while resolving an *unpinned* request, the server falls back to the snapshot already in its cache, which keeps cache hits working during a Hub outage.

`StreamModelFiles` and `ListModelFiles` resolve a requested revision against the local cache only — the model is already downloaded, so serving it must not depend on Hub reachability.

When a no-shared-storage client installs a snapshot over the file stream, it also writes the provider's local revision bookkeeping (`ModelProviderTrait::record_local_revision`; for Hugging Face, `refs/<requested-revision>`). Without it the files would be on disk but `huggingface_hub` could not resolve them by branch or tag under `HF_HUB_OFFLINE=1`.

#### Registry entry keys

Registry backends key every record, lease, and claim on one string. That key is an [`EntryKey`](../modelexpress_server/src/registry/entry_key.rs) encoding of `model_name` + resolved revision + weight mode. This is what keeps two revisions of one model from sharing a lease (and coalescing onto each other's download), and what keeps a metadata-only download from satisfying a later full-weight request with a weightless snapshot.

An unpinned, full-weight entry encodes to the bare model name, so records written before revisions existed keep parsing and providers without a revision concept keep the keys they have always used. Anything else encodes as `mx1:<revision>:<flags>:<model_name>`. The model name goes last because it is the only field with no character restrictions — a GCS object path accepts almost any byte, so a name containing the separator must not be able to impersonate the other fields and aim a delete at the wrong model.

Cache eviction parses the key back to delete the right snapshot. It keeps the files while another entry still references the same commit, treating a revisionless entry as covering every snapshot of its model. When the evicted entry is the last one for its model the whole repository directory goes, so a snapshot kept alive by an earlier shared-file decision cannot outlive the last record pointing at it. Hugging Face snapshot entries are symlinks into `blobs/`, so a per-revision delete also reclaims the blobs no surviving snapshot references — otherwise it would free almost no disk.

### p2p.proto - P2pService

| RPC | Request | Response | Purpose |
|-----|---------|----------|---------|
| `PublishMetadata` | `PublishMetadataRequest` | `PublishMetadataResponse` | Source publishes worker metadata (identity + tensors + backend metadata) |
| `ListSources` | `ListSourcesRequest` | `ListSourcesResponse` | Lightweight listing of available source workers (no tensor data) |
| `GetMetadata` | `GetMetadataRequest` | `GetMetadataResponse` | Fetch full tensor metadata for one specific worker (MB-scale, on demand) |
| `UpdateStatus` | `UpdateStatusRequest` | `UpdateStatusResponse` | Update per-worker lifecycle status (Initializing/Ready/Stale) |

Key message types: `SourceIdentity` (all fields affecting tensor layout compatibility and `mx_source_id`), `WorkerMetadata` (rank, runtime `accelerator`, oneof backend_metadata, tensors, status, P2P endpoint fields), `TensorDescriptor` (name, addr, size, device_id, dtype), `SourceInstanceRef` (lightweight worker reference for listing, including runtime `accelerator` for pre-fetch compatibility filtering).

Rust and Python compute `mx_source_id` from the same normalized, sorted-key JSON encoded as UTF-8, preserving non-ASCII identity strings.

### p2p.proto - WorkerService (P2P, opt-in)

| RPC | Request | Response | Purpose |
|-----|---------|----------|---------|
| `GetTensorManifest` | `GetTensorManifestRequest` | `GetTensorManifestResponse` | Fetch tensor descriptors directly from a source worker |
| `GetArtifactManifestHeader` | `GetArtifactManifestHeaderRequest` | `GetArtifactManifestHeaderResponse` | Fetch artifact identity, counts, file table, and worker endpoints |
| `GetArtifactManifestChunks` | `GetArtifactManifestChunksRequest` | `GetArtifactManifestChunksResponse` | Fetch artifact chunk metadata pages by `chunk_index` |
| `PrepareArtifactChunk` | `PrepareArtifactChunkRequest` | `PrepareArtifactChunkResponse` | Read one artifact range into source registered DRAM and return a NIXL descriptor lease |
| `ReleaseArtifactChunk` | `ReleaseArtifactChunkRequest` | `ReleaseArtifactChunkResponse` | Release a prepared artifact chunk lease |

Per-worker gRPC service started when `MX_P2P_METADATA=1`, or unconditionally when using a decentralized metadata backend (the backend's client sets `REQUIRES_P2P_METADATA = True` and the env var is ignored). Targets call this instead of fetching tensor descriptors or artifact manifest metadata from the central server. `GetTensorManifestResponse` carries the source worker's runtime `accelerator` value so decentralized targets can apply the same compatibility filter as central metadata mode. RL generator lease behavior is documented in the [refit architecture](rl/ARCHITECTURE.md#refit-service-and-version-model). Ordinary inference loading continues to use `GetTensorManifest` without the RL mutation fence. Artifact byte transfer still uses NIXL; `PrepareArtifactChunk` only exposes a source-side registered DRAM range for one sealed artifact chunk. Tensor RPCs require `mx_source_id` and validate the selected runtime `worker_id` when present to catch stale discovery records whose endpoint has been reused by a new process.

See [`metadata.md`](metadata.md) for the full metadata architecture including storage schemas and coordination protocol.

### RL refit service

The optional Redis-backed RefitService, immutable WeightVersion model, shard publication protocol, and generator/trainer control clients are described in the [RL refit architecture](rl/ARCHITECTURE.md#refit-service-and-version-model).

## Rust Server

### Startup Flow

1. Parse CLI args (`ServerArgs` via clap)
2. Load config (`ServerConfig::load()`) - CLI > env vars > config file > defaults
3. Initialize structured logging (tracing-subscriber)
4. Build the Prometheus registry and register every metric family, then start the
   `/metrics` listener on its own port. Deliberately before anything that can
   fail, so a scrape proves the exporter came up even on a server that never
   reaches a healthy state. A bind failure logs and continues rather than
   aborting startup.
5. Connect to the model registry backend (`MX_METADATA_BACKEND` — same selector as P2P). Fails fast on connect error.
6. Initialize the process-wide `ModelDownloadTracker` with the registry, seed the `MODEL_TRACKER` OnceLock
7. Start `CacheEvictionService` background task (reads the same registry)
8. Connect to the P2P metadata backend (`MX_METADATA_BACKEND`, Redis or Kubernetes CRD)
9. Start reaper background task for stale source detection and GC
10. Start the registry-statistics refresh task (`MX_REGISTRY_STATS_INTERVAL_SECS`,
    default 60). Independent of `CacheEvictionService`: that one ticks hourly and
    is skipped entirely when eviction is disabled, which would leave the gauges
    permanently absent.
11. Register the configured gRPC services with tonic (max message size: 100MB),
    wrapped in the per-RPC metrics layer
12. Listen on configured address (default `0.0.0.0:8001`)
13. Graceful shutdown on CTRL+C: signals the cache eviction service, the reaper
    and the statistics refresh, then joins them. The `/metrics` listener stops
    **last**, after the gRPC server has drained, so the drain window stays
    scrapeable.

### ServerConfig

```yaml
server:
  host: "0.0.0.0"        # MODEL_EXPRESS_SERVER_HOST
  port: 8001              # MODEL_EXPRESS_SERVER_PORT
cache:
  directory: "./cache"    # MODEL_EXPRESS_CACHE_DIRECTORY
  max_size_bytes: null
  eviction:
    enabled: true         # MODEL_EXPRESS_CACHE_EVICTION_ENABLED
    policy:
      type: lru
      unused_threshold: "7d"
      max_models: null
      min_free_space_bytes: null
    check_interval: "1h"
logging:
  level: info             # MODEL_EXPRESS_LOG_LEVEL
  format: pretty          # MODEL_EXPRESS_LOG_FORMAT
  file: null
  structured: false
```

Distributed backend selection lives outside the YAML, in env vars: `MX_METADATA_BACKEND` (drives both P2P and registry) plus the corresponding connection vars (`REDIS_URL` or `POD_NAMESPACE`). See [`DEPLOYMENT.md`](DEPLOYMENT.md#distributed-backend-selection).

### ModelRegistryBackend (Redis and Kubernetes CRD)

**Redis backend**: single Redis Hash per cached model at `mx:model:{provider}:{name}` with fields `provider`, `status`, `created_at` (RFC3339), `last_used_at` (RFC3339), and optional `message`. No secondary indexes — LRU ordering and status counts are computed on demand by `SCAN` + pipelined `HGETALL`/`HGET`. Claim, retry, and `set_status` updates use Lua scripts so concurrent readers see either the pre-update record or the complete post-update record, never a partially-written hash.

**Kubernetes CRD backend**: one `ModelCacheEntry` CR per cached model in the server's namespace. `spec.modelName` + `spec.provider` are immutable; `status.{phase,createdAt,lastUsedAt,message}` are patched via the status subresource. Atomicity on the claim path comes from etcd's name-uniqueness on `create` (409 Conflict on the loser). CR names are `mx-cache-` followed by `sanitize_registry_name("{provider}/{model_name}")`, which lowercases, maps `/` to `--`, and appends a sha256 suffix over the original name. The P2P `ModelMetadata` CRs use their own `sanitize_model_name`; the two are separate implementations, not a shared helper.

Key operations on the async `RegistryBackend` trait:

| Method | Redis implementation |
|--------|----------------------|
| `get_status(name)` | `HGET mx:model:{provider}:{name} status` (falls back to the legacy name-only key so pre-0.5.0 records still resolve) |
| `set_status(name, provider, status, msg)` | Lua `EVAL` updates status/provider/last_used_at/message atomically and `HSETNX`s `created_at` to preserve the first-write timestamp |
| `try_claim_for_download(name, provider)` | `HSETNX status DOWNLOADING`; winner populates remaining fields without contention |
| `touch_model(name)` | `HSET last_used_at {now}` (gated on `EXISTS` so touch is update-only, never create) |
| `delete_model(name)` | `DEL` |
| `get_models_by_last_used(limit)` | `SCAN mx:model:*` + pipelined `HGETALL` + Rust-side sort |
| `get_status_counts()` | `SCAN mx:model:*` + pipelined `HGET status` + Rust-side tally |

Concurrency: Redis backends use cloned async `ConnectionManager`s behind an `RwLock` for lazy initialization and reconnect caching.

### CacheEvictionService

Runs in a background tokio task on a configurable interval (default 1 hour). LRU eviction policy:

1. Time-based: evict models with `last_used_at` older than `unused_threshold` (default 7 days)
2. Count-based: if total > `max_models`, evict oldest excess
3. Only DOWNLOADED models are eligible for eviction

### P2P Metadata Backends

Two families of backends exist: **server-coordinated** (the server owns the metadata store) and **decentralized** (no central server in the loop).

Server-coordinated backends live in the Rust server and are selected via `MX_METADATA_BACKEND`:

- **Redis** (`redis`): Source index hashes (`mx:source:{source_id}`) with an `__attributes__` field storing `SourceIdentity` and `{worker_id}` fields as presence markers. Worker data stored in separate hashes (`mx:source:{source_id}:{worker_id}`). Stale detection and cleanup handled by the server-side reaper.
- **Kubernetes** (`kubernetes`/`k8s`/`crd`): `ModelMetadata` CRDs (one per worker) with `ConfigMap`s for tensor descriptors. Owner references for automatic garbage collection. Standard Kubernetes `status.conditions` (`Ready`) and `status.observedGeneration` are maintained so that `kubectl wait --for=condition=Ready` works. Stale detection handled by the server-side reaper.

The decentralized backend lives in the Python client and is selected via `MX_METADATA_BACKEND`:

- **K8s-service** (`k8s-service`/`service`): each source pool sits behind a Kubernetes Service (one per tensor-parallel rank, label selector pinned to `mx.rank=R`). Clients open a direct gRPC channel to the Service DNS and call `GetTensorManifest`; kube-proxy load-balances across ready backends. No central server is involved. `mx_source_id` is computed client-side via the same canonical JSON + SHA256 scheme and validated on the response. See [`../examples/k8s_service_sources/`](../examples/k8s_service_sources/) for the deployment shape.

Each worker publishes independently. The `mx_source_id` is a 16-char hex key computed from `SHA256(canonical_json(SourceIdentity))` where `SourceIdentity` includes a `revision` field for content-addressed identity (HuggingFace commit SHA, S3 object version, or a deployer-provided string). When `revision` names immutable content, two sources with identical `mx_source_id` are expected to serve bit-identical weight bytes; the ID itself validates declared identity rather than hashing tensor contents, so the guarantee is only as strong as the revision pin and the local cache being intact. Large u64 values (GPU addresses) are serialized as strings to avoid JSON precision loss.

The Rust and Python implementations of `compute_mx_source_id` are locked together via cross-checked pinned-hash unit tests (`source_identity.rs::test_python_cross_check_*` and `test_source_id.py::test_pinned_hash_*`). Either side drifting on canonical JSON encoding or hashing breaks both test sets together.

See [`metadata.md`](metadata.md) for the full storage layout and schemas.

### k8s-service Metadata Backend

The decentralized `k8s-service` backend lives in the Python client as `MxK8sServiceClient` (duck-typed to `MxClientBase`). Clients open a direct gRPC channel to a Kubernetes Service DNS name and call `GetTensorManifest`; kube-proxy load-balances across ready backends; `mx_source_id` is computed client-side (Python `compute_mx_source_id` matches the Rust implementation via pinned cross-check tests) and validated on every response. This backend currently serves tensor manifests only; file-backed artifact discovery requires a central-coordinator backend (`redis` or `kubernetes`) until `k8s-service` grows an artifact-source discovery path.

**Pattern encoding:** `MX_K8S_SERVICE_PATTERN` supports two shapes:

- Explicit `:port` in the pattern (e.g. `mx-sources-rank-{rank}:6555`) is used verbatim after `{rank}` substitution. Rank encoded in the hostname; one Service per rank with a rank-specific label selector.
- No port in the pattern (e.g. `mx-sources`, the default) triggers auto-append of `:{MX_WORKER_GRPC_PORT + rank}`. Rank encoded in the port; one Service with N named ports, each targeting the matching in-pod `WorkerGrpcServer`.

**Pool constraint:** every ready pod behind a given Service must serve the same `mx_source_id`. Transient revision skew during rolling updates is handled by client-side retry on `FAILED_PRECONDITION` over fresh gRPC channels, up to `MX_K8S_SOURCE_RETRIES`. Workloads that need per-worker addressability, including mixed-version fleets, must use the central-coordinator backends instead; the k8s-service backend's Service-routing model has no way to express "this specific worker."

See [`K8S_SERVICE_BACKEND.md`](K8S_SERVICE_BACKEND.md) for design rationale, limitations, and backend-selection guidance.

### ModelDownloadTracker

Global singleton (`LazyLock<ModelDownloadTracker>`) that coordinates concurrent downloads. Uses `try_claim_for_download()` for race-free model claiming and tokio channels for streaming status updates to multiple waiting clients.

## Rust Client

### Client Public API

The `Client` struct in `modelexpress_client/src/lib.rs` wraps gRPC connections:

| Method | Purpose |
|--------|---------|
| `new(config)` | Create a client with the given configuration |
| `new_with_cache(config, cache_config)` | Create a client with an explicit cache configuration |
| `get_cache_config()` | Get the client's cache configuration, if any |
| `set_cache_config(cache_config)` | Set the client's cache configuration |
| `list_cached_models()` | List locally cached models as `CacheStats` |
| `clear_cached_model(name, provider)` | Remove a model's local files for a given provider |
| `clear_all_cached_models()` | Clear the entire local cache |
| `delete_model_on_server(name, provider)` | Delete the model's record from the server-side registry, so a cleared model leaves no stale `DOWNLOADED` record |
| `get_model_path(name, provider)` | Resolve the local cache path for a model through its provider |
| `health_check()` | Call HealthService and return the server `Status` |
| `send_request(action, payload)` | Send a generic ApiService request and deserialize the response |
| `request_model_on_server(name, provider)` | Request a download on the server at the provider's default revision |
| `request_model_on_server_revision(name, provider, revision)` | Same, pinned to a branch, tag, or commit SHA; returns the resolved revision |
| `request_model(name, provider)` | Request a model using the server as source of truth, streaming files locally when shared storage is disabled |
| `request_model_revision(name, provider, revision)` | Same, pinned to a revision; returns the snapshot path and the revision it resolved to |
| `request_model_with_smart_fallback(name, provider, ...)` | Request via the server, falling back to a direct provider download when the connection cannot be established |
| `request_model_with_smart_fallback_revision(name, provider, revision, ...)` | Same, with the direct-download fallback honouring the same pinned revision |

The `_revision` variants all take an optional branch, tag, or commit SHA, but they do not share a return type. `request_model_revision` and `request_model_with_smart_fallback_revision` return a `ModelDownloadResult { path, resolved_revision }`. `request_model_on_server_revision` only asks the server to fetch the model, so it returns the resolved revision on its own as an `Option<String>`. The non-`_revision` methods delegate to them with no revision pinned and discard the result.

### Download Strategies

| Strategy | Behavior |
|----------|----------|
| `SmartFallback` | Try server first, fall back to direct download on failure |
| `ServerOnly` | Only download through the server |
| `DirectOnly` | Only download directly from provider |

### CLI (modelexpress-cli)

The `Cli` struct in `args.rs` embeds `ClientArgs` via `#[command(flatten)]`. Commands:

| Command | Purpose |
|---------|---------|
| `health` | Check server health and status |
| `model download <name>` | Download a model with various strategies (automatically cached). `--revision <branch\|tag\|sha>` pins a revision; the resolved commit SHA is reported back |
| `model init` | Initialize model storage configuration |
| `model list` | List downloaded models |
| `model status` | Show model storage status and usage |
| `model clear <name>` | Clear a specific model from storage |
| `model clear-all` | Clear all models from storage |
| `model validate [name]` | Validate model integrity |
| `model stats` | Show model storage statistics |
| `api send <action>` | Send a custom API request |

Output formats: `--format human` (default), `--format json`, `--format json-pretty`.

## Common Library

### Modules

| Module | Purpose |
|--------|---------|
| `cache` | `CacheEvictionConfig`, `LruConfig`, `DurationConfig` (used by both server and client configs) |
| `client_config` | `ClientConfig` + `ClientArgs` shared struct for CLI argument handling |
| `config` | Config trait utilities |
| `download` | Download orchestration with strategy pattern |
| `models` | `Status`, `ModelProvider`, `ModelStatus`, `ModelStatusResponse` |
| `providers` | `ModelProviderTrait` + `HuggingFaceProvider` + `NgcProvider` + `GcsProvider` |
| `grpc` | Generated tonic stubs for all 4 services |
| `constants` | `DEFAULT_GRPC_PORT` (8001), `DEFAULT_TIMEOUT_SECS` (30), `DEFAULT_TRANSFER_CHUNK_SIZE` (32KB) |

### ModelProviderTrait

```rust
#[async_trait]
pub trait ModelProviderTrait: Send + Sync {
    async fn download_model(&self, name: &str, cache_path: Option<PathBuf>, ignore_weights: bool) -> Result<PathBuf>;
    async fn delete_model(&self, name: &str, cache_dir: PathBuf) -> Result<()>;
    async fn get_model_path(&self, name: &str, cache_dir: PathBuf) -> Result<PathBuf>;

    // Revision-aware variants. The defaults reject a pinned revision and delegate to the
    // methods above, so only providers that expose revisions need to implement them.
    async fn download_model_revision(&self, name: &str, cache_path: Option<PathBuf>, ignore_weights: bool, revision: Option<&str>) -> Result<ModelDownloadOutcome>;
    async fn resolve_revision(&self, name: &str, cache_dir: Option<PathBuf>, revision: Option<&str>) -> Result<Option<String>>;
    async fn delete_model_revision(&self, name: &str, cache_dir: PathBuf, revision: Option<&str>) -> Result<()>;
    async fn get_model_path_revision(&self, name: &str, cache_dir: PathBuf, revision: Option<&str>) -> Result<PathBuf>;
    async fn record_local_revision(&self, name: &str, cache_dir: &Path, requested: Option<&str>, commit: &str) -> Result<()>;
    fn supports_revisions(&self) -> bool;

    fn provider_name(&self) -> &'static str;
    fn is_ignored(filename: &str) -> bool;
    fn is_image(path: &Path) -> bool;
    fn is_weight_file(filename: &str) -> bool;
}
```

`ModelDownloadOutcome` carries the snapshot `path` plus the `resolved_revision` the request landed on (`None` for providers with no revision concept).

Three implementations:
- `HuggingFaceProvider` - uses the `hf-hub` crate with high-CPU download mode. Implements the revision-aware methods: it resolves a branch, tag, or SHA through `/api/models/<repo>/revision/<rev>`, downloads every file from the resolved commit, and writes `refs/<requested-revision>` so the snapshot stays resolvable by the name the caller used. `delete_model_revision` removes a single snapshot, and drops the whole repository directory once its last snapshot is gone.
- `NgcProvider` - downloads from NVIDIA NGC via the files-manifest endpoint `…/v2/org/{org}[/team/{team}]/{type}/{name}/{version}/files` (no `versions/` segment), which returns self-authenticating presigned URLs paired with relative paths for every file, for both V1 and V2 storage and both org and team scopes (so nested paths and downloads need no per-file URL construction or Authorization forwarding). Falls back to `checksums.blake3` manifest enumeration against the versioned `…/versions/{version}/files` endpoint (Bearer-authenticated `/files/{path}` downloads) when the listing returns 400/401, as some UAM-gated orgs (e.g. the `nim` catalog) do. Resolves the NGC API key from `NGC_API_KEY`, `NGC_CLI_API_KEY`, or `~/.ngc/config`.
- `GcsProvider` - downloads objects under a full `gs://<bucket>/<object-prefix>` URL using Google Application Default Credentials. It writes a `.mx/manifest.json` cache manifest, verifies downloaded files with GCS CRC32C checksums, skips dotfiles, README, and images, and stores models under `<cache>/gcs/<bucket>/<object-prefix>`. See [`GCS_PROVIDER.md`](GCS_PROVIDER.md) for the detailed design.

### ClientConfig / ClientArgs

`ClientArgs` is the single source of truth for shared CLI arguments. `Cli` embeds it via `#[command(flatten)]`.

Loading precedence: CLI args > environment variables > config file > defaults.

## Python Client

### Modules

| Module | Purpose |
|--------|---------|
| `__init__.py` | Package init, exports `register_modelexpress_loaders()` for callers to register the `modelexpress` and `mx` loaders with vLLM |
| `client.py` | `MxClient` - gRPC client wrapping `PublishMetadata`, `ListSources`, `GetMetadata`, and `UpdateStatus` RPCs |
| `accelerators/` | `AcceleratorBackend` boundary for accelerator-specific torch device control and fast-path capability gates, split into `base.py` (protocol), `cuda.py` (`CudaAcceleratorBackend`), and `xpu.py` (`XpuAcceleratorBackend`). CUDA and XPU are implemented backends; XPU keeps CUDA-only fast paths (pool registration, VMM arena, GDS) disabled and falls back to generic per-tensor NIXL registration. Further backends can be added behind the same interface |
| `nixl_transfer.py` | `NixlTransferManager` - NIXL agent lifecycle, tensor registration, RDMA transfers |
| `gds_transfer.py` | GPUDirect Storage availability check and transfer utilities |
| `gds_loader.py` | `MxGdsLoader` - GDS-based model loader (direct file-to-GPU) |
| `adapter.py` | `EngineAdapter` lifecycle hooks and strategy retry errors |
| `plugins.py` | Discovery, validation, and selection of external loading strategy classes |
| `vllm_loader.py` | Compatibility shim for `modelexpress.engines.vllm.loader` |
| `metadata/` | Metadata publishing, source identity, heartbeat, worker manifest serving, metadata client selection, and engine-agnostic cache-artifact transfer |
| `load_strategy/` | Engine-neutral loading strategy chain: `RdmaStrategy`, `ServerCacheStrategy` (weights streamed from MX Server), `InstantTensorStrategy` (fast local safetensors), `ModelStreamerStrategy` (S3/GCS/Azure/local), `GdsStrategy`, `DefaultStrategy` |
| `model_client.py` | `ModelCacheClient` - `ModelService` RPCs plus stream validation for server-cached models |
| `model_snapshot.py` | Hugging Face cache layout: path validation, atomic snapshot publication, revision refs, metadata inventories |
| `model_prefetch.py` | Pre-engine metadata prefetch and repo-id resolution for server-backed loading |
| `engines/vllm/` | `VllmAdapter` and `MxModelLoader` map initial-load strategy hooks to vLLM loader APIs |
| `engines/sglang/` | `SglangAdapter` and `MxModelLoader` map strategy hooks to SGLang's `remote_instance` backend; `draft_weights.py` owns model-specific role rules and `checkpoint_selection.py` filters complete safetensors shards before ModelStreamer reads them |
| `tensor_utils.py` | Tensor collection, checksums, storage views, `capture_tensor_attrs` |
| `rank_utils.py` | `get_global_rank`, `get_worker_rank` |
| `vllm_worker.py` | `ModelExpressWorker` - compatibility worker class for older manual-registration workflows |
| `types.py` | `TensorDescriptor`, `WorkerMetadata`, `GetMetadataResponse` dataclasses |
| `p2p_pb2.py` / `p2p_pb2_grpc.py` | Generated protobuf/gRPC stubs |

RL trainers and generators use the optional `modelexpress_rl` namespace from the same Python distribution. See the [RL package guide](rl/README.md) and [RL refit architecture](rl/ARCHITECTURE.md).

### MxClient

gRPC client wrapping the P2P service stubs:

| Method | Purpose |
|--------|---------|
| `publish_metadata(identity, worker, worker_id)` | Publish worker metadata; returns `mx_source_id` |
| `list_sources(identity, status_filter)` | List available source workers (lightweight, no tensor data) |
| `get_metadata(mx_source_id, worker_id)` | Fetch full tensor metadata for one worker (on demand) |
| `update_status(mx_source_id, worker_id, worker_rank, status, source_load)` | Update worker lifecycle status (e.g., `READY`) and its published `source_load` |
| `close()` | Close the underlying gRPC channel |

### NixlTransferManager

Manages a NIXL agent and RDMA transfers for a single GPU worker:

| Method | Purpose |
|--------|---------|
| `__init__(agent_name, device_id, listen_port, accelerator_backend)` | Create NIXL agent with UCX backend; `listen_port` enables P2P listen thread; `accelerator_backend` owns torch device operations and accelerator capability gates |
| `register_tensors(tensors)` | Register GPU tensors for RDMA, return serialized metadata. With `MX_POOL_REG=1` on a backend that supports pool registration, registers each unique cudaMalloc allocation backing the tensors instead of registering each tensor individually |
| `register_arena(arena, tensors)` | Register the used VMM arena range once through dmabuf when the active accelerator backend supports the VMM arena fast path, then publish every tensor descriptor against that single MR. Falls back to per-tensor registration when a tensor lies outside the arena range, or when the arena spans several `cuMemCreate` handles (a single MR cannot be addressed by cuda_ipc then; override with `MX_ARENA_SINGLE_MR=1`) |
| `register_additional_tensors(tensors)` | Append tensors to an already-registered catalog instead of replacing it (`register_tensors` / `register_arena` replace the name -> tensor catalog). Tensors fully covered by the main or a prior appended registration reuse its range; only uncovered ranges are registered. The call refreshes agent metadata, skips names already registered at the same address, and rejects a name registered at a different address. Used by the speculative draft pass on the main load's agent |
| `deregister_tensors(names)` | Release tensors added by `register_additional_tensors` (catalog entries and, once unused, their registration handle). Base-catalog names are ignored; `shutdown()` releases those |
| `fetch_remote_and_wait(agent_name, ip, port)` | P2P: fetch remote NIXL metadata via listen thread (polls until loaded) |
| `receive_from_source(source_metadata, source_tensors, ..., remote_agent_name)` | Execute RDMA read transfer; `remote_agent_name` skips `add_remote_agent` (P2P) |
| `shutdown()` | Clean up NIXL agent and resources |

**Optional NIC pinning.** `MX_RDMA_NIC_PIN=auto` probes PCIe topology at agent init and pins `UCX_NET_DEVICES` to a NUMA-local IB NIC per worker. Workaround for [openucx/ucx#11259](https://github.com/openucx/ucx/issues/11259); see [`docs/DEPLOYMENT.md`](DEPLOYMENT.md) for details.

### vLLM Loader

**MxModelLoader** (extends `BaseModelLoader`, registered as `--load-format modelexpress`; `mx` alias):

Thin orchestration layer that delegates to `LoadStrategyChain.run()`. Builds a `LoadContext` from vLLM config, initializes the model, runs the strategy chain, and updates global registries.

After successful RDMA receive and target-local finalization, the vLLM adapter
synchronizes the accelerator and clears unused allocator cache before subsequent
initialization. Live model tensors remain allocated.

**MTP two-pass load.** vLLM loads the main model and then the speculative draft on the same worker. The main pass owns one NIXL agent, metadata listener, and source publication. A draft from the same checkpoint reuses that agent and appends its tensors under the `mx_draft::` namespace; it does not bind a second listener or replace the main tensor registry. A draft with a different `SourceIdentity` remains storage-only. The receiving main pass ignores draft descriptors, while the receiving draft pass requires complete coverage of its own descriptors. The main publication waits for the draft pass (or the engine health endpoint) so a peer does not discover a half-populated source. Failed draft registration rolls back only draft tensors, not the live main agent.

For a cold MTP source, ModelStreamer asks the initialized vLLM model for `is_unused_checkpoint_weight`, when available, and selects **complete shard files** before reading them. A RunAI loader that accepts the predicate handles selection itself; otherwise MX uses vLLM's local safetensors filter or a generic checkpoint-index selector. The model-owned hook always takes precedence. For older engines without the hook, MX has a bounded draft-only fallback for the Qwen3.5, Qwen3-Next, DeepSeek MTP, GLM4-MoE MTP and GLM4-MoE-Lite MTP loader classes. Qwen's legacy rule retains `mtp.*`, embedding and head weights; the DeepSeek/GLM paths reuse the installed loader's layer-selection helper with its runtime config. Unsupported classes, missing helpers/configuration or unusable optional index metadata retain full-file loading. Main models without a hook also retain full-file loading. Mixed shards may be read twice. This is not tensor-range filtering; vLLM still performs tensor mapping and `load_weights`. A draft that lacks a compatible main agent or READY peer can use the usual storage strategies. For P2P, Step3.5 keeps its per-layer `shared_head.head` tensors: unlike the base MTP proposer, its proposer does not replace them with the target head.

With `MX_VMM_ARENA=1`, the draft loads outside the main VMM arena and registers its tensors individually. Re-entering or replacing the arena would invalidate the already-published main weights and peer RDMA keys.

```mermaid
flowchart LR
    I[Checkpoint index] --> S[Role-aware shard selection]
    S -->|main files| M[vLLM main load]
    S -->|draft files| D[vLLM draft load]
    M --> N[One NIXL agent and source]
    D -->|append mx_draft tensors| N
    N -->|main descriptors| PM[Peer main load]
    N -->|draft descriptors| PD[Peer draft load]
```

**Sleep/wake lifecycle hooks.** `MxModelLoader` implements vLLM's optional `BaseModelLoader.on_sleep(level)` / `on_wake_up(tags)` hooks so a sleeping replica stops advertising itself as a P2P source before `sleep()` unmaps its GPU memory.
- `on_sleep` calls `unpublish_metadata` (gates discovery) and `nixl_manager.shutdown()` (stops receiving new requests from peers).
- `on_wake_up` recreates the NIXL agent and calls `publish_metadata` to mark the replica ready for new incoming P2P connections. It does so only after a level-1 sleep: level 2 discards the weights, so this has to happen on `reload_weights()` instead. Waking the weights clears the recorded sleep level after either level.
- After a level-2 sleep, `reload_weights()` refreshes weight content in place through `reload_model()` (see below), whose strategy chain registers the tensors and republishes the replica, as on any successful load.
- `reload_model()` without a preceding sleep first stops serving the same way `on_sleep` does, so peers never read weights while they are overwritten.
- With `MX_VMM_ARENA=1`, a reload neither replaces nor enters the arena: the weights are written into their existing arena-backed storage, so only transient allocations happen. The reload's context reuses the existing arena so registration again produces a single MR.
- An RDMA reload skips target-side post-processing and keeps state derived at the original load, so RDMA is only eligible when the reload's `SourceIdentity` matches the identity of the weights the model holds (snapshotted when the reload starts, so it follows refits). Otherwise a streaming strategy reprocesses the weights. The check is as strong as the identity's `revision`: an empty or branch revision only proves the same model name.
- A strategy that fails mid-reload, or an RDMA candidate that does, is not followed by `reinit_for_retry`: the engine keeps serving the model it passed in, so the next strategy or candidate overwrites that same model in place.
### SGLang Refit Integration

SGLang obtains its rank-local S3 refit client through the public factory:

```python
from modelexpress_rl.inference.engines.sglang import get_modelexpress_generator

generator = get_modelexpress_generator(model_runner)
```

The factory reads `model_runner.server_args.modelexpress_config` as JSON or a
dictionary, resolves the local seed checkpoint when no seed path is supplied,
and reuses the client cached on `model_runner.modelexpress_generator`. Connection
settings use `object_storage_endpoint_url` and `object_storage_region_name`,
matching the vLLM integration. SGLang owns update coordination and closes the
cached client during scheduler shutdown.

### RL generator installation

The vLLM RL update lifecycle, `stage_weight()` behavior, checkpoint caching, and failure recovery are covered in the [RL refit architecture](rl/ARCHITECTURE.md#python-engine-integration).

### SGLang Loader

**MxModelLoader** is instantiated by SGLang's `remote_instance` loader when
`--remote-instance-weight-loader-backend modelexpress` is used. SGLang
initializes the model, then delegates to this loader. The loader builds a
`LoadContext` from SGLang config and dispatches by `modelexpress-config`
transport. For `transport=nixl`, it runs `LoadStrategyChain.run()` and updates
global registries. For `transport=transfer_engine`, it uses SGLang's initialized
TransferEngine and ModelExpress metadata directly: first replica loads natively,
registers TransferEngine memory, and publishes its session; later replicas
discover that source and pull with `batch_transfer_sync_read`.
`SglangAdapter` owns SGLang-specific rank/device mapping, native fallback
loading, quantized-weight post-processing, and tensor discovery, including the
storage-view naming used for non-contiguous SGLang parameters.
The SGLang side does not expose separate source and target modes; transport
selection and source discovery remain inside the ModelExpress package.

**Speculative draft pass.** SGLang's EAGLE worker builds a second ModelRunner
for the draft on the same GPU. Its `ModelConfig` has `is_draft_model=True` and,
unless `speculative_draft_load_format` names another format, it loads through the
same `remote_instance` backend. The generic MX draft path recognizes that
second pass, reuses the main load's NIXL agent instead of binding the same
metadata port again, and gives draft tensors a separate manifest namespace.
Cold-load shard selection is separate from P2P compatibility. MX recognizes
exact SGLang main and draft model classes and mirrors their unconditional
checkpoint-name exclusions in `DraftSelection` rules. For example, Qwen3.5
text-only mains skip vision and MTP tensors, whereas conditional-generation
mains retain vision tensors; their draft reads MTP tensors and its required
embedding. Qwen4-Exp conditional-generation mains also honor SGLang's
`language_model_only` flag: a text-only instance omits vision-only shards,
while a shard mixed with text weights remains selected. DeepSeek/GLM NextN
rules use the configured base-layer count to
separate the appended prediction layer. DeepSeek V4's DSpark draft instead
reads `mtp.<stage>.*` weights and pairs with the DeepSeek V4 main; it does
not use the single-layer NextN rule. Compatible class pairs are explicit,
and draft manifests are scoped by the class, engine version, checkpoint
identity, and role-specific layout parameters. Rules are shared only where
SGLang uses the same loading branch. These are pre-read file-selection rules,
not a replacement for SGLang's tensor mapping, quantization, conditional
parameter checks, or `load_weights`. When SGLang exposes a stable model-owned
selection hook, MX can consume it instead of maintaining these temporary
mirrors. The checkpoint's `model.safetensors.index.json` maps tensor names to shard files;
it does not by itself define which tensors belong to every architecture's draft.

- A recognized draft may use P2P only when it shares the main `SourceIdentity`,
  their exact model rules are compatible, pipeline parallelism is one, and
  the source identity includes an explicit revision and SGLang package version.
  A hash of these inputs and the model URI scopes its draft tensor names. The
  receiver accepts only that namespace and verifies complete descriptor
  coverage. Unknown or incompatible drafts fall back to SGLang's storage load.
- After the main and draft loads, only draft-owned tensors are appended to the
  main NIXL registration and metadata publication; target-shared embedding and
  head storage is excluded using SGLang's `get_embed_and_head`, independent of
  parameter names. The source's publication gate does not open for a recognized
  draft until that extension succeeds. If a health URL is configured,
  both draft publication and engine health must be ready.
- On a cold source using ModelStreamer, a recognized model rule selects
  **files** separately for the main and draft passes. The generic tensor-name
  to index to shard algorithm is the same for every rule. MX first lets SGLang's
  `RunaiModelStreamerLoader._prepare_weights` resolve the checkpoint and its
  normal file list. An override then reads `model.safetensors.index.json` from
  the root of that resolved local directory or object-store URI, maps the role-specific
  tensor names to shard filenames, and intersects them with SGLang's file
  list **before** SGLang creates its streamer iterator. SGLang still owns
  `Source` handling, `draft_model_idx`, name prefixing, tensor mapping,
  quantization, and `load_weights`. Only the primary model URI is filtered;
  unrelated `secondary_weights` sources retain SGLang's original file list.
  Missing/malformed index data, a failing model rule, a role with
  no matches, or a selected filename absent from SGLang's list leaves the
  original full-file list unchanged. Models without a rule use the unmodified
  SGLang loader. A mixed shard containing both main and draft
  tensors is selected by both passes, so file-level selection does **not**
  guarantee disjoint object-store bytes or eliminate every duplicate read.
  SGLang may append an unindexed `mtp.safetensors` for affected GLM layouts
  after `_prepare_weights`; that compatibility path is not covered by index
  selection and can still read the extra file in both passes.
  Extra-layer adapters are disabled when the required layer metadata is absent,
  describes more than one draft layer, or uses the ambiguous one-layer legacy
  layout; those checkpoints retain the full-file storage fallback.
- Cold loading uses shard selection without a persistent local draft staging
  directory. For object storage, the index reader uses a temporary directory for
  `model.safetensors.index.json` only and removes it after parsing; the
  selected weight shards still flow through SGLang's ModelStreamer. This does
  not change the warm target's role-scoped GPU P2P path.
- On `transport=transfer_engine`, the draft loads natively and never publishes.
  A second TransferEngine source would advertise the head under the target's
  identity and replace the target's heartbeat, and its same-named tensors would
  match the target's manifest. `MX_LOAD_STRATEGY_CHAIN=RL` rejects a draft.

```mermaid
flowchart LR
    I[Checkpoint index<br/>tensor to shard] --> S[One generic shard selector]
    R[SGLang resolves URI<br/>and available shards] --> S
    W[Recognized SGLang model rule<br/>mirrored in MX] --> S
    S -->|main shard subset| M[SGLang ModelStreamer<br/>main load_weights]
    S -->|draft shard subset| D[SGLang ModelStreamer<br/>draft load_weights]
    S -. unsupported or incomplete .-> F[SGLang full-shard fallback]
    M --> P[One NIXL agent<br/>one READY source]
    D -->|append draft-namespaced tensors| P
    P -->|one manifest, two role-scoped tensor sets| T[New replica<br/>main and draft P2P]
```

**LoadStrategyChain** (`load_strategy/`):

Auto-detects the best loading strategy with a prioritized chain. Each strategy is a subclass of `LoadStrategy` (ABC) with `is_available(ctx)` and `load(result, ctx)` methods. Engine-specific work is delegated to `ctx.adapter`; `LoadResult` carries the value returned to the engine plus the model used for tensor discovery and publication. The chain filters to eligible strategies and runs them in order until one succeeds:

| Priority | Strategy | `is_available()` | Behavior |
|---|---|---|---|
| p0 | `RdmaStrategy` | NIXL available | `ListSources(READY)`, filter by `worker_rank` and runtime `accelerator`, order the survivors via the configured `SourceSelector` (`MX_P2P_SOURCE_SELECTOR`: `random` default, `rendezvous_hash`, `load_aware`, or `topology_aware`), then try candidates (max 3). Filtering before the retry slice prevents incompatible sources from exhausting the retry budget; a post-`GetMetadata` accelerator check remains as defense-in-depth. Before preparing target tensors, P2P sources must serve a manifest for the selected runtime `worker_id`; generation mismatches and transfer failures retry the next candidate, reinitializing the target first when it may have been mutated. |
| p1 | `ServerCacheStrategy` | `MODEL_EXPRESS_NO_SHARED_STORAGE` enabled + server address configured + adapter implements `load_via_native` | Stream the model's weight files from ModelExpress Server into the snapshot the engine already resolved, then hand off to the engine's native loader. The cold-miss path for workers with no route to Hugging Face: the server downloads and caches the model once, and every later worker is served from that cache. Non-weight files arrive earlier, before the engine starts — see [Server-Backed Model Cache](#server-backed-model-cache). Falls through on failure. |
| p2 | `InstantTensorStrategy` | `MX_INSTANT_TENSOR` enabled (default) + no object-store `MX_MODEL_URI` + `instanttensor` installed + CUDA device + adapter implements `build_instanttensor_weight_iter` (and `apply_weight_iter`) | Load the model's own safetensors directly onto CUDA via the `instanttensor` library (distributed loading, pipelined prefetch, direct I/O, GDS when available). Reuses vLLM's built-in `--load-format instanttensor` path, so the engine resolves and (if needed) downloads the weight files. An `s3://`, `gs://`, or `az://` `MX_MODEL_URI` skips this strategy and proceeds to ModelStreamer. Falls through on other failures. |
| p3 | `ModelStreamerStrategy` | `MX_MODEL_URI` set + `runai_model_streamer` installed | Stream safetensors to GPU via CPU staging buffer. `MX_MODEL_URI` accepts remote URIs (`s3://`, `gs://`, `az://`), absolute local paths, or HF model IDs (resolved via `HF_HUB_CACHE`). All storage backends (S3, GCS, Azure) included by default. |
| p4 | `GdsStrategy` | Active accelerator backend supports GDS and GDS hardware is available | Load via `MxGdsLoader` (direct file-to-GPU). Falls through on failure. Reads full checkpoint tensors and slices for TP downstream — see [GDS Reads Full Checkpoint Tensors Under TP](#gds-reads-full-checkpoint-tensors-under-tp). |
| p5 | `DefaultStrategy` | Engine native fallback loader available | Native loader fallback (for vLLM, `DefaultModelLoader`, CPU-staged, auto-downloads from HF Hub). |

When a vLLM loading attempt fails after mutating the model, retry cleanup
unregisters its layers and clears the old model's tensor graph before allocating
the replacement. Caller frames can still retain the old root object, so clearing
only `LoadResult` cannot release its GPU storage. Cleanup also clears the state
of child modules that own parameters: native tensor aliases can retain replaced
parameters and their bound weight-loader callbacks outside Python's GC traversal.
Shared parameterless caches, such as rotary embeddings, remain intact. Remaining
cycles are collected before the allocator cache is emptied and initialization starts.
If initialization fails, recovery aborts without attempting another loader.

See [ModelExpress Benchmarks](BENCHMARKS.md) for measured loading-path, NIXL registration, and artifact-transfer results with explicit timing boundaries.

### External Loading Strategies

`plugins.py` loads external `LoadStrategy` implementations for the shared `INFERENCE` chain. `MX_LOAD_PLUGIN` selects one plugin, inserted immediately before the built-in strategy named by `MX_LOAD_PLUGIN_BEFORE`. The default is `server-cache`, placing the plugin after RDMA. Unknown target names fail before discovery or loading. The six built-in strategies retain their relative order. RL and SGLang's `transfer_engine` path are unchanged. Unset or empty `MX_LOAD_PLUGIN` skips discovery and external imports and ignores `MX_LOAD_PLUGIN_BEFORE`.

Entries are named `mx_plugin_<name>.py` and export a no-argument `register_plugin()` returning a concrete, no-argument `LoadStrategy` class with the matching `name`. The registry stores external classes only. Discovery validates entries and duplicate names; the inference chain separately rejects built-in name collisions before selecting a plugin. Unknown selections also fail before loading. Modules are cached per worker, while each model load creates a new instance of the selected class. Unselected classes are not instantiated.

Plugins reuse the interfaces exported by `modelexpress.load_strategy`:

| Interface | Contract |
|-----------|----------|
| `requires` | Declare required `EngineAdapter` capabilities. |
| `is_available(ctx)` | Check suitability without downloading artifacts or mutating the model. |
| `load(result, ctx)` | Complete weight loading, then return `LoadResult`. Preparing files alone is not success. A plain `LoadStrategy` subclass runs engine post-processing itself; one that loads through `apply_weight_iter` or `load_via_native` can subclass `WeightsIteratorLoadStrategy` or `NativeLoadStrategy` from `modelexpress.load_strategy.base` and leave it to `finalize`. The executor owns publication. |
| `prepare` / `finalize` / `abort` | Optional phase hooks the executor runs around `load`. The defaults enter and leave the engine's layerwise reload when `ctx.is_reload`. On a cold load, `finalize` runs `post_process` and registers publishable results with NIXL. |
| `rollback(ctx)` | Release strategy-owned resources. Model reinitialization belongs to the executor. |
| `StrategyFailed` | A clean miss uses `mutated=False`; a failure requiring model recovery uses `mutated=True`. |
| `StrategyRecoveryError` | Abort loading rather than attempt another strategy. |

A private strategy wrapper delegates availability, the phase hooks, and cleanup to the plugin. Unknown load or phase exceptions and invalid results become mutated failures, so the existing executor rolls back and attempts model reinitialization before fallback. Availability, cleanup, and recovery errors abort loading. Built-in exception behavior is unchanged. Import and construction must not load weights.

`LoadContext` provides model identity, ranks, target device, and engine capabilities. Plugins must not change identity or publish sources themselves. Internal context fields are not a cross-version compatibility guarantee; validate plugins against the target client and adapter. Distributed plugins must coordinate their ranks and agree on availability.

See [Loading Strategy Plugins](DEPLOYMENT.md#loading-strategy-plugins) for deployment and the [strategy template](../examples/plugins/mx_plugin_example.py) for the entry interface.

### Server-Backed Model Cache

Workers without shared storage need repository files at two different moments, and only one of them is late enough for the strategy chain.

An engine resolves the model long before it loads weights: vLLM calls `snapshot_download` while parsing engine args and rewrites `ModelConfig.model` with the resolved local path, and the tokenizer follows immediately after. Under `HF_HUB_OFFLINE=1` with an empty cache, that call fails before any loader exists. P2P cannot cover it either — it transfers GPU tensors and never repository files, so even a worker that will end up loading over RDMA still needs config and tokenizer on local disk first.

So the two halves are fetched separately:

| Phase | What | When | Where |
|---|---|---|---|
| Metadata | Everything except weights | Before the engine resolves the model | `model_prefetch.ensure_metadata()`, invoked from a `snapshot_download` patch |
| Weights | Weight files only | After `RdmaStrategy` finds no source | `ServerCacheStrategy` |

Fetching metadata unconditionally does not weaken P2P-first, because no weight moves on that path — a live source still serves every byte of the weights.

`model_client.py` wraps the `ModelService` RPCs (`EnsureModelDownloaded`, `ListModelFiles`, `StreamModelFiles`) and validates the stream: chunk offsets must be contiguous, sizes must match the manifest, and every listed file must arrive before the final marker. `model_snapshot.py` owns the local layout, writing `snapshots/<commit>/` and the ref the engine will look it up by — `refs/main` for an unpinned request, `refs/<revision>` for a pinned branch or tag, and none at all for a pin that is already the commit hash. Without the ref it needs, `snapshot_download(local_files_only=True)` cannot resolve a repo id no matter how complete the snapshot is. Metadata is published by renaming a staging directory; weights are added to the live snapshot one atomic rename at a time, and a failure part-way through rolls back the files it already published rather than leaving a partial weight set the engine would load as complete. Both phases target the snapshot root the engine resolved: `ServerCacheStrategy` recovers the root and the commit from the standard `<root>/models--<org>--<name>/snapshots/<commit>` path in `ModelConfig` and passes them down, because the `EngineCore` process that loads the weights never ran the prefetch and its own default root may hold nothing the engine will read.

The metadata cache keeps a private inventory at `<root>/models--<org>--<name>/.modelexpress-metadata/<commit>.json`, outside `snapshots/` so it cannot become an apparent repository file or quantization config. It records the repository, commit, schema version, and the complete server-provided non-weight manifest's paths and sizes. The inventory is written atomically and fsynced only after the snapshot is published or an existing snapshot is validated against a revision-confirmed server manifest. Missing inventory is a cache miss; malformed, unsafe, or unreadable inventory is diagnosed and cannot establish readiness. File presence, size and cache containment are rechecked on every reuse, including the in-process prefetch memo. This is a completeness check, not content hashing against equal-size changes.

The inventory is an optional optimization credential. Failure to persist it after successful metadata validation emits a warning and returns the usable snapshot; invalid manifests and incomplete metadata still raise errors. A file occupying the reserved directory name, a symlinked inventory directory, or a non-regular inventory target is not overwritten. Temporary-file cleanup is best-effort and never deletes a file whose creation failed. Without a valid persisted inventory, prefetch memo reuse is unavailable: legacy unconfirmed installs or sidecar persistence failures can cause repeated metadata RPCs, including repeated streams within one process.

A request whose original revision is a **lowercase 40-hex commit** can reuse a matching inventory before acquiring the repository lock or creating an RPC stub. This lets a warm pinned reader proceed even while another process holds that lock to install metadata for a mutable ref. The warm hit writes no ref. A miss runs the existing `EnsureModelDownloaded` and `ListModelFiles` path before taking the repository lock; the existing in-lock reuse check picks up a snapshot another worker published meanwhile without streaming it again. Branches, tags, uppercase hashes and unpinned requests retain server confirmation; legacy or synthetic snapshot names outside the lowercase commit format do not take this shortcut.

On the RPC path, each phase opens with `EnsureModelDownloaded` and pins everything after it to the revision that call reported, so the manifest and the stream come from one commit and a default revision that moves mid-phase cannot mix two. When the server names no revision, later calls remain unpinned and stream validation still applies, but the install does not create an inventory that could bypass a later pinned request's confirmation. The metadata phase asks with `ignore_weights=true`; the server keys its registry entry on the weight mode, so that claim cannot satisfy the weight phase's later full-weight request, and a cold server does not fetch weights before `RdmaStrategy` has had its chance at them.

A pinned revision travels the whole way. When the inventory shortcut does not apply, the metadata phase asks the server for the revision the engine requested and refuses to continue unless the server confirms it: a server predating the revision field reports no `resolved_revision` at all, and a commit hash that resolves to a different commit is refused outright, since a commit names one revision and cannot resolve to another. What lands on disk is shaped by the engine's own lookup rather than by the answer: `snapshots/<commit>/` always, plus `refs/<revision>` when the requested revision is not the commit hash — the rule `huggingface_hub` applies to its own cache, since a lowercase 40-hex resolves by directory name while a branch, a tag, or an uppercase hash needs the ref. The ref belongs to the request rather than to the snapshot, so it is recorded when an existing snapshot is reused as well: a commit installed under its own hash leaves no ref behind, and a later request for a branch resolving to it would otherwise reuse the directory with nothing to find it by. A pin for anything other than `main` leaves `refs/main` untouched — the default revision is a different question, and answering it with a pin would misdirect every later unpinned resolution sharing the cache. Reuse of a pinned snapshot is decided by looking at `snapshots/<commit>/` directly, for the same reason.

`model_prefetch` shares its strict metadata installer with `ServerCacheStrategy`. Missing snapshots and snapshots with known or unreadable inventories require validation or recovery at the path's own root and commit. The strategy alone preserves the legacy path for an existing standard snapshot directory whose inventory is positively absent: it passes that directory to `install_weight_files` without making a new metadata pin request. This only identifies a weight-loading target; it is not an inventoried cache hit, does not claim complete metadata, and does not populate the prefetch memo. An invalid inventory or missing metadata covered by an inventory cannot use this directory fallback. Ordinary local and nonstandard paths keep their previous behavior.

The legacy weight path continues to accept an unconfirmed download response, retry an unsuccessful pinned download without a revision on `grpc.RpcError`, and reject streamed weights whose commit differs from the snapshot directory. The metadata helper and client retain their stricter pin-confirmation rules. This cache layer does not add frontend, EngineCore or worker lifecycle hooks: the existing prefetch patch and post-RDMA-miss strategy still determine when preparation runs. A weight-loading target is not proof that tokenizer and processor files are available before initialization in a separate filesystem.

Cold RPC timeouts are unchanged; a cache miss can still wait for a slow or unreachable server. The weight phase pins to the commit its snapshot is named after and degrades to an unpinned request when the pinned call fails with a `grpc.RpcError`, the shape a failed pin resolve takes; a download failure the server reports through the status stream is raised, not retried. On an unpinned request a server that already holds the model may report no revision at all, so the metadata phase restreams instead of reusing what is on disk. That legacy behavior remains available without turning the unconfirmed result into a pinned-cache shortcut.

Strategies handle the loading path and NIXL tensor registration. `LoadContext.accelerator_backend` centralizes accelerator-specific torch operations and capability gates for fast paths such as pool registration, VMM arena registration, and GDS. Backends that do not support those CUDA-specific paths, such as XPU, leave the gates disabled and use the generic fallback path. XPU transfer deployments still require a UCX/NIXL runtime that can register XPU device memory. Adapter hooks handle engine lifecycle such as vLLM `process_weights_after_loading`, and the chain performs best-effort metadata publication after a successful strategy. New shared inference strategies are registered in `LoadStrategyChain.run()`.

### Source Selection

Accelerator compatibility (`metadata/payload.py::accelerators_compatible`) is the single rule shared by RDMA tensor source selection and artifact discovery, in both their pre-fetch and post-fetch checks. Empty values are unknown and accepted for backward compatibility. On the authoritative post-fetch check an unknown (empty) accelerator on a quantized weight identity fails closed — an unknown family could be a different vendor whose kernels expect a different quantized layout, so quantized weights ride P2P only on a verified same-family match. The pre-fetch check defers an unknown accelerator instead of rejecting it (the lightweight `SourceInstanceRef` may legitimately omit the accelerator — the `k8s-service` backend publishes a synthetic ref and only learns the real accelerator from `GetTensorManifest`), so a valid same-family quantized source is not stranded before its accelerator is known. Exact family matches are always compatible. Cross-family transfer is allowed only for source types in `HETEROGENEOUS_NIXL_SOURCE_TYPES` and family pairs in `HETEROGENEOUS_NIXL_WEIGHT_PAIRS` — today `MX_SOURCE_TYPE_WEIGHTS` over the `cuda`/`xpu` pair (both directions) — and only for **unquantized** weights. Unquantized weights (`SourceIdentity.quantization` empty/`none` and a non-quantized `dtype`) are plain tensor bytes whose shape and dtype are stable across accelerator families, so they carry cross-vendor. Quantized weights are rejected cross-family: `process_weights_after_loading()` repacks fp8 scales, pads/swizzles fp4/nvfp4, and stashes hidden quant-config tensors into kernel- and hardware-specific layouts, so the same logical weights have different bytes on a different vendor's kernels. Copying them cross-vendor corrupts inference silently — the transfer API succeeds and the tensor manifest and CRC match, but the receiver's kernel misreads the layout. Quantized cross-family transfer stays blocked by default; a future hardware validation may allowlist specific quantization/layout pairs that are proven inference-correct after RDMA (not just that the transfer API succeeds). ModelExpress does not dequantize/requantize or convert post-processed layouts during transfer — the RDMA path overwrites already post-processed target tensors with source bytes, so a proven pair is enabled by allowlisting it, not by correcting weights after transfer. Expressing a quantized allowlist entry would need an extension beyond the family-level `HETEROGENEOUS_NIXL_WEIGHT_PAIRS` (keyed on the accelerator pair plus quantization/dtype/backend/arch facts); it is intentionally not built until a pair is proven. See the FP8 Model Handling section. Generated artifacts (CUDA graphs, torch.compile, Triton, DeepGEMM, TileLang, CuTe, FlashInfer caches) are accelerator/arch-specific and stay strict same-family. The `mx_source_type` argument defaults to `None`, and `quantization`/`dtype` default to a sentinel, so the gate fails closed: a caller that does not explicitly pass a hetero-eligible source type and declare the identity's quantization only ever gets strict same-family compatibility. Same-family transfer is unconditional — quantization never gates it, since source and target run identical post-processing. `MX_SOURCE_TYPE_LORA` is deferred — it has no live tensor publish/transfer path yet, so heterogeneous LoRA is rejected until one exists and is tested (add it to `HETEROGENEOUS_NIXL_SOURCE_TYPES` at that point). Heterogeneous transfer still requires a UCX/NIXL runtime that can register both families' device memory; the Rust server is a pure passthrough of the runtime `accelerator` string and applies no compatibility logic.

After `RdmaStrategy` filters listed READY sources to the target's `worker_rank` and to a compatible runtime `accelerator`, it ranks the surviving candidates through a `SourceSelector` (`source_selection.py`) before slicing to `MAX_SOURCE_RETRIES`. Selectors are scoring-based: `ScoredSelector` subclasses implement `score(candidate, context)` and the base orders by descending score. The policies below ship today, resolved by name through a small registry (`MX_P2P_SOURCE_SELECTOR`, default `random`, unknown values fall back to `random`):

- `random` — behavior-preserving default; shuffles with a local RNG so it does not perturb process-global state.
- `rendezvous_hash` — stateless deterministic spreading via HRW hashing of the target identity plus each candidate identity. Different targets get different first choices without a shared counter or server coordination, it is stable across process restarts (blake2b, not Python's salted `hash()`), and adding/removing one source perturbs only a fraction of rankings.
- `load_aware` — biases the `rendezvous_hash` ordering away from busy sources: `score = unit_hash(target, candidate) − w_load · source_load`, where `source_load ∈ [0,1]` is each candidate's server-advertised busyness and `w_load` is `MX_P2P_LOAD_WEIGHT` (default 1.0). It steers new targets toward sources with spare headroom, so pulling weights does not contend with a source's in-flight inference RDMA (e.g. a prefill node streaming KV cache). When `source_load` is 0/unset (idle sources, or older servers) every penalty is equal and ordering collapses exactly to `rendezvous_hash` — never worse than the deterministic baseline.
- `topology_aware` — locality-first spreading for the RDMA fabric. Ranks each candidate by the **narrowest topology domain the target and source share** (same rack → block → datacenter → cross-datacenter), with the `rendezvous_hash` digest as an intra-tier tiebreak so equidistant peers still spread. The level hierarchy is not hard-coded: it comes from `MX_P2P_TOPOLOGY_LEVELS` (the ordered domains of the cluster's **Grove** `ClusterTopology` — `clustertopologies.grove.io` — `spec.levels`), and per-source domain values from each candidate's published `topology` metadata — so MX consumes the same node-label topology Grove/Dynamo already define. NVLink is not modeled (co-located replicas let NIXL auto-select the NVLink backend). See below.

**The `source_load` signal (stateless).** `source_load` is produced by the source about *itself* and only passed through by the server, so MX servers stay stateless behind a load balancer (no server-side counters; every replica ranks identically). The signal has two providers, composed by `make_source_load_provider` (`nic_metrics.py`) and reported on each `UpdateStatus`; the server stamps the value onto `SourceInstanceRef`. (1) The **NIC provider** (`nic_metrics.py`, always on) samples the source's RDMA NIC utilization from its own InfiniBand port counters (`/sys/class/infiniband/<dev>/ports/1/counters`) over the heartbeat interval — the GPU-affine rail, falling back to the busiest node NIC. (2) The **runtime provider** (`runtime_load.py`, enabled by `MX_P2P_RUNTIME_METRICS_URL`) scrapes the co-located vLLM/SGLang `/metrics` for a serving-load proxy (vLLM KV-cache usage, SGLang token usage), which is *predictive* of imminent RDMA. When both are active the published `source_load` is their **max**. Because the wire contract is just the normalized value, adding providers touches neither the server, the proto, nor the selector.

The selector controls ordering only; `RdmaStrategy` enforces the fixed three-candidate retry budget. Selectors read the few fields they need (`worker_rank`, `worker_id`, `identity.model_name`, and `source_load` for `load_aware`) directly off the live `LoadContext` and candidate refs, so there is no parallel context type; `topology_aware` additionally reads the target's own domain from the topology provider and each candidate's from its `topology` metadata. The same scoring interface accepts further policies as drop-ins, and `topology_aware` composes with `load_aware` (see the topology mechanism below). Selection decisions are emitted as structured logs, and an opt-in Prometheus collector (`metrics.py`, `MX_METRICS_ENABLED=1`) re-emits them as `mx_p2p_*` metrics (including `mx_p2p_source_load`) for benchmarking different schemes.

**Topology mechanism.** A worker reports its RDMA-fabric location as a `{domain: value}` map keyed by **Grove `ClusterTopology` domain** (the enum `region`/`zone`/`datacenter`/`block`/`rack`/`host`/`numa`), e.g. `{"block": "b1", "rack": "r3", "host": "node7"}`, read from the environment (`MX_P2P_TOPOLOGY`, populated by the operator from the node's labels via the `ClusterTopology` domain→key mapping) by `topology.py` and published **once at registration** on `WorkerMetadata.topology`. Because it is static per node, it rides the existing publish path (not the heartbeat) — the server passes it straight through onto `SourceInstanceRef.topology`, keeping it stateless. `TopologyAwareSelector.score` returns `(shared_depth, tiebreak)`: `shared_depth` is the deepest configured level at which the target and candidate maps agree, so `order()` puts the closest source first. Setting `MX_P2P_TOPOLOGY_LOAD_WEIGHT > 0` swaps the intra-tier tiebreak for `unit_hash − w·source_load` (read defensively), so **within a locality tier** selection also steers away from busy sources — this is how topology- and load-aware selection compose without either depending on the other. Missing node topology, or an old server that predates the field, collapses the policy to `rendezvous_hash`, so it is never worse than the deterministic baseline; missing levels alone do not, since `MX_P2P_TOPOLOGY_LEVELS` defaults to the full Grove order. Meaningful benefit needs a topology-diverse allocation (sources across multiple racks/blocks); packed into one rack all sources are equidistant and the policy collapses to `rendezvous_hash`.

**Transfer-domain policy.** Ordering alone cannot keep a target off an unreachable source: with fewer than three same-domain sources, the retry slice still reaches cross-domain ones. `MX_P2P_TOPOLOGY_DOMAIN` / `MX_P2P_TOPOLOGY_ENFORCEMENT` add a constraint modeled on Dynamo's KV-transfer policy (`DYN_KV_TRANSFER_DOMAIN` / `DYN_KV_TRANSFER_ENFORCEMENT`). `topology.apply_policy` runs on the selector's output in both `RdmaStrategy._find_source_instances` and the SGLang transfer-engine path, and on NIXL artifact source refs before metadata fetch. `required` keeps only sources whose published value for the domain equals this node's, failing closed on missing values; `preferred` is a stable same-domain-first partition. Under `required`, NIXL artifact discovery also rechecks fetched worker topology before returning an endpoint, and an unknown local domain returns a cache miss without listing artifact sources, allowing local compilation. Artifact discovery reuses `warn_if_no_domain_match` when all rank- and accelerator-compatible refs are outside the required domain. Artifact publication includes topology from the current publish attempt; an unknown required domain fails before cache-bundle preparation, leaving the background publisher to retry when topology arrives. A final topology check after preparation validates published metadata against the current required domain before source registration or chunk-manager allocation. Older artifact sources without topology are ineligible under `required`. When a policy is set, the weight-selection `mx_p2p_candidates` funnel gains a `topology_matched` stage counting same-domain candidates; without one the stage is not emitted.

#### Selection efficacy (measured)

The two Phase 1 policies were compared two ways. An offline simulation drives the real selector code over synthetic `(target, source)` identity sets; because the score depends only on identity and policy (not on hardware or RDMA), it reproduces the on-cluster `mx_p2p_source_selections_total` distribution without standing up transfers. Across M sources / N targets configs (4×20, 4×40, 8×32), **first-choice balance is equivalent** for both policies (both are uniform hashes — max-source share within a few percent of each other and of the `ceil(N/M)/N` ideal). The difference is elsewhere: `rendezvous_hash` has **0% re-pick churn** on a repeated, unchanged source set (vs `random`'s ~`(M-1)/M`), and removing one source changes only ~`1/M` of the unaffected targets' picks.

On-cluster (8×B200 nodes, InfiniBand RDMA; vLLM `--load-format modelexpress`, TP=1, central-coordinator backend), both policies were run against pre-warmed source pools that grow as targets become sources. Across Qwen2.5-0.5B (0.99 GB) and Qwen2.5-7B (~15 GB), 42 real cross-node NIXL RDMA transfers completed and **bandwidth was policy-independent** (~180 Gbps for the 0.5B transfers, ~211 Gbps for the larger 7B transfers as setup amortizes). So `rendezvous_hash`'s win is determinism and low disruption at no balance or bandwidth cost; the `load_aware` policy closes the live-fan-out gap where deterministic hashing can pile onto an always-present source, by biasing selection away from sources that publish a high `source_load`.

### Transfer Safety

`RdmaStrategy.is_available()` calls `transfer_safety.check_transfer_allowed()` before attempting P2P transfer. The function logs the model's detected features (attention type, quantization, MoE) and currently allows all combinations — no feature is blocked. The function is kept as a hook for future safety gates.

NVFP4 MoE models (known case: Kimi-K2.5-NVFP4) previously produced corrupted inference after RDMA transfer despite all registered tensor bytes matching. This is a specific instance of a broader class of bugs: post-processing stashes computed state on non-Module objects (e.g. `FusedMoEQuantConfig.a1_gscale = 1/activation_scale` on the quant method), which is invisible to `named_parameters()`/`named_buffers()`. On the target those values are computed from dummy weights before RDMA, and RDMA only overwrites the registered tensors, so the stashed values stay wrong. The fix (`adopt_hidden_tensors()`) recursively scans module attributes for orphaned accelerator tensors (via `is_accel_tensor()`) and registers them as non-persistent buffers so they are included in the RDMA manifest. Verified correct on vLLM v0.17.1 and v0.19.0 with Kimi-K2.5-NVFP4 and on DeepSeek-V3 (MLA + FP8).

During transfer, `ManifestMismatchError` is raised if source and target tensor names or sizes don't match (a likely symptom of a rolling update where pods run different image versions). The receive path converts it, like any receive failure, into a `SourceTransferError`. Before trying the next ranked source, `RdmaStrategy` clears its NIXL state and uses the engine adapter to replace a possibly mutated model with a fresh instance. If the final transfer fails after mutation, the outer strategy chain performs the same cleanup before falling through to the next strategy.

After loading by any strategy, the worker starts a `PublisherThread` that owns initial publication retry and then periodically sends `UpdateStatus(READY)` to keep `updated_at` fresh. On clean shutdown, the publisher sends `UpdateStatus(STALE)` via an `atexit` handler. Metadata publish failures are logged and retried in the publisher thread instead of crashing the worker.

Each GPU worker generates a unique `worker_id` (`uuid4().hex[:8]`) at init and publishes independently. Workers use `torch.distributed.get_rank()` as their global rank (captures both TP and PP position).

### Tensor Discovery

The loader uses `iter_module_tensors()` (in `tensor_utils.py`) to walk the full PyTorch module tree via `named_parameters()` and `named_buffers()`, keeping tensors accepted by the active `AcceleratorBackend.is_accel_tensor()` predicate after post-processing. `CudaAcceleratorBackend` and `XpuAcceleratorBackend` are the concrete backends today, so collection targets CUDA or XPU tensors depending on the active backend. This discovers three categories:

| Category | Source | Example |
|----------|--------|---------|
| Parameters | `named_parameters()` | `layers.0.attention.weight` |
| Buffers | `named_buffers()` | Batch norm running mean |
| Promoted tensor attributes | non-persistent buffers added by `capture_tensor_attrs()` / `adopt_hidden_tensors()` | FP8 `weight_scale`, `_k_scale` |

This is more thorough than `named_parameters()` alone, which only finds parameters and would miss tensors created during `process_weights_after_loading()`. Those bare-attribute tensors are surfaced by promoting them to non-persistent buffers (see below) before discovery, so `named_buffers()` then includes them. Non-contiguous tensors (e.g. MLA's `W_UV` and `W_UK_T` sharing a dequantized intermediate) are registered as flat byte views of their underlying storage via `storage_view()`. Multiple views into the same storage are deduplicated by `data_ptr()` so tied weights are only transferred once.

Two mechanisms promote tensors that `named_parameters()`/`named_buffers()` would otherwise miss, both gated on `is_accel_tensor()`:

- `capture_tensor_attrs()` wraps `process_weights_after_loading()` and intercepts bare accelerator tensors assigned directly as module attributes, registering each as a non-persistent buffer.
- `adopt_hidden_tensors()` recursively scans each module's non-Module attributes for accelerator tensors stashed on plain Python objects (e.g. `FusedMoEQuantConfig.a1_gscale`, FlashInfer workspace buffers) and registers them as non-persistent buffers so they appear in the manifest. Without this, the target's quant method objects retain values computed from dummy weights, causing incorrect inference despite all registered tensor bytes matching.

### VMM Arena (CUDAPluggableAllocator hook)

`MX_VMM_ARENA=1` installs a `CUDAPluggableAllocator` that routes every CUDA allocation issued during `initialize_model`, `load_weights`, and `process_weights_after_loading` into a CUDA VMM arena. The arena reserves 16.0 TiB of virtual address space up front via `cuMemAddressReserve`. The reservation consumes VA only. Physical memory is committed later, one PyTorch allocator call at a time, when the backend creates and maps a VMM handle.

The `modelexpress.vmm` subpackage implements the arena (see `modelexpress_client/python/modelexpress/vmm/README.md` for the user-facing overview):

| Module | Role |
|--------|------|
| `vmm/arena.py` | `VmmArena` reserves the large VA range, bump-assigns each `mx_malloc` a stable address, tracks live allocations by VA, and exposes the used high-water range for registration |
| `vmm/backend.py` | `CudaVmmBackend` implements the CUDA VMM calls: reserve VA, `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`, `cuMemUnmap`, `cuMemRelease`, and reserve release. Each allocation rolls back cleanly if a later CUDA call fails |
| `vmm/hook.py` + `vmm/_alloc_ext.cpp` | `use_arena` combines `torch.cuda.use_mem_pool` with the C extension's `mx_malloc` and `mx_free` shims. The C extension is built best-effort by `setup.py`; if compilation fails the install still succeeds, `vmm.ARENA_AVAILABLE` is `False` at runtime, and the loader logs a warning and falls back when `MX_VMM_ARENA=1` is requested |
| `vmm/runtime.py` | Engine-agnostic lifecycle helpers: `maybe_enter_vmm_arena(ctx)` is the contextmanager each engine adapter wraps around its load envelope (env-var handling, per-device arena dict, `use_arena` wrapping). `log_arena_post_load(ctx)` emits a diagnostic line after the load body returns |

The allocation model matches PyTorch's caching allocator boundary. PyTorch already groups tensors into allocator segments before it calls the plugin, so ModelExpress gives each plugin allocation its own VMM physical handle instead of sub-allocating again inside larger chunks. `mx_malloc(size)` rounds to CUDA VMM granularity, calls `cuMemCreate`, maps that handle at the next arena VA, and returns the mapped address. `mx_free(ptr)` looks up the allocation, calls `cuMemUnmap`, then `cuMemRelease`, so selective frees during post-processing return physical memory before NIXL registration.

After the load strategy succeeds and the engine has finished post-processing, `LoadStrategyChain` registers the arena instead of registering tensors one by one. `register_arena` calls `cuMemGetHandleForAddressRange` over `[arena.base, arena.base + arena.used_bytes)`, then performs one dmabuf-backed registration for the used range. Tensor descriptors still list every tensor by name and offset, but all offsets share one lkey from the arena MR.

The arena path does not require `MX_POOL_REG=1`. Pool-reg remains the optimization for normal cudaMalloc deployments. Arena deployments use the direct `register_arena` seam because they already know the contiguous VA range and do not need `cuMemGetAddressRange` to rediscover allocation boundaries.

Empirical validation on B200 + ConnectX showed that a dmabuf MR over a multi-handle VMM range remains valid when holes exist inside the registered VA range. That is the property that lets `process_weights_after_loading` allocate replacement tensors, free discarded tensors, and still finish with one MR for the surviving used arena range. That property is specific to the dmabuf/IB path: `cuda_ipc` cannot address a single MR spanning several `cuMemCreate` handles, so multi-allocation arenas fall back to per-tensor registration there. See [Multi-handle arenas](DEPLOYMENT.md#multi-handle-arenas).

## NIXL Integration

### What is NIXL?

NIXL (NVIDIA Interconnect eXchange Library) provides zero-copy GPU-to-GPU RDMA transfers on top of UCX:

| Concept | Description |
|---------|-------------|
| **Agent** | NIXL instance managing one GPU's memory registrations and transfers |
| **Memory Registration** | GPU memory must be registered before RDMA access; generates rkeys |
| **Metadata** | Serialized agent info (address, rkeys) shared between source/target |
| **Transfer Descriptor** | Prepared list of (addr, size, device) for bulk transfer |
| **rkey** | Remote key - RDMA authorization token for remote memory access |

### How ModelExpress Uses NIXL

```python
# 1. Create NIXL agent (one per GPU worker)
from nixl._api import nixl_agent, nixl_agent_config
config = nixl_agent_config(backends=["UCX"])
agent = nixl_agent("worker-0", config)

# 2. Register GPU tensors for RDMA access
tensors = [(tensor.data_ptr(), tensor.numel() * tensor.element_size(), device_id, "")]
agent.register_memory(tensors, "VRAM")  # NIXL accelerator memory segment

# 3. Get metadata for remote agent connection
metadata = agent.get_local_md()  # Share this with target

# 4. On target: connect to source and transfer
agent.add_remote_agent("source-worker-0", source_metadata)
src_descs = agent.prep_xfer_dlist("source-worker-0", source_tensors, "VRAM", ["UCX"])
dst_descs = agent.prep_xfer_dlist("", local_tensors, "VRAM", ["UCX"])
handle = agent.make_prepped_xfer("READ", dst_descs, indices, src_descs, indices, ["UCX"])
agent.transfer(handle)

# 5. Wait for completion
while agent.check_xfer_state(handle) not in ("DONE", "SUCCESS"):
    time.sleep(0.001)
agent.release_xfer_handle(handle)
```

## FP8 Model Handling (DeepSeek-V3)

vLLM's `process_weights_after_loading()` transforms model weights into kernel-friendly formats (FP8 scale repacking, NVFP4 padding/swizzling, MLA dequantized projections) and may create new tensors as bare attributes, buffers, or on quant method objects.

The solution: both source and target run `process_weights_after_loading()` first, then `adopt_hidden_tensors()` discovers transferable accelerator tensors on non-Module objects (quant configs, kernel objects), and finally `register_tensors()` collects everything for RDMA. The target runs post-processing on dummy data to establish the correct tensor layout and receives the real data, including hidden quant config tensors, via RDMA.

The vLLM adapter passes an exclusion predicate to engine-agnostic hidden-tensor discovery, skipping the exact `vllm.models.kimi_k3.nvidia.ops.latent_moe_tail.KimiK3LatentMoETailOp` class. Its GPU peer-pointer arrays, barriers, and multicast workspaces are process-local collective state and must retain the receiver's allocations and contents. Transferring donor collective state can cause CUDA illegal-address errors during receiver warmup. Hidden quantization tensors outside this operator remain eligible for adoption and RDMA transfer.

```mermaid
graph TD
    subgraph Source
        S1[Load real weights from disk]
        S2[process_weights_after_loading]
        S3[adopt_hidden_tensors - find orphaned accelerator tensors]
        S4[Register ALL tensors with NIXL]
        S5[Publish metadata]
        S1 --> S2 --> S3 --> S4 --> S5
    end
    subgraph Target
        T1[Load dummy weights]
        T2[process_weights_after_loading on dummy]
        T3[adopt_hidden_tensors - same layout as source]
        T4[Register ALL tensors with NIXL]
        T5[Receive processed weights via RDMA]
        T1 --> T2 --> T3 --> T4 --> T5
    end
    S4 -- "RDMA" --> T5
```

## Coordination Protocol

### Flow

1. **Source loads**: Loads weights through the fixed strategy chain (server cache, InstantTensor, ModelStreamer, GDS, or the native loader when no P2P source is available), or receives them over P2P, then runs `process_weights_after_loading()`
2. **Source publishes**: Registers tensors with NIXL, or prepares a sealed cache artifact bundle, then a `PublisherThread` calls `PublishMetadata(identity, worker, worker_id)` -> gets `mx_source_id` (status=INITIALIZING). `WorkerMetadata.accelerator` records runtime accelerator family for compatibility filtering; it is not part of `SourceIdentity` or the source-id hash. In P2P mode (`MX_P2P_METADATA=1`, or auto-forced on by decentralized backends like `k8s-service`), publishes only lightweight endpoint pointers and starts a `WorkerGrpcServer` for tensor manifests or artifact manifest/chunk serving.
3. **Publisher heartbeats**: `PublisherThread` sends `UpdateStatus(READY)` every 30s after publication succeeds, refreshing `updated_at`
4. **Target discovers**: Calls `ListSources(identity, status=READY)`, which returns only READY workers whose `updated_at` is still within `MX_HEARTBEAT_TIMEOUT_SECS`, then filters by `worker_rank` and compatible runtime `accelerator`
5. **Target fetches on demand**: Calls `GetMetadata(mx_source_id, worker_id)` for the chosen candidate. Accelerator-incompatible sources were already dropped using `SourceInstanceRef.accelerator`; this step re-checks the authoritative `WorkerMetadata.accelerator` as defense-in-depth. Empty accelerator metadata is accepted for backward compatibility. If `worker_grpc_endpoint` is populated, the target fetches and retains a version-validated tensor manifest before target preparation.
6. **Target transfers**: Executes the NIXL reads using the prefetched manifest. Preparation failures may select the next configured source. A failure after a possible model mutation requires engine reset or restart before serving. For cache artifacts, the target receives one source chunk at a time into registered DRAM, verifies CRC32C, writes to target-local staging, then installs the staged tar into the runtime cache directory. Generation mismatches and transfer failures try the next candidate; a possibly mutated target is reinitialized before retry.
7. **Target becomes source**: After receiving weights or installing a cache artifact, publishes own metadata and starts its own heartbeat
8. **Stale detection**: Server-side reaper marks workers STALE if `updated_at` > 90s old; `ListSources(READY)` also applies this heartbeat freshness check at query time so expired READY records are not returned while waiting for the next reaper pass. GC deletes STALE workers after 1 hour

Zero-byte and unallocated (null-address) tensors remain in manifests and
participate in exact name, size, and dtype validation. On the sender they stay
in the published manifest but are excluded from NIXL registration; on the
receiver they count as matched tensors but are omitted from NIXL descriptor
lists because they have no registered memory range; a nonzero-size tensor
skipped this way logs a warning, since the local tensor keeps its current
values. Arena registration ignores such tensors in its containment check. Both
sides share one predicate for this. A manifest made entirely of empty tensors completes as a
successful no-op transfer.

Tarred cache artifacts carry regular files and directories only. The source
side enumerates members explicitly and hands tar that list, so nothing else can
enter the archive regardless of how a cache directory is laid out; the target
side enforces the same invariant in `_validate_tar_members` before extraction.
Symlinks are skipped at packaging time (see [`DEPLOYMENT.md`](DEPLOYMENT.md#p2p-metadata-exchange)).

Cache artifact checksums protect transfer integrity but do not authenticate the
source or attest the contents. TorchInductor, Triton, DeepGEMM, TileLang, CuTe
DSL, and FlashInfer caches may
contain executable code, so artifact transfer must be limited to trusted
replicas and a trusted MX server, with MX and worker gRPC endpoints isolated
from untrusted networks. ModelExpress does not currently sign cache artifacts.

See [`metadata.md`](metadata.md) for the full storage schema and debugging guide.

## Environment Variables

### ModelExpress Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MX_SERVER_ADDRESS` | `localhost:8001` | gRPC server address (recommended) |
| `MODEL_EXPRESS_URL` | `localhost:8001` | Deprecated in favor of `MX_SERVER_ADDRESS`. Still read by all client paths and still takes precedence when both are set, because the TRT-LLM live-transfer integration reads only this name. It is removed once that path reads `MX_SERVER_ADDRESS`; until then set both to the same value. |
| `MX_DISABLE_PATCHES` | `0` | Emergency escape hatch that skips all runtime compatibility patches. Set to `1`, `true`, `yes`, or `on` if a patch is incompatible with the installed engine. |
| `MX_METADATA_BACKEND` | (required on server; `""` on client) | Server: `redis` or `kubernetes`. Client: `""` / `server` / `redis` / `kubernetes` (central server) or `k8s-service` (decentralized via K8s Service routing) |
| `MX_POOL_REG` | `0` | Discover cudaMalloc allocations via `cuMemGetAddressRange` and register each as a single NIXL block instead of registering tensors individually. Reduces NIXL registration count by 80-99% on typical vLLM models, cutting `ibv_reg_mr` time and metadata blob size; transfer semantics unchanged. Not required for `MX_VMM_ARENA=1`, which registers the arena directly |
| `MX_VMM_ARENA` | `0` | Install a `CUDAPluggableAllocator` that routes weight-loading allocations into a CUDA VMM arena, then registers the used arena range once through dmabuf at end-of-load. Reserves 16.0 TiB of VA by default and commits physical memory only for mapped allocations. See [VMM Arena](#vmm-arena-cudapluggableallocator-hook) |
| `UCX_CUDA_COPY_REG_WHOLE_ALLOC` | (UCX default) | Set to `off` with `MX_VMM_ARENA=1` on any UCX predating the `cuda_copy_md` length-truncation fix (openucx/ucx#11461). Scoped to the `cuda_copy` transport; it does not affect `cuda_ipc`. |
| `MX_P2P_METADATA` | `1` | Enable P2P metadata exchange on source workers. Set to `0` to publish full metadata through a central-coordinator backend; ignored on decentralized backends that require P2P metadata |
| `MX_METADATA_PORT` | `5555` | Base NIXL listen port; effective port is `MX_METADATA_PORT + device_id` |
| `MX_WORKER_GRPC_PORT` | `6555` | Base worker gRPC port for P2P tensor and artifact manifest serving; effective port is `MX_WORKER_GRPC_PORT + device_id` |
| `MX_WORKER_HOST` | (auto-detect) | Override worker IP/hostname for P2P endpoints |
| `MX_ARTIFACT_TRANSFER` | `0` | Opt in to cache artifact transfer. The vLLM loader uses it for torch compile, Triton, DeepGEMM, TileLang, CuTe DSL, and FlashInfer JIT caches, including persistent autotune files when supported by vLLM. The SGLang NIXL loader uses it for compatible torch compile, Triton, TVM-FFI, DeepGEMM, TileLang, CuTe DSL, and FlashInfer caches. The default `nixl` artifact backend requires P2P metadata and is skipped when `MX_P2P_METADATA=0`; the `mooncake` backend is independent of P2P metadata. |
| `MX_ARTIFACT_BUNDLE_ROOT` | `$TMPDIR/modelexpress-artifacts` | Staging root for tarred cache artifact bundles |
| `MX_ARTIFACT_READY_URL` | Framework default | Readiness endpoint polled before publishing weight metadata or preparing and publishing cache bundles. Defaults to `http://127.0.0.1:8000/health` for vLLM and `http://127.0.0.1:30000/health` for SGLang. Each probe allows 1 second for vLLM and 5 seconds for SGLang, whose health endpoint may generate a token before responding. On the non-head nodes of a multi-node engine a loopback host is rewritten onto the head's address, preserving the configured port and path; a non-loopback host is used verbatim |
| `MX_ARTIFACT_READY_TIMEOUT_SECS` | `1800` | Maximum time the artifact publisher waits for readiness and successful publication before giving up |
| `MX_ARTIFACT_COMPILE_CONFIG_DIGEST` | `""` (unset) | Feeds the torch compile cache `SourceIdentity`, adding compile configuration as a partitioning dimension for artifact discovery. Unset leaves the field empty, which drops it from the `mx_source_id` input, so workers whose other identity fields match — model, tensor/pipeline/expert parallel size, dtype, quantization, revision, vLLM/torch/CUDA/Triton versions, GPU arch — share one pool even when their compile configurations differ. See [Pairing workers by compile configuration](DEPLOYMENT.md#pairing-workers-by-compile-configuration) |
| `MX_MODEL_REVISION` | (from vLLM config) | Override for `SourceIdentity.revision`. Pin to the exact checkpoint identifier so `mx_source_id` is content-addressed |
| `MX_K8S_SERVICE_PATTERN` | `mx-sources` | DNS template for the `k8s-service` backend; `{rank}` is substituted with the worker's own rank. Client auto-appends `:{MX_WORKER_GRPC_PORT + rank}` if the resolved pattern has no explicit port |
| `MX_K8S_SOURCE_RETRIES` | `5` | `k8s-service` max retries on `FAILED_PRECONDITION` (rolling-update transients). Fresh gRPC channel per attempt so kube-proxy re-picks a backend |
| `MX_K8S_SOURCE_BACKOFF_SECONDS` | `0.5` | `k8s-service` sleep between retry attempts |
| `MX_HEARTBEAT_INTERVAL_SECS` | `30` | Client heartbeat frequency |
| `MX_HEARTBEAT_TIMEOUT_SECS` | `90` | Server reaper staleness threshold |
| `MX_REAPER_SCAN_INTERVAL_SECS` | `30` | Server reaper scan frequency |
| `MX_REGISTRY_STATS_INTERVAL_SECS` | `60` | Registry-statistics refresh frequency. Writes `mx_registry_entries` and `mx_state_entries`; each pass walks the keyspace, so it is deliberately coarser than a scrape and runs independently of cache eviction |
| `MX_GC_TIMEOUT_SECS` | `3600` | Time before stale entries are deleted |
| `VLLM_RPC_TIMEOUT` | `7200000` | vLLM RPC timeout in ms |

### UCX/NIXL Tuning

| Variable | Recommended | Description |
|----------|-------------|-------------|
| `UCX_RNDV_SCHEME` | `get_zcopy` | Zero-copy RDMA reads |
| `UCX_RNDV_THRESH` | `0` | Force rendezvous for all transfers |
| `NIXL_LOG_LEVEL` | `INFO` | NIXL logging (DEBUG for troubleshooting) |
| `UCX_LOG_LEVEL` | `WARN` | UCX logging (DEBUG for troubleshooting) |

## Known Issues

### NIXL_ERR_REMOTE_DISCONNECT

Target fails with `Remote access error on mlx5_X:1/IB`. Common causes: source crashed/restarted (stale rkeys), UCX transport misconfiguration, premature target connection. Fix: use robust ready coordination, check for restarts, enable `UCX_LOG_LEVEL=DEBUG`.

### Long Source Warmup

DeepSeek-V3 takes ~40 minutes to warm up (loading + DeepGemm + CUDA graphs). Target must wait via ready coordination.

### GDS Reads Full Checkpoint Tensors Under TP

`MxGdsLoader` issues NIXL/GDS reads using the full `(file_offset, size)` from each tensor's `data_offsets`, then hands the full checkpoint tensor to the engine through the normal `model.load_weights()` path. vLLM's parameter loaders apply TP sharding themselves (`loaded_weight.narrow(...)`) before copying into the final parameter buffer.

So with TP > 1, every rank reads the entire checkpoint tensor through GDS and keeps only its local shard: GDS/disk bytes read scale with the TP degree instead of staying constant, and the path can regress as TP grows while the default disk loader leans on mmap and lazy page faults. End-to-end numbers per TP degree are still to be measured.

A fix needs a TP-aware load contract, or a vLLM-specific path that plans file ranges from parameter metadata and reads directly into the final param buffers; returning rank-local tensors through the same `load_weights()` path would just be sliced again. Optimizing only simple contiguous column-parallel shards is partial: the large column-parallel params (QKV, gate_up) are fused/packed with multiple `narrow()` ranges and the row-parallel params (o_proj, down_proj) are strided, so most bytes stay on the full-tensor path.

## Performance

| Metric | Value |
|--------|-------|
| Model | DeepSeek-V3 (671B, FP8) |
| Total Data | 681 GB (8 workers x 85 GB) |
| Transfer Time | ~15 seconds (8 parallel RDMA streams @ ~45 Gbps each) |
| Per-Worker Speed | ~45 Gbps |
| Theoretical Max | 400 Gbps per NIC |

Optimization opportunities: contiguous regions (blocked), warm source pool, DeepGemm kernel caching, multi-rail RDMA (`UCX_IB_NUM_PATHS=2`).

## Deployment and Configuration

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for the full deployment guide covering server/client configuration, Docker, Kubernetes, Helm, P2P transfer setup, and debugging commands.


## RL staged transfer

The bounded arena design, logical and physical transfer plans, descriptor caching, checksum verification, and failure cleanup are covered in the [RL refit architecture](rl/ARCHITECTURE.md#staged-transfer-caching-and-failure-handling).


## RL refit CI harness

The RL refit CI harness and its scenario lifecycle are described in the [RL refit architecture](rl/ARCHITECTURE.md#rl-refit-ci-harness).
