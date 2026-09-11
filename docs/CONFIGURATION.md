<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Configuration reference

This is the canonical reference for ModelExpress-owned server, client, loader, transport, artifact, and metrics configuration. The Python client reads its environment values at use time from [modelexpress/envs.py](../modelexpress_client/python/modelexpress/envs.py); the Rust server and CLI use [modelexpress_common/src/envs.rs](../modelexpress_common/src/envs.rs) and the server configuration types. Advanced refit and resharding controls remain with the feature implementation because their safety contracts are workload-specific. When a third-party runtime owns a setting, this page names the runtime rather than pretending it is an MX setting.

## Precedence

For the Rust server and CLI, configuration is resolved in this order:

1. CLI flags.
2. Environment variables.
3. YAML/TOML/JSON configuration file.
4. Built-in defaults.

The server still requires the metadata backend environment variables even when a config file is supplied. Use `cargo run --bin config_gen -- --output model-express.yaml` to generate a file and `cargo run --bin modelexpress-server -- --config model-express.yaml --validate-config` to validate it.

## Server configuration

### Minimal server environment

~~~bash
export MX_METADATA_BACKEND=redis
export REDIS_URL=redis://localhost:6379
cargo run --bin modelexpress-server
~~~

For Kubernetes CRDs:

~~~bash
export MX_METADATA_BACKEND=kubernetes
export POD_NAMESPACE=modelexpress
cargo run --bin modelexpress-server
~~~

The server accepts redis, kubernetes, k8s, and crd. Redis requires REDIS_URL, or both MX_REDIS_HOST and MX_REDIS_PORT (the older REDIS_HOST and REDIS_PORT names remain aliases). Kubernetes requires POD_NAMESPACE or MX_METADATA_NAMESPACE. There is no production fallback to localhost Redis or the default namespace.

### Server flags and config-file fields

| Config field | CLI flag | Environment variable | Default |
|---|---|---|---|
| server.host | --host | MODEL_EXPRESS_SERVER_HOST | 0.0.0.0 |
| server.port | --port / -p | MODEL_EXPRESS_SERVER_PORT | 8001 |
| server.metrics_port | --metrics-port | MODEL_EXPRESS_SERVER_METRICS_PORT | 9401; 0 disables |
| cache.directory | --cache-directory | MODEL_EXPRESS_CACHE_DIRECTORY | ./cache |
| cache.eviction.enabled | --cache-eviction-enabled | MODEL_EXPRESS_CACHE_EVICTION_ENABLED | true |
| logging.level | --log-level / -l | MODEL_EXPRESS_LOG_LEVEL | info |
| logging.format | --log-format | MODEL_EXPRESS_LOG_FORMAT | pretty |
| logging.file | config file only | — | unset; stdout |
| logging.structured | config file only | — | false |

The metrics listener is separate from the gRPC port because the server exposes Prometheus metrics over HTTP/1.1 while tonic serves gRPC over HTTP/2. See [Metrics](METRICS.md).

### Cache and security fields

| Config field | Default | Meaning |
|---|---|---|
| cache.max_size_bytes | unset | Maximum cache size; unset means unlimited |
| cache.eviction.policy.type | lru | Eviction policy |
| cache.eviction.policy.unused_threshold | 7d | Age after which an unused model may be evicted |
| cache.eviction.policy.max_models | unset | Maximum model count; unset means unlimited |
| cache.eviction.policy.min_free_space_bytes | unset | Evict when free space falls below this value |
| cache.eviction.check_interval | 1h | Eviction scan interval |
| security.mode | off | ServiceAccount token authentication mode |
| security.token_audiences | [] | Required when security mode is enforce |
| security.allowed_service_accounts | [] | Required when security mode is enforce; values are <namespace>:<serviceaccount> |
| security.cache_ttl_secs | 60 | Verified-token and rejection cache TTL |

When security mode is enforce, set MODEL_EXPRESS_SECURITY_MODE=enforce, at least one token audience, and at least one allowed ServiceAccount. The corresponding CLI flags are --security-mode, --security-token-audiences, --security-allowed-service-accounts, and --security-cache-ttl-secs.

## CLI client configuration

| Environment variable | Default | Meaning |
|---|---|---|
| MODEL_EXPRESS_ENDPOINT | http://localhost:8001 | CLI server endpoint |
| MODEL_EXPRESS_TIMEOUT | 30 seconds | CLI request timeout |
| MODEL_EXPRESS_CACHE_DIRECTORY | auto | Cache override; Hugging Face resolution falls back to HF_HUB_CACHE and then the default Hub cache |
| MODEL_EXPRESS_NO_SHARED_STORAGE | false | Fetch repository files from the server instead of assuming a shared filesystem |
| MODEL_EXPRESS_TRANSFER_CHUNK_SIZE | 32768 bytes for the Rust CLI; 1048576 bytes for the Python server-cache client | File-transfer chunk size; values must be positive and no larger than the implementation maximum |
| MODEL_EXPRESS_LOG_LEVEL | runtime-dependent | CLI log level |

See [CLI](CLI.md) for commands and output formats.

## Client authentication

Python and Rust clients automatically attach a bearer token when the configured projected ServiceAccount token file exists. If the file is absent, clients send unauthenticated RPCs; this is valid only when the server security mode is `off`. The token audience must match one of the server's configured `security.token_audiences` values.

| Environment variable | Default | Meaning |
|---|---|---|
| `MX_AUTH_TOKEN_PATH` | `/var/run/secrets/tokens/modelexpress` | Projected ServiceAccount token file to read |
| `MX_AUTH_TOKEN_TTL_SECONDS` | `60` | Minimum cache time before rereading the token; file mtime changes are also detected |

See [Deployment authentication](DEPLOYMENT.md#serviceaccount-authentication) for the Kubernetes projected-token volume and the server-side security settings.

## Loading strategy selection

ModelExpress does not let callers define an arbitrary order. LoadStrategyChain constructs the following fixed chain:

1. rdma: P2P from a compatible serving peer.
2. server-cache: server-backed weight snapshot for no-shared-storage deployments.
3. instant_tensor: local safetensor loading when the package, CUDA-like device, and adapter capability are available.
4. model_streamer: ModelStreamer when MX_MODEL_URI is set and the adapter/package support it.
5. gds: GPUDirect Storage when the accelerator and adapter support it.
6. default: the engine's native loader.

An eligible strategy can still fail and allow the next strategy to run. If a strategy mutates the model before failing, the adapter reinitializes the model before retrying.

### Eligibility controls

| Environment variable | Default | Effect |
|---|---|---|
| MODEL_EXPRESS_URL | unset | Legacy client/server address; takes precedence over MX_SERVER_ADDRESS when both are set |
| MX_SERVER_ADDRESS | unset | Preferred client address; the Python client falls back to localhost:8001 when neither address is set |
| MODEL_EXPRESS_NO_SHARED_STORAGE | false | Enables server-backed repository-file and weight fetching when an address is configured |
| MX_P2P_METADATA | 1 | Enables on-demand P2P metadata exchange; set 0 for full metadata through a central coordinator |
| MX_MODEL_URI | unset | Enables ModelStreamer for s3://, gs://, az://, or absolute local paths |
| MX_MS_DISTRIBUTED | 1 | Distributes ModelStreamer reads across CUDA TP ranks when TP > 1 |
| MX_INSTANT_TENSOR | 1 | Enables the InstantTensor eligibility gate |
| MX_DISABLE_PATCHES | false | Disables ModelExpress runtime compatibility patches |
| MODEL_EXPRESS_LOG_LEVEL | runtime-dependent | Use DEBUG to inspect Eligible loaders and Trying strategy |

MX_P2P_METADATA=0 only changes the central-coordinator metadata representation; it does not make a decentralized k8s-service backend usable because that backend requires P2P metadata.

## P2P and worker settings

| Environment variable | Default | Effect |
|---|---|---|
| MX_METADATA_BACKEND | empty on client; required on server | redis, kubernetes/k8s/crd, or client-only k8s-service |
| MX_METADATA_PORT | 5555 | Base NIXL metadata port; worker port is base plus device ID |
| MX_WORKER_GRPC_PORT | 6555 | Base worker gRPC port for tensor and artifact manifests |
| MX_WORKER_HOST | auto-detect | Advertised worker host override |
| MX_MODEL_REVISION | unset | Source-identity revision label; pin an exact revision for decentralized source pools |
| MX_NIXL_BACKEND | UCX | NIXL backend; LIBFABRIC is used for AWS EFA |
| MX_P2P_SOURCE_SELECTOR | random | Source ordering: random or rendezvous_hash; unknown values fall back to random |
| MX_SOURCE_QUERY_TIMEOUT | 3600 seconds | TRT-LLM source query timeout |
| MX_TRANSFER_TIMEOUT | 900 seconds for the general client; 300 seconds for RDMA when unset | Transfer timeout used by integrations; the RDMA receive path uses its 300-second fallback until this variable is explicitly set |
| MX_HEARTBEAT_INTERVAL_SECS | 30 | Source heartbeat interval |
| MX_PUBLISH_TIMEOUT_SECS | 1800 | Maximum source publication wait |
| MX_K8S_SERVICE_PATTERN | mx-sources | Service DNS pattern; {rank} is replaced with the worker rank |
| MX_K8S_SOURCE_RETRIES | 5 | Fresh-channel retries for k8s-service revision mismatches |
| MX_K8S_SOURCE_BACKOFF_SECONDS | 0.5 | Backoff between k8s-service retries |

Use a central backend for mixed revisions, live updates, or per-worker addressability. Use k8s-service only for stable, interchangeable source pods; see [Kubernetes Service backend](K8S_SERVICE_BACKEND.md).

## P2P transport and registration tuning

| Environment variable | Default | Effect |
|---|---|---|
| MX_RDMA_NIC_PIN | unset | Set auto for topology-based NIC selection or provide an explicit NIC list |
| MX_RDMA_NIC_PIN_MIN_RATE_GBPS | auto | Minimum link rate for automatic NIC selection |
| NIXL_UCX_TLS | unset | NIXL/UCX transport selection |
| UCX_TLS | unset | UCX transport selection |
| UCX_NET_DEVICES | unset | Explicit UCX network device selection |
| MX_POOL_REG | 0 | Register each CUDA allocation once instead of each tensor |
| MX_VMM_ARENA | 0 | Allocate load-time weights in a CUDA VMM arena for range registration |
| MX_ARENA_SINGLE_MR | 0 | Keep a single arena memory registration across multiple handles when the transport supports it |
| UCX_CUDA_COPY_REG_WHOLE_ALLOC | UCX default | UCX CUDA-copy registration behavior; relevant to older UCX VMM combinations |
| NIXL_LOG_LEVEL | runtime default | NIXL logging level |
| UCX_LOG_LEVEL | runtime default | UCX logging level |

MX_POOL_REG and MX_VMM_ARENA are alternative registration optimizations. Enable one at a time and qualify the chosen combination on the exact GPU, driver, NIXL, and UCX stack. The older `MX_VMM_ARENA_BYTES` and `MX_VMM_ARENA_CHUNK_BYTES` settings are ignored apart from a deprecation warning.

## ModelStreamer and local-load tuning

| Environment variable | Default | Effect |
|---|---|---|
| RUNAI_STREAMER_CONCURRENCY | 8 | Concurrent storage reads |
| RUNAI_STREAMER_MEMORY_LIMIT | unset | CPU staging buffer size; 0 uses a single-tensor buffer |
| MX_GDS_MAX_CHUNK_KB | unset | GDS chunk-size override |
| MX_GDS_THREADS | 8 | GDS worker threads |
| MX_GDS_TIMEOUT | 120 seconds | GDS operation timeout |
| INSTANTTENSOR_BACKEND | runtime default | InstantTensor backend such as URING, AIO, CUFILE, or MMAP |

ModelStreamer credentials are third-party settings. See [Load from object storage or a local path](guides/choose-a-path.md#load-from-object-storage-or-a-local-path).

## Artifact transfer

| Environment variable | Default | Effect |
|---|---|---|
| MX_ARTIFACT_TRANSFER | false | Enable compatible file-backed JIT artifact installation and publication |
| MX_ARTIFACT_BACKEND | p2p | Artifact transport: `p2p` or `mooncake`; exactly one is selected and there is no fallback |
| MX_ARTIFACT_BUNDLE_ROOT | $TMPDIR/modelexpress-artifacts | Local staging root for artifact bundles |
| MX_ARTIFACT_COMPILE_CONFIG_DIGEST | empty | Partitions torch compile artifact sources by compile configuration |
| MX_ARTIFACT_MOONCAKE_NAMESPACE | modelexpress/artifacts | Key prefix used by the Mooncake artifact backend |
| MX_ARTIFACT_MOONCAKE_POOL_BYTES | 536870912 bytes | Local buffer-pool size for the process-local Mooncake artifact client; a config-file `local_buffer_size` takes precedence |
| MX_ARTIFACT_MOONCAKE_DELETE_RETRIES | 10 | Additional attempts when stale-object or repair-marker removal reports an active lease |
| MX_ARTIFACT_MOONCAKE_DELETE_RETRY_DELAY_SECS | 0.7 | Delay in seconds between Mooncake removal attempts |
| MX_MOONCAKE_CONFIG_PATH | unset | Optional Mooncake JSON configuration file; `MOONCAKE_CONFIG_PATH` is the fallback |
| MX_MC_* | unset | Artifact-store Mooncake settings promoted temporarily to the corresponding native `MC_*` names |
| MX_ETCD_* | unset | Artifact-store etcd settings promoted temporarily to the corresponding native `ETCD_*` names |
| MX_ARTIFACT_READY_URL | framework default | Readiness endpoint checked before publication |
| MX_ARTIFACT_READY_TIMEOUT_SECS | 1800 | Readiness/publication timeout |
| MX_ARTIFACT_TRANSFER_CHUNK_SIZE | 67108864 bytes | Artifact transfer chunk size; maximum is 4 GiB |

The default `p2p` backend requires `MX_P2P_METADATA=1`, a central coordinator,
and NIXL. The `mooncake` backend requires a Python package exposing
`mooncake.store`, plus `MX_MC_METADATA_ADDR` and `MX_MC_MASTER_SERVER` (or the
equivalent JSON/native configuration); it does not use the MX server or
`MX_P2P_METADATA`. Prefixed settings are restored after each artifact-store
operation, allowing a separately initialized Mooncake client such as a KV
cache to keep its own account and configuration.

Both backends require writable target cache directories and a trusted
deployment. Artifact transfer moves file-backed caches; it does not replace
model-weight transfer.

## Client metrics

| Environment variable | Default | Effect |
|---|---|---|
| MX_METRICS_ENABLED | 0 | Enable the Python client Prometheus collector |
| MX_METRICS_PORT | unset | Client pull endpoint port |
| MX_METRICS_PUSHGATEWAY | unset | Pushgateway destination; mutually exclusive with MX_METRICS_PORT |
| MX_METRICS_SCHEME | empty | Run/scheme label for comparisons |
| MX_METRICS_BIND_RETRY_SECS | 15 | Retry interval for endpoint ownership migration |
| MX_METRICS_SOURCE_ID_LABEL | 0 | Restore per-peer source labels for benchmark-only runs |
| PROMETHEUS_MULTIPROC_DIR | unset | Shared pod directory required for multi-process client metrics |

See [Metrics](METRICS.md) for endpoint behavior and PromQL.

## Advanced refit and resharding

Live refit and resharding have additional `MX_REFIT_*` and `MX_RESHARD_*` settings that are intentionally documented with the feature implementation because their defaults and safety contracts are workload-specific. See [the refit README](../modelexpress_client/python/modelexpress/refit/README.md) and the [architecture reference](ARCHITECTURE.md) before enabling them.
