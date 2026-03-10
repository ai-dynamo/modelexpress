<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

<img src="ModelExpressTrainLogo.jpeg" alt="ModelExpress Logo" width="50%">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# ModelExpress

ModelExpress is a model weight management service for large language model (LLM) inference deployments. It reduces cold-start latency and operational cost by caching, routing, and transferring weights through the fastest available path—from external storage to live GPU memory via peer-to-peer RDMA.

## Overview

ModelExpress deploys standalone or as a sidecar alongside inference runtimes such as vLLM, SGLang, and NVIDIA Dynamo. It addresses two core challenges in production LLM serving:

| Challenge | Solution |
|-----------|----------|
| **Cold start latency** | Local weight caching with GPU-to-GPU transfer in seconds via NIXL/RDMA, avoiding external downloads. In P2P mode, GPU-resident weights serving inference act as the cache with no additional storage and minimal impact on latency. |
| **Multi-node coordination** | Metadata management coordinates weight sharing: a single node downloads or loads; others receive via local or P2P paths. |

### Capabilities

| Feature | Description |
|---------|-------------|
| **HuggingFace caching** | PVC-backed cache with `HF_HUB_OFFLINE` for air-gapped environments, handling of empty files and dotfiles, `ignore_weights` for config-only download, and `get_model_path` API for Dynamo. |
| **Shared storage mode** | Client and server share a network volume; inference engines read directly from the cache. |
| **Distributed (gRPC) mode** | File transfer over gRPC when shared storage is unavailable (`--no-shared-storage`). |
| **GPU-to-GPU P2P transfer** | vLLM loaders (`mx-source`, `mx-target`) with NVIDIA NIXL over RDMA/InfiniBand; raw tensor transfer before FP8 processing. GPU-loaded weights serving inference act as the cache—no additional storage required, minimal impact on latency. |
| **Metadata backends** | In-memory (default), Redis, or Kubernetes CRD; layered (in-memory with write-through) for high availability. |
| **Kubernetes** | Helm chart with configurable persistence, metadata via CRDs or Redis, no-shared-storage deployments, ephemeral storage limits, and secure defaults. |
| **CLI** | `modelexpress-cli` for health, download, list, validate, clear, and API operations; init-container support for cache pre-warming. |

---

## Architecture

ModelExpress manages the full lifecycle of model weights from acquisition to GPU memory. The diagram below illustrates how ModelExpress accelerates vLLM bootup by using GPU-to-GPU RDMA transfer instead of traditional disk-to-GPU loading—reducing weight transfer from ~25 minutes to ~15 seconds (~100× faster) and overall bootup from ~38 minutes to ~16 minutes.

![ModelExpress Architecture: Accelerating vLLM Bootup](model-express-architecture.png)

- **`modelexpress_server`**: gRPC server with configurable metadata backends (in-memory, Redis, Kubernetes CRD). Layered in-memory with persistence is recommended for high availability.
- **`modelexpress_client`**: Rust CLI for cache management; Python package with vLLM loaders (`MxSourceModelLoader`, `MxTargetModelLoader`) and `MxClient` for gRPC.
- **`modelexpress_common`**: Protobuf definitions, provider trait (HuggingFace), shared configuration.

---

## Quick Start

### Prerequisites

- Rust 1.90+ (or use the container image)
- `protoc` (Protocol Buffers compiler)
- Docker (optional, for containerized deployment)
- NVIDIA GPU + InfiniBand (optional, for P2P transfers)

### Build and Run

```bash
git clone https://github.com/ai-dynamo/modelexpress.git
cd modelexpress

cargo build
cargo run --bin modelexpress-server
```

The server listens on `0.0.0.0:8001` by default. See [Configuration](#configuration) for customization.

### Download a Model

```bash
# With shared storage (default)
modelexpress-cli model download meta-llama/Llama-3.3-70B-Instruct

# Without shared storage (stream over gRPC)
modelexpress-cli --no-shared-storage model download meta-llama/Llama-3.3-70B-Instruct

# Air-gapped: serve from cache only
HF_HUB_OFFLINE=1 modelexpress-cli model get meta-llama/Llama-3.3-70B-Instruct
```

---

## Deployment

### Kubernetes (Helm)

ModelExpress is production-ready for Kubernetes:

```bash
# Create HuggingFace token secret (required for model downloads)
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n <your-namespace>

helm install modelexpress ./helm --namespace modelexpress --create-namespace
```

- **Helm chart**: Configurable persistence, resource limits, health probes
- **Metadata management**: K8s CRDs and Redis for multi-node P2P coordination
- **No-shared-storage**: Models stream over gRPC when a shared volume is not available

See [helm/README.md](helm/README.md) for configuration and production overrides.

### P2P GPU Transfer (vLLM)

GPU-to-GPU weight transfer between vLLM instances using NVIDIA NIXL over RDMA/InfiniBand. The source GPU's loaded weights—already serving inference—act as the cache for P2P transfer with no additional storage and minimal latency impact.

1. Deploy the ModelExpress server (gRPC; metadata backend: memory, Redis, or Kubernetes CRD).
2. **Source**: Set `VLLM_LOAD_FORMAT=mx-source` — loads from disk, registers raw tensors with NIXL before FP8 processing, publishes metadata.
3. **Target**: Set `VLLM_LOAD_FORMAT=mx-target` — receives raw tensors via RDMA, performs FP8 processing locally.

Register loaders in Python:

```python
from modelexpress import register_modelexpress_loaders
register_modelexpress_loaders()
```

See [examples/p2p_transfer_k8s/README.md](examples/p2p_transfer_k8s/README.md) for deployment and [examples/p2p_transfer_k8s/deploy/persistence/README.md](examples/p2p_transfer_k8s/deploy/persistence/README.md) for metadata persistence (HA).

### Docker

```bash
docker-compose up --build
# Or: docker run -p 8001:8001 -v /path/to/cache:/cache model-express
```

---

## Configuration

ModelExpress uses layered configuration with the following precedence (highest to lowest):

- Command line arguments
- Environment variables (`MODEL_EXPRESS_*`)
- YAML config file
- Default values

### Example Configuration

```bash
cargo run --bin config_gen -- --output model-express.yaml
cargo run --bin modelexpress-server -- --config model-express.yaml

# Validate config without starting the server
cargo run --bin modelexpress-server -- --config model-express.yaml --validate-config
```

### Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_EXPRESS_SERVER_HOST` | `0.0.0.0` | Bind address |
| `MODEL_EXPRESS_SERVER_PORT` | `8001` | gRPC port |
| `MODEL_EXPRESS_CACHE_DIRECTORY` | `./cache` | Cache root |
| `HF_HUB_OFFLINE` | — | Set to `1` for air-gapped (serve from cache only) |

### Metadata Backend (P2P)

| Variable | Values | Description |
|----------|--------|-------------|
| `MX_METADATA_BACKEND` | `memory`, `redis`, `kubernetes`/`k8s` | In-memory (default); Redis or Kubernetes CRD for persistence. Layered in-memory with write-through for `redis` and `kubernetes`. |
| `REDIS_URL` / `MX_REDIS_HOST` | — | When backend is Redis |
| `MX_METADATA_NAMESPACE` / `POD_NAMESPACE` | — | When backend is Kubernetes |

### P2P Transfer Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_EXPRESS_URL` | `localhost:8001` | gRPC server address (also reads `MX_SERVER_ADDRESS` for compat) |
| `MX_EXPECTED_WORKERS` | `8` | GPU workers to wait for before transfer |
| `MX_REGISTER_LOADERS` | `1` | Auto-register `mx-source`/`mx-target` loaders with vLLM |
| `MX_SYNC_PUBLISH` | `1` | Source: wait for all workers before publishing metadata |
| `MX_SYNC_START` | `1` | Target: wait for all workers before starting transfer |
| `MX_CONTIGUOUS_REG` | `0` | Enable contiguous region registration (experimental, currently blocked) |
| `VLLM_RPC_TIMEOUT` | `7200000` | vLLM RPC timeout in ms (2 hours for large models) |

### UCX/NIXL Tuning

| Variable | Recommended | Description |
|----------|-------------|-------------|
| `UCX_TLS` | `rc_x,rc,dc_x,dc,cuda_copy` | Transport layers for InfiniBand + CUDA copy |
| `UCX_RNDV_SCHEME` | `get_zcopy` | Zero-copy RDMA reads |
| `UCX_RNDV_THRESH` | `0` | Force rendezvous protocol for all transfers |
| `NIXL_LOG_LEVEL` | `INFO` | NIXL logging (`DEBUG` for troubleshooting) |
| `UCX_LOG_LEVEL` | `WARN` | UCX logging (`DEBUG` for troubleshooting) |

---

## CLI Reference

See [docs/CLI.md](docs/CLI.md) for full documentation.

```bash
modelexpress-cli health
modelexpress-cli model download <model-id>
modelexpress-cli model list
modelexpress-cli model validate <model-id>
modelexpress-cli model clear <model-id>
```

---

## Project Structure

```
modelexpress/
├── modelexpress_server/     # gRPC server, Redis metadata
├── modelexpress_client/     # Rust CLI + Python vLLM loaders
├── modelexpress_common/     # Protobuf, shared types
├── examples/                # Kubernetes deployments (P2P, aggregated)
├── helm/                    # Helm chart
├── docs/                    # CLI, metadata specs
├── workspace-tests/         # Integration tests
├── docker-compose.yml
├── Dockerfile
└── k8s-deployment.yaml
```

---

## Testing

```bash
cargo test
cargo test --test integration_tests
cargo run --bin test_client -- --test-model "google-t5/t5-small"
./run_integration_tests.sh

# Benchmarks
cargo bench

# Coverage (requires cargo-tarpaulin)
cargo tarpaulin --out Html
```

## Development

Set up pre-commit hooks to enforce formatting and linting before each commit:

```bash
pip install pre-commit
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

Hooks run `cargo fmt`, `cargo clippy`, `cargo check`, and file hygiene checks. Run them early and often rather than waiting until commit time.

---

## Known Issues

- **NIXL_ERR_REMOTE_DISCONNECT**: If the source pod restarts during transfer, stale rkeys can cause remote access errors. Flush Redis and redeploy both source and target pods.
- **Long source warmup (DeepSeek-V3)**: Source warmup (DeepGemm compilation, CUDA graph capture) can take significant time; targets coordinate via Redis and wait before starting transfer.
- **Large model gRPC stream**: For very large models in distributed mode, the gRPC stream may not close automatically. Use client-side timeout or file-based completion checks.
- **Contiguous region transfers** (`MX_CONTIGUOUS_REG=1`): Currently blocked — transfers fail with remote access errors even when the source is stable. Use baseline mode (`MX_CONTIGUOUS_REG=0`) until resolved.

---

## Roadmap

### Priorities Under Development

- **P2P compile/warmup caching**: Implementing torch.compile/deepGEMM cache strategy for follower workers. Operational model: leader performs full engine warmup; followers consume caches and skip full warmup.
- **ModelStreamer Integration**: Integrating Model Streamer for pulling weights from cold storage, inheriting multi-cloud and multi-engine support.
- **DRAM and NVMe-resident shard streaming**: Stream shards across workers while keeping weights in DRAM and host local high speed NVMe.
- **RL workloads**: Exploring ways to utilize fast P2P transfers to optimize refit phase of RL and adding support for weight resharding.
- **Earlier weight availability**: Bring weights to prefill earlier in the lifecycle and identify prefill workers that can act as strong source nodes.
- **Expanded model pull providers**: Support NGC in addition to Hugging Face.
- **GDS (GPUDirect Storage) integration**: Load model weights directly from NVMe into GPU memory, bypassing the CPU/DRAM copy path entirely.
- **Multi-tier cache hierarchy**: Automatically promote and demote models across DRAM, NVMe, and PVC tiers based on access patterns and available capacity.
- **Distributed sharded cache**: Shard large models across nodes using consistent hashing with parallel shard assembly, reducing per-node storage requirements for very large models.
- **Training checkpoint management**: Cache and reuse CUDA kernel compilations (torch.compile, deepGEMM) and CUDA graphs across restarts, dramatically reducing cold-start overhead.
- **Metrics and observability**: Expose cache hit rates, eviction frequency, transfer throughput, and P2P RDMA utilization via Prometheus/OpenTelemetry.
- **Predictive prefetching**: Pre-warm caches based on workload history or scheduling hints to reduce model load latency before a request arrives.
- **P2P transfer fault tolerance**: Automatic recovery from stale rkeys on source restart, with retry logic and fallback to storage-based loading for production reliability.
- **Multi-cloud storage backends**: Native support for AWS S3, Azure Blob, and NFS as model pull sources.

Contributions and feedback are welcome. See [Contributing](#contributing) or open an [issue](https://github.com/ai-dynamo/modelexpress/issues).

---

## Documentation

| Resource | Description |
|----------|-------------|
| [CLI Reference](docs/CLI.md) | Full `modelexpress-cli` documentation |
| [Helm Chart](helm/README.md) | Kubernetes deployment and configuration |
| [P2P Transfer (vLLM)](examples/p2p_transfer_k8s/README.md) | GPU-to-GPU transfer setup |
| [Persistence Backends](examples/p2p_transfer_k8s/deploy/persistence/README.md) | Redis and Kubernetes CRD metadata persistence for HA |
| [Metadata Spec](docs/metadata.md) | NIXL metadata, Redis keys, K8s CRD schema |
| [Dynamo + K8s](examples/aggregated_k8s/README.md) | ModelExpress with Dynamo on Kubernetes |

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/ai-dynamo/modelexpress/issues)
- **Examples**: `workspace-tests/` and `examples/` for integration and deployment walkthroughs

---

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
