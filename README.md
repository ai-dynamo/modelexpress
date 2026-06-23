<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

<p align="center">
  <img src="ModelExpressTrainLogo.jpeg" alt="ModelExpress Logo" width="50%">
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/rust-1.90%2B-orange" alt="Rust"></a>
</p>

<h1 align="center">Dynamo ModelExpress</h1>

<p align="center">
  <strong>Model weight management for LLM inference</strong> — cache, transfer, and serve weights at scale with GPU-to-GPU RDMA and multi-node coordination.
</p>

<p align="center">
  <a href="#start-here">Start Here</a> •
  <a href="docs/COMPATIBILITY.md">Compatibility</a> •
  <a href="#features">Features</a> •
  <a href="#modelexpress-architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#deployment">Deployment</a> •
  <a href="#documentation">Docs</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## Overview

ModelExpress (MX) is a Rust-based service that manages the complete model weight lifecycle in a GPU cluster — from acquisition to GPU memory. It accelerates the startup times of new LLM inference deployments by caching, routing, and transferring model weights through the fastest available path(s). 

ModelExpress works beyond NVIDIA Dynamo. You can run it standalone as a model-weight service, integrate it directly with stock inference runtimes such as vLLM and  SGLang, or let higher-level systems such as Dynamo, llm-d, Prime-RL, and similar orchestration layers use MX as the weight lifecycle and transfer substrate.

| LLM serving problem | How ModelExpress helps |
|---------------------|------------------------|
| **Models take too long to load** | GPU-to-GPU transfer via NIXL/RDMA instead of loading from storage. In P2P mode, weights already serving inference act as the cache—no extra storage. |
| **Many nodes need the same model** | Metadata backends (Redis, K8s CRD) coordinate sharing: one node loads; others receive via P2P or local paths. |

### How ModelExpress manages weights in the cluster

ModelExpress orchestrates the full flow—from download to GPU memory. It ensures only one node downloads a model from external sources (e.g., HuggingFace); other nodes receive weights via P2P or shared storage—eliminating duplicate downloads and reducing cluster ingress.

1. **Download from HuggingFace/S3** — One node pulls the model; ModelExpress coordinates so no other node duplicates this download, reducing external ingress. In air-gapped mode, serve from cache only (`HF_HUB_OFFLINE=1`).
2. **Persist to disk** — Store in a cache backed by disk:
   - **Host-attached disk** — Local disk on the node (single-node or per-node cache).
   - **PVC** — RWO (ReadWriteOnce) for single-node; RWX (ReadWriteMany) for shared access across nodes.
3. **Disk to GPU** — Inference engine (vLLM, etc.) loads weights from the cache (disk) into GPU memory.
4. **P2P transfer** — Additional nodes receive weights via GPU-to-GPU RDMA from the first node instead of reading from disk—no duplicate downloads or disk reads.

---

## Start Here

New to ModelExpress? Follow this path:

1. **Run the server and CLI locally:** [Quick Start](#quick-start) gets a local server running and verifies it with `modelexpress-cli health`. No GPU required.
2. **Check what is supported:** [Compatibility](docs/COMPATIBILITY.md) lists tested engines, image pins, and beta/experimental paths.
3. **Pick one deployment path:**

| If your goal is... | Start here |
|--------------------|------------|
| Faster vLLM replica startup with P2P RDMA | [Kubernetes P2P examples](examples/p2p_transfer_k8s/README.md) |
| Loading weights directly from S3, Azure Blob, GCS, or PVC | [ModelStreamer examples](examples/model_streamer_k8s/README.md) |
| Using ModelExpress inside Dynamo | [Dynamo model cache](examples/dynamo_model_cache_k8s/README.md) or [Dynamo P2P](examples/dynamo_p2p_transfer_k8s/README.md) |
| Using SGLang | [SGLang guide](docs/SGLANG.md) |
| Evaluating TensorRT-LLM P2P | [TRT-LLM examples](examples/p2p_transfer_k8s/client/trtllm/) |

For Kubernetes deployments, choose a metadata backend before rollout: see the [deployment backend guide](docs/DEPLOYMENT.md#choosing-a-metadata-backend).

---

## Features

- **Cold start reduction** — GPU-to-GPU P2P transfer over InfiniBand instead of disk load
- **HuggingFace caching** — PVC-backed cache, `HF_HUB_OFFLINE`, `ignore_weights`, `get_model_path` for Dynamo
- **P2P GPU transfer** — vLLM `modelexpress` loader (`mx` alias), SGLang `remote_instance` backend, and TRT-LLM `PRESHARDED` beta path
- **Metadata backends** — Redis or Kubernetes CRD for coordinated deployments; feature-gated in-memory backend for local development and tests
- **Kubernetes** — Helm chart, CRDs/Redis for P2P, no-shared-storage support
- **CLI** — Health, download, list, validate, clear; init-container support for pre-warming
- **ModelStreamer integration**: stream weights from object storage (AWS S3, Azure Blob, GCS), with vLLM support today and additional engine coverage evolving
- **Expanded model pull providers**: NGC catalog and Google Cloud Storage in addition to Hugging Face
- **GDS (GPUDirect Storage)**: hardware-dependent path for loading model weights directly from NVMe into GPU memory

### Integrations

| Runtime | Integration |
|---------|-------------|
| vLLM | `--load-format modelexpress` for P2P weight transfer; `mx` is a backward-compatible alias |
| NVIDIA Dynamo (vLLM) | `get_model_path` API; [Dynamo model cache K8s example](examples/dynamo_model_cache_k8s/README.md) |
| TensorRT-LLM | `LoadFormat.PRESHARDED` with `MxLiveCheckpointLoader` for P2P weight transfer (beta) — [TRT-LLM examples](examples/p2p_transfer_k8s/client/trtllm/) |
| SGLang | `remote_instance` + `modelexpress` backend with `transport=nixl` or `transport=transfer_engine` — see [`docs/SGLANG.md`](docs/SGLANG.md) |

---

## ModelExpress Architecture

![ModelExpress Architecture: Upload once, then autoscale new pods via NIXL GPUDirect RDMA from seed GPU](model-express-architecture.png)

*Phase 1 — Upload once:* Model Source (HuggingFace Hub, NFS) downloads to the Seed Pod (GPU), which loads and postprocesses weights, registers VRAM with NIXL, and publishes metadata to the MX Server. *Phase 2 — Autoscale:* New pods receive weights via NIXL GPUDirect RDMA (GPU VRAM → GPU VRAM, zero-copy) from the seed GPU, using `--load-format modelexpress` for inference.

```
                    ┌─────────────────────────────────────────────────────────────────┐
                    │                    ModelExpress Server                          │
                    │   Health • Model • P2P Metadata • Redis/K8s CRD backends        │
                    └──────────────────────┬──────────────────────────────────────────┘
                                           │
                         ┌─────────────────┼─────────────────┐
                         │ metadata        │                 │ metadata
                         ▼                 │                 ▼
              ┌──────────────────┐         │       ┌──────────────────┐
              │  Source (vLLM)   │  RDMA   │       │  Target (vLLM)   │
              │  mx loader       │════════►│       │  mx loader       │
              │  Load → NIXL     │  NIXL   │       │  Receive → FP8   │
              │  Publish metadata│         │       │  Serve inference │
              └──────────────────┘         │       └──────────────────┘
```

*Source and Target exchange metadata with the server for coordination; weights transfer directly over RDMA between GPUs.*

- **modelexpress_server**: gRPC server with configurable metadata backends (Redis, Kubernetes CRD).
- **modelexpress_client**: Rust CLI for cache management; Python package with inference engine loaders and `MxClient` for gRPC.
- **modelexpress_common**: Protobuf definitions, provider trait (HuggingFace), shared configuration.

See [Architecture](docs/ARCHITECTURE.md).

---

## Quick Start

**Requirements:** Rust 1.90+, `protoc`, Docker

```bash
git clone https://github.com/ai-dynamo/modelexpress.git
cd modelexpress

# Start a local Redis instance for metadata storage
docker run -d --name redis -p 6379:6379 redis:8-alpine

cargo build
# REDIS_URL is required; the server does not fall back to localhost:6379.
REDIS_URL=redis://localhost:6379 MX_METADATA_BACKEND=redis cargo run --bin modelexpress-server
```

Server listens on `0.0.0.0:8001`. In another terminal:

```bash
# Download a model (shared storage)
modelexpress-cli model download meta-llama/Llama-3.3-70B-Instruct

# Verify
modelexpress-cli health
```

**Without shared storage:** use `--no-shared-storage` for gRPC streaming.  
**Air-gapped:** with the model already in the local HF cache, `HF_HUB_OFFLINE=1 modelexpress-cli model download <model-id>` resolves it without network access.

---

## Deployment

### Kubernetes (Helm)

```bash
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=${HF_TOKEN} -n <namespace>
helm install modelexpress ./helm --namespace modelexpress --create-namespace
```

Override [values-production.yaml](helm/values-production.yaml) for your env. Full config: [helm/README.md](helm/README.md).

### P2P GPU Transfer (vLLM)

```python
from modelexpress import register_modelexpress_loaders
register_modelexpress_loaders()
# vllm serve <model> --load-format modelexpress
# The mx load format is kept as a backward-compatible alias.
```

First instance loads from disk; subsequent instances receive via RDMA. [P2P guide](examples/p2p_transfer_k8s/README.md) · [Server setup](examples/p2p_transfer_k8s/server/README.md).

### ModelStreamer on Kubernetes

Load model weights directly from Azure Blob Storage, S3, or a PVC-backed local path through ModelStreamer. [ModelStreamer examples](examples/model_streamer_k8s/README.md) · [vLLM recipes](examples/model_streamer_k8s/client/vllm/README.md).

### Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

---

## Configuration

**Precedence:** CLI → env vars (`MODEL_EXPRESS_*`, `MX_*`) → YAML → defaults.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_EXPRESS_SERVER_PORT` | `8001` | gRPC port |
| `MODEL_EXPRESS_CACHE_DIRECTORY` | `./cache` | Cache root |
| `MX_METADATA_BACKEND` | (required) | `redis` \| `kubernetes` |
| `REDIS_URL` | (required for `redis`) | Redis connection URL. Alternatively set `MX_REDIS_HOST` + `MX_REDIS_PORT`. No localhost fallback. |
| `MX_SERVER_ADDRESS` | `localhost:8001` | Client-side gRPC server address (P2P). Recommended. |
| `MODEL_EXPRESS_URL` | `localhost:8001` | Deprecated, pending removal in a future release. Still read by all client paths and takes precedence when both are set; keep setting it during the transition. |

```bash
cargo run --bin config_gen -- --output model-express.yaml
cargo run --bin modelexpress-server -- --config model-express.yaml --validate-config
```

Full reference: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

## CLI

```bash
modelexpress-cli health
modelexpress-cli model download <model-id>
modelexpress-cli model list
modelexpress-cli model validate <model-id>
modelexpress-cli model clear <model-id>
```

[CLI Reference](docs/CLI.md)

---

## Testing

```bash
cargo test
cargo test --test integration_tests
cargo run --bin test_client -- --test-model "google-t5/t5-small"
./run_integration_tests.sh
cargo bench
```

---

## Documentation

| Doc | Description |
|-----|-------------|
| [Deployment](docs/DEPLOYMENT.md) | Server/client config, Docker, K8s, P2P |
| [Compatibility](docs/COMPATIBILITY.md) | Tested runtime and platform pins |
| [Architecture](docs/ARCHITECTURE.md) | Components, gRPC, NIXL, FP8 |
| [CLI](docs/CLI.md) | Full CLI reference |
| [Metadata](docs/metadata.md) | Redis keys, K8s CRD schema |
| [Helm](helm/README.md) | Kubernetes configuration |

---

## Known Issues

- **NIXL_ERR_REMOTE_DISCONNECT** — Source restarts invalidate rkeys. Flush Redis, redeploy.
- **Long source warmup** — DeepSeek-V3 (DeepGemm, CUDA graphs) can take significant time; targets wait via coordination.
- **Large model gRPC stream** — May not close automatically; use client timeout.

---

## Roadmap

### Priorities Under Development

- **P2P compile/warmup caching**: torch.compile/deepGEMM cache for follower workers. Leader performs full warmup; followers consume caches and skip full warmup.
- **DRAM and NVMe-resident shard streaming**: Stream shards across workers while keeping weights in DRAM and host local high-speed NVMe.
- **RL workloads**: Explore fast P2P transfers to optimize RL refit phase and support for weight resharding.
- **Earlier weight availability**: Bring weights to prefill earlier; identify prefill workers that can act as strong source nodes.
- **Multi-tier cache hierarchy**: Promote and demote models across DRAM, NVMe, and PVC tiers based on access patterns.
- **Distributed sharded cache**: Shard large models across nodes using consistent hashing and parallel shard assembly.
- **Training checkpoint management**: Cache and reuse CUDA kernel compilations (torch.compile, deepGEMM) and CUDA graphs across restarts.
- **Metrics and observability**: Cache hit rates, eviction frequency, transfer throughput, and P2P RDMA utilization via Prometheus/OpenTelemetry.
- **Predictive prefetching**: Pre-warm caches from workload history or scheduling hints.
- **P2P transfer fault tolerance**: Auto-recovery from stale rkeys on source restart; retry and fallback to storage loading.
- **Dynamic EPLB (Expert Parallelism Load Balancer)**: Rebalance MoE expert placement across GPUs at runtime via P2P transfer of expert weights as load shifts.

---

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
pip install pre-commit && pre-commit install
pre-commit run --all-files
```

**Issues:** [GitHub Issues](https://github.com/ai-dynamo/modelexpress/issues)

---

## License

Apache 2.0. See [LICENSE](LICENSE).
