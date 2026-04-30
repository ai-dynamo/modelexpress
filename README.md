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
  <a href="#features">Features</a> •
  <a href="#dynamo-modelexpress-architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#deployment">Deployment</a> •
  <a href="#documentation">Docs</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## Overview

ModelExpress is a Rust-based service that manages the complete model weight lifecycle in the cluster—from acquisition to GPU memory. It accelerates LLM inference by caching, routing, and transferring weights through the fastest available path. Deploy standalone or as a sidecar alongside vLLM, NVIDIA Dynamo, and other inference runtimes.

| LLM serving problem | How ModelExpress helps |
|---------------------|------------------------|
| **Models take too long to load** | GPU-to-GPU transfer via NIXL/RDMA instead of loading from storage. In P2P mode, weights already serving inference act as the cache—no extra storage. |
| **Many nodes need the same model** | Metadata backends (Redis, K8s CRD) coordinate sharing: one node loads; others receive via P2P or local paths. This reduces ingress bandwidth from external providers such as Hugging Face and ensures only one model copy is downloaded even when multiple clients request the same model concurrently. |

### How ModelExpress manages weights in the cluster

ModelExpress orchestrates the full flow—from download to GPU memory. It ensures only one node downloads or streams a model from external sources (for example Hugging Face, NGC, GCS, or object storage through ModelStreamer); other nodes receive weights via P2P or shared storage—eliminating duplicate downloads and reducing cluster ingress.

1. **Download or stream from external storage** — One node pulls or streams the model from Hugging Face, NGC, GCS, or object storage through ModelStreamer; ModelExpress coordinates so no other node duplicates this work. In air-gapped mode, serve from cache only (`HF_HUB_OFFLINE=1`).
2. **Persist to disk** — Store in a cache backed by disk:
   - **Host-attached disk** — Local disk on the node (single-node or per-node cache).
   - **PVC** — RWO (ReadWriteOnce) for single-node; RWX (ReadWriteMany) for shared access across nodes.
3. **Disk to GPU** — Inference engine (vLLM, etc.) loads weights from the cache (disk) into GPU memory.
4. **P2P transfer** — Additional nodes receive weights via GPU-to-GPU RDMA from the first node instead of reading from disk—no duplicate downloads or disk reads.

---

## Features

- **Cold start reduction** — GPU-to-GPU P2P transfer over InfiniBand instead of disk load
- **Model store providers** — built-in providers for Hugging Face, NVIDIA NGC, and Google Cloud Storage
- **ModelStreamer loading** — stream weights from S3, GCS, Azure Blob, local paths, or Hugging Face cache into vLLM with `MX_MODEL_URI`
- **GPUDirect Storage** — direct file-to-GPU loading path when GDS hardware and software are available
- **Cache and path resolution** — PVC-backed cache, `HF_HUB_OFFLINE`, `ignore_weights`, `get_model_path` for Dynamo, and provider-specific cache layouts
- **P2P GPU transfer** — vLLM `mx` loader and TRT-LLM `PRESHARDED` loader with NVIDIA NIXL over RDMA
- **Metadata backends** — Redis or Kubernetes CRD for distributed coordination
- **Kubernetes** — Helm chart, CRDs/Redis for P2P, no-shared-storage support
- **CLI** — Health, download, list, validate, clear; init-container support for pre-warming

### Integrations

| Runtime | Integration |
|---------|-------------|
| vLLM | `--load-format mx` for P2P weight transfer |
| NVIDIA Dynamo (vLLM) | `get_model_path` API; [aggregated K8s example](examples/aggregated_k8s/README.md) |
| TensorRT-LLM | `LoadFormat.PRESHARDED` with `MxLiveCheckpointLoader` for P2P weight transfer (beta) — [TRT-LLM examples](examples/p2p_transfer_k8s/client/trtllm/) |
| SGLang | Coming soon |

### Model Store Providers

ModelExpress supports two storage-access paths:

| Path | Supported sources |
|------|-------------------|
| Model providers | Hugging Face, NVIDIA NGC, Google Cloud Storage |
| ModelStreamer (`MX_MODEL_URI`) | S3 / S3-compatible, GCS, Azure Blob Storage, local filesystem, and Hugging Face cache-resolved model IDs |
| GPUDirect Storage | Local filesystem or cached model files loaded directly to GPU |

---

## ModelExpress Architecture

![ModelExpress Architecture: Upload once, then autoscale new pods via NIXL GPUDirect RDMA from seed GPU](model-express-architecture.png)

*Phase 1 — Upload once:* Model Source (HuggingFace Hub, NFS) downloads to the Seed Pod (GPU), which loads and postprocesses weights, registers VRAM with NIXL, and publishes metadata to the MX Server. *Phase 2 — Autoscale:* New pods receive weights via NIXL GPUDirect RDMA (GPU VRAM → GPU VRAM, zero-copy) from the seed GPU, using `--load-format mx` for inference.

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

- **modelexpress_server**: gRPC server with distributed metadata backends (Redis or Kubernetes CRD)
- **modelexpress_client**: Rust CLI for cache management; Python package with vLLM loaders and `MxClient`
- **modelexpress_common**: Protobuf definitions, provider abstractions, and shared configuration

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
MX_METADATA_BACKEND=redis cargo run --bin modelexpress-server
```

Server listens on `0.0.0.0:8001`. In another terminal:

```bash
# Download a model (shared storage)
modelexpress-cli model download meta-llama/Llama-3.3-70B-Instruct

# Verify
modelexpress-cli health
```

**Without shared storage:** use `--no-shared-storage` for gRPC streaming.  
**Air-gapped:** `HF_HUB_OFFLINE=1 modelexpress-cli model download <model-id>`.

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
# vllm serve <model> --load-format mx --worker-cls=modelexpress.vllm_worker.ModelExpressWorker
```

First instance loads from disk; subsequent instances receive via RDMA. [P2P guide](examples/p2p_transfer_k8s/README.md) · [Server setup](examples/p2p_transfer_k8s/server/README.md).

### Docker

```bash
docker-compose up --build
```

---

## Configuration

**Precedence:** CLI → env vars (`MODEL_EXPRESS_*`, `MX_*`) → YAML → defaults.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_EXPRESS_SERVER_PORT` | `8001` | gRPC port |
| `MODEL_EXPRESS_CACHE_DIRECTORY` | `./cache` | Cache root |
| `MX_METADATA_BACKEND` | (required) | `redis` \| `kubernetes` |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL (`redis` backend only) |
| `MODEL_EXPRESS_URL` | `localhost:8001` | gRPC server (P2P) |
| `MX_MODEL_URI` | (none) | Enable ModelStreamer with `s3://`, `gs://`, `az://`, absolute local paths, or Hugging Face model IDs |
| `UCX_TLS` | `rc_x,rc,dc_x,dc,cuda_copy` | InfiniBand transports |

```bash
cargo run --bin config_gen -- --output model-express.yaml
cargo run --bin modelexpress-server -- --config model-express.yaml --validate-config
```

Full reference: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

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
| [Architecture](docs/ARCHITECTURE.md) | Components, gRPC, NIXL, FP8 |
| [CLI](docs/CLI.md) | Full CLI reference |
| [GCS Provider](docs/GCS_PROVIDER.md) | GCS provider design, cache layout, manifest behavior, and credentials |
| [Metadata](docs/metadata.md) | Redis keys, K8s CRD schema |
| [Helm](helm/README.md) | Kubernetes configuration |

---

## Known Issues

- **MLA models blocked from P2P transfer** — Models using Multi-head Latent Attention (DeepSeek-V2/V3, Kimi K2/K2.5) are automatically blocked from GPU-to-GPU transfer and fall back to disk loading. Bytes transfer correctly but inference produces corrupted output. Set `MX_SKIP_FEATURE_CHECK=1` to bypass for debugging. See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.
- **NIXL_ERR_REMOTE_DISCONNECT** — Source restarts invalidate rkeys. Flush Redis, redeploy.
- **Long source warmup** — DeepSeek-V3 (DeepGemm, CUDA graphs) can take significant time; targets wait via coordination.
- **Large model gRPC stream** — May not close automatically; use client timeout.
- **MX_CONTIGUOUS_REG=1** — Blocked; use `0`.

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
