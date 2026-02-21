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

database:
  path: "./models.db"

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

```bash
# With defaults
cargo run --bin modelexpress-server

# With a configuration file
cargo run --bin modelexpress-server -- --config model-express.yaml

# With CLI overrides
cargo run --bin modelexpress-server -- --port 8080 --log-level debug

# Validate config without starting
cargo run --bin modelexpress-server -- --config model-express.yaml --validate-config
```

### Configuration Options

#### Server Settings

| Option | CLI Flag | Env Var | Default | Description |
|--------|----------|---------|---------|-------------|
| host | `--host` | `MODEL_EXPRESS_SERVER_HOST` | `0.0.0.0` | Bind address |
| port | `--port`, `-p` | `MODEL_EXPRESS_SERVER_PORT` | `8001` | gRPC port |

#### Database Settings

| Option | CLI Flag | Env Var | Default | Description |
|--------|----------|---------|---------|-------------|
| path | `--database-path`, `-d` | `MODEL_EXPRESS_DATABASE_PATH` | `./models.db` | SQLite file path |

In multi-node Kubernetes deployments, the database should be on a shared persistent volume.

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
export MODEL_EXPRESS_DATABASE_PATH="/data/models.db"
export MODEL_EXPRESS_CACHE_DIRECTORY="/data/cache"
export MODEL_EXPRESS_CACHE_EVICTION_ENABLED=true
export MODEL_EXPRESS_LOG_LEVEL=debug
export MODEL_EXPRESS_LOG_FORMAT=json
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

Cache directory resolution: `MODEL_EXPRESS_CACHE_DIRECTORY` -> `HF_HUB_CACHE` -> `~/.cache/huggingface/hub`.

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
docker build -f examples/p2p_transfer_k8s/Dockerfile.client \
  -t your-registry/mx-client:TAG .
docker push your-registry/mx-client:TAG
```

## Kubernetes

### Standalone Deployment

Deploy the server with the included manifest:

```bash
kubectl apply -f k8s-deployment.yaml
```

This creates a Deployment with the ModelExpress server and a Redis sidecar on port 8001.

### HuggingFace Token

Most deployments need a HuggingFace token for model downloads:

```bash
export HF_TOKEN=your_hf_token
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

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

### Aggregated Deployment (with Dynamo)

For deploying ModelExpress alongside Dynamo with a vLLM worker:

```bash
kubectl apply -f examples/aggregated_k8s/agg.yaml
```

See [`../examples/aggregated_k8s/README.md`](../examples/aggregated_k8s/README.md) for the full guide.

## P2P GPU Weight Transfers

ModelExpress supports GPU-to-GPU model weight transfers between vLLM instances using NVIDIA NIXL over RDMA. One "source" vLLM instance loads model weights from disk and transfers them directly to "target" instances via GPU memory.

### P2P Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_EXPRESS_URL` | `localhost:8001` | gRPC server address |
| `MX_REGISTER_LOADERS` | `1` | Auto-register mx-source/mx-target loaders with vLLM |
| `MX_CONTIGUOUS_REG` | `0` | Contiguous region registration (experimental) |
| `MX_EXPECTED_WORKERS` | `8` | Number of GPU workers to wait for |
| `MX_SYNC_PUBLISH` | `1` | Source: wait for all workers before publishing |
| `MX_SYNC_START` | `1` | Target: wait for all workers before transferring |
| `VLLM_RPC_TIMEOUT` | `7200000` | vLLM RPC timeout in ms (2 hours for large models) |

### UCX/NIXL Tuning

| Variable | Recommended | Description |
|----------|-------------|-------------|
| `UCX_TLS` | `rc_x,rc,dc_x,dc,cuda_copy` | Transport layers for InfiniBand |
| `UCX_RNDV_SCHEME` | `get_zcopy` | Zero-copy RDMA reads |
| `UCX_RNDV_THRESH` | `0` | Force rendezvous for all transfers |
| `NIXL_LOG_LEVEL` | `INFO` | NIXL logging (DEBUG for troubleshooting) |
| `UCX_LOG_LEVEL` | `WARN` | UCX logging (DEBUG for troubleshooting) |

### P2P Kubernetes Deployment

```bash
NAMESPACE=my-namespace

# Deploy server + Redis
kubectl -n $NAMESPACE apply -f examples/p2p_transfer_k8s/modelexpress-server.yaml

# Deploy source vLLM instance (loads real weights)
kubectl -n $NAMESPACE apply -f examples/p2p_transfer_k8s/vllm-source.yaml

# Deploy target vLLM instance (receives via RDMA)
kubectl -n $NAMESPACE apply -f examples/p2p_transfer_k8s/vllm-target.yaml

# Monitor
watch kubectl -n $NAMESPACE get pods -l 'app in (mx-source, mx-target)'
```

See [`../examples/p2p_transfer_k8s/README.md`](../examples/p2p_transfer_k8s/README.md) for the full P2P transfer guide including architecture, prerequisites, and performance expectations.

## Debugging

```bash
# Stream server logs
kubectl -n $NAMESPACE logs -f deploy/modelexpress-server

# Stream source/target logs
kubectl -n $NAMESPACE logs -f deploy/mx-source
kubectl -n $NAMESPACE logs -f deploy/mx-target

# Check Redis state (P2P metadata)
kubectl -n $NAMESPACE exec deploy/modelexpress-server -c redis -- redis-cli KEYS '*'

# Flush Redis (clear stale metadata - do this on redeploy)
kubectl -n $NAMESPACE exec deploy/modelexpress-server -c redis -- redis-cli FLUSHALL

# Test inference on target
kubectl -n $NAMESPACE exec deploy/mx-target -- curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-ai/DeepSeek-V3", "prompt": "Hello", "max_tokens": 10}'
```

## Performance Reference

| Model | Total Data | Transfer Time | Per-Worker Speed |
|-------|-----------|---------------|------------------|
| DeepSeek-V3 (671B, FP8) | 681 GB (8 GPUs) | ~15 seconds | 60-112 Gbps |
| Llama 3.3 70B | 140 GB (8 GPUs) | ~5 seconds | ~112 Gbps |
