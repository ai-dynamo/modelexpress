# Container Images

## MX Server (`Dockerfile.server`)

Builds the ModelExpress gRPC server (Rust) with CLI tools.

```bash
docker build -f container/Dockerfile.server -t modelexpress-server .
```

The image includes:
- `modelexpress-server` -- gRPC coordinator with Redis-backed metadata storage
- `modelexpress-cli` -- CLI for querying server state
- `test_client` / `fallback_test` -- integration test utilities

Exposes port 8001. Requires a Redis sidecar (see `gms_mx/modelexpress-server.yaml` for K8s deployment with Redis).

## MX Client (`Dockerfile.client`)

Builds the ModelExpress Python client with vLLM for weight loading.

```bash
# Base client (RDMA P2P only)
docker build -f container/Dockerfile.client -t modelexpress-client .

# With GMS support (adds gpu-memory-service)
docker build -f container/Dockerfile.client \
    --build-arg ENABLE_GMS=true \
    -t modelexpress-client:gms .
```

### Build Args

| Arg | Default | Description |
|-----|---------|-------------|
| `ENABLE_GMS` | `false` | Install `gpu-memory-service` for GMS integration |
| `VLLM_VERSION` | `0.15.1` | vLLM version to install |
| `CUDA_VERSION` | `12.9.1` | CUDA toolkit version |
| `UBUNTU_VERSION` | `ubuntu24.04` | Ubuntu base version |

### Layer Caching

The Dockerfile is structured for efficient rebuilds:

1. **CUDA base + system packages** -- rarely changes
2. **vLLM + PyTorch** -- changes only when `VLLM_VERSION` changes
3. **ModelExpress dependencies** -- changes only when `pyproject.toml` changes
4. **ModelExpress source** -- changes on every code edit (fast: `pip install --no-deps`)
5. **vLLM loader patch** -- changes rarely

Changing GMS Python code only rebuilds step 4 (seconds).
