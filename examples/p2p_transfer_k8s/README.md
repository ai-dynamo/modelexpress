# P2P GPU Transfer Example

This example demonstrates how to set up ModelExpress for P2P GPU weight transfers between vLLM instances on Kubernetes. For the broader deployment guide and environment variable reference, see [`docs/DEPLOYMENT.md`](../../docs/DEPLOYMENT.md). For NIXL architecture details, see [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Architecture

```mermaid
graph TD
    subgraph "Node A"
        A[vLLM + MxModelLoader<br/>- Loads weights from disk<br/>- Registers tensors with NIXL<br/>- PublishMetadata + UpdateStatus]
    end
    subgraph "Node B"
        B[vLLM + MxModelLoader<br/>- GetMetadata, checks worker status<br/>- Receives weights via NIXL<br/>- Runs FP8 processing]
    end
    A -- "RDMA via NIXL" --> B
    A --> S
    B --> S
    S[ModelExpress Server - gRPC<br/>PublishMetadata / GetMetadata / UpdateStatus]
```

### Key Design Points

1. **Custom vLLM Loader**: NIXL transfer logic runs inside vLLM via `--load-format mx`
2. **MxClient**: All gRPC communication goes through `MxClient` (workers never access Redis directly)
3. **FP8 Support**: Raw tensors (including `weight_scale_inv`) transfer BEFORE FP8 processing
4. **Tensor Parallelism**: Full TP support with rank-matched transfers (one NIXL agent per GPU)

## Prerequisites

1. Kubernetes cluster with GPU nodes (H100/H200 recommended)
2. InfiniBand network between nodes for optimal performance
3. NVIDIA drivers and CUDA toolkit
4. Container images with NIXL and ZMQ support

## Deployment

### 1. Create HuggingFace Token Secret

```bash
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=<your-token>
```

### 2. Choose a Metadata Backend and Deploy the Server

See [`server/`](server/) for backend-specific server manifests:
- **Redis**: [`server/redis_backend/`](server/redis_backend/)
- **Kubernetes CRD**: [`server/kubernetes_backend/`](server/kubernetes_backend/)

### 3. Deploy vLLM Clients

See [`client/vllm/`](client/vllm/) for vLLM deployment manifests:
- **Single-node** (TP only): [`client/vllm/vllm-single-node.yaml`](client/vllm/vllm-single-node.yaml)
- **Multi-node** (TP + PP): [`client/vllm/vllm-multi-node.yaml`](client/vllm/vllm-multi-node.yaml)

The `mx` loader checks the MX server on startup. If a ready source exists, it receives via RDMA. Otherwise it loads from disk and becomes a source for future nodes.

## Environment Variables

### vLLM Container

| Variable | Description | Example |
|----------|-------------|---------|
| `MX_ZMQ_ADDRESS` | Base ZMQ address (rank appended) | `ipc:///tmp/mx/vllm.sock` |

With TP=4, this creates sockets: `/tmp/mx/vllm-0.sock`, `/tmp/mx/vllm-1.sock`, etc.

### ModelExpress Server

| Variable | Description | Default |
|----------|-------------|---------|
| `MX_METADATA_BACKEND` | Backend type: `redis` or `kubernetes` | (required) |
| `REDIS_URL` | Redis connection URL (redis backend) | `redis://localhost:6379` |

## Performance Expectations

| Model | Size | TP | Transfer Time | Throughput |
|-------|------|-----|---------------|------------|
| Llama 3.1 70B | ~140GB | 4 | ~4-5s | ~300 Gbps |
| Llama 3.1 405B | ~750GB | 8 | ~25s | ~300 Gbps |

## Troubleshooting

### Check ZMQ sockets are created

```bash
kubectl exec -it deployment/mx-vllm -c vllm -- ls -la /tmp/mx/
```

### Check vLLM logs

```bash
kubectl logs deployment/mx-vllm -c vllm
```

### Check Redis connectivity (Redis backend)

```bash
kubectl exec -it deployment/modelexpress-server -c redis -- redis-cli ping
```

### Verify InfiniBand is working

```bash
kubectl exec -it deployment/mx-vllm -c vllm -- ibstat
```

### Check UCX configuration

```bash
kubectl exec -it deployment/mx-vllm -c vllm -- ucx_info -d
```

## License

Apache-2.0
