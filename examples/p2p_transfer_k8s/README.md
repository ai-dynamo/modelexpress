# P2P GPU Transfer Example

This example demonstrates how to set up ModelExpress for P2P GPU weight transfers between vLLM instances on Kubernetes.

## Architecture

```
Node A (Source)                               Node B (Target)
+----------------------------------+          +----------------------------------+
| vLLM + MxSourceModelLoader       |          | vLLM + MxTargetModelLoader       |
| - Loads weights from disk        |          | - Starts with dummy weights      |
| - Registers tensors with NIXL    |          | - Waits for source ready flag    |
| - Publishes metadata via MxClient|   RDMA   | - Receives weights via NIXL      |
| - Publishes ready flag           |=========>| - Runs FP8 processing            |
+----------------------------------+  NIXL    | - Serves inference               |
        |                                     +----------------------------------+
        |                                             |
        v                                             v
+--------------------------------------------------------------------+
|                    ModelExpress Server (gRPC + Redis)               |
|   - PublishMetadata / GetMetadata: tensor metadata coordination     |
|   - PublishReady / GetReady: source readiness coordination         |
+--------------------------------------------------------------------+
```

### Key Design Points

1. **Custom vLLM Loaders**: NIXL transfer logic runs inside vLLM via `--load-format mx-source` / `--load-format mx-target`
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

### 2. Deploy ModelExpress Server

```bash
kubectl apply -f modelexpress-server.yaml
```

### 3. Deploy Source vLLM Instance

This instance loads real weights from HuggingFace and becomes the source:

```bash
kubectl apply -f vllm-source.yaml
```

Wait for it to be ready and client to publish metadata:

```bash
kubectl logs deployment/mx-source -c client -f
```

### 4. Deploy Target vLLM Instance

This instance starts with dummy weights and receives real weights via P2P:

```bash
kubectl apply -f vllm-target.yaml
```

The client will automatically:
1. Wait for vLLM ZMQ sockets to be ready
2. Query ModelExpress server for the model
3. Find source metadata and receive weights via NIXL RDMA
4. Publish its own metadata (becomes another source)

```bash
kubectl logs deployment/mx-target -c client -f
```

## Environment Variables

### vLLM Container

| Variable | Description | Example |
|----------|-------------|---------|
| `MX_ZMQ_ADDRESS` | Base ZMQ address (rank appended) | `ipc:///tmp/mx/vllm.sock` |

With TP=4, this creates sockets: `/tmp/mx/vllm-0.sock`, `/tmp/mx/vllm-1.sock`, etc.

### ModelExpress Server

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |

## Performance Expectations

| Model | Size | TP | Transfer Time | Throughput |
|-------|------|-----|---------------|------------|
| Llama 3.1 70B | ~140GB | 4 | ~4-5s | ~300 Gbps |
| Llama 3.1 405B | ~750GB | 8 | ~25s | ~300 Gbps |

## Troubleshooting

### Check ZMQ sockets are created

```bash
kubectl exec -it deployment/mx-source -c vllm -- ls -la /tmp/mx/
```

### Check client logs

```bash
kubectl logs deployment/mx-source -c client
kubectl logs deployment/mx-target -c client
```

### Check Redis connectivity

```bash
kubectl exec -it deployment/modelexpress-server -- redis-cli -h modelexpress-redis ping
```

### Verify InfiniBand is working

```bash
kubectl exec -it deployment/mx-source -c client -- ibstat
```

### Check UCX configuration

```bash
kubectl exec -it deployment/mx-source -c client -- ucx_info -d
```

## License

Apache-2.0
