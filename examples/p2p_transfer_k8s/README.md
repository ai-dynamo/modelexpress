# P2P GPU Transfer Example

This example demonstrates how to set up ModelExpress for P2P GPU weight transfers between vLLM instances on Kubernetes.

## Architecture

```
Node A (Source - first to start)              Node B (Target - starts later)
+----------------------------------+          +----------------------------------+
| vLLM Container                   |          | vLLM Container                   |
| - Loads real model weights       |          | - Starts with dummy weights      |
| - Exposes weights via ZMQ        |          | - Exposes buffers via ZMQ        |
| - MX_ZMQ_ADDRESS=ipc:///tmp/mx/  |          | - MX_ZMQ_ADDRESS=ipc:///tmp/mx/  |
+----------------------------------+          +----------------------------------+
        |  ZMQ (IPC sockets)                          |  ZMQ (IPC sockets)
        v                                             v
+----------------------------------+          +----------------------------------+
| Client Container                 |          | Client Container                 |
| - Creates NIXL agents (1 per GPU)|          | - Creates NIXL agents (1 per GPU)|
| - Queries server: no source found|   RDMA  | - Queries server: finds source A |
| - Becomes source, publishes meta |<========>| - Receives weights via NIXL      |
+----------------------------------+  NIXL   | - Also publishes metadata        |
        |                                     +----------------------------------+
        |                                             |
        v                                             v
+--------------------------------------------------------------------+
|                    ModelExpress Server (CPU)                        |
|   - Stores model metadata (NIXL metadata + tensor descriptors)      |
|   - Keyed by model name                                            |
|   - Redis backend for persistence                                  |
+--------------------------------------------------------------------+
```

### Key Design Points

1. **Client Container**: NIXL transfer logic runs in a separate client container, not in vLLM
2. **Symmetric Clients**: Both source and target run identical client code; role is determined dynamically
3. **ZMQ Communication**: vLLM exposes weights via ZMQ IPC sockets (one per TP rank)
4. **Tensor Parallelism**: Full TP > 1 support with rank-matched transfers

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
