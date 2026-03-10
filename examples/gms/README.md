# MX GMS Deployment

Deploy ModelExpress with GMS integration on Kubernetes or locally with Docker.

## Kubernetes (DGD)

### Prerequisites

- Dynamo operator installed with GMS support
- ModelExpress server image (`modelexpress-server`)
- MX client image (`modelexpress-client`) built with `ENABLE_GMS=true`
- Dynamo runtime image with GMS model loader
- PVC with model weights
- Secrets: `ngc-secret-hwoo` (image pull), `hf-token-secret-hwoo` (HuggingFace)

### Deploy

```bash
cd gms_mx/

# Qwen3-0.6B on 2 GPUs (defaults)
./run.sh

# Kimi K2.5 on 8 GPUs (auto-enables EP + reasoning parsers)
MODEL_NAME=moonshotai/Kimi-K2.5 TP_SIZE=8 ./run.sh

# DeepSeek-V3 on 8 GPUs (auto-enables EP)
MODEL_NAME=deepseek-ai/DeepSeek-V3 TP_SIZE=8 ./run.sh

# Llama 70B on 8 GPUs (dense, no EP)
MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct TP_SIZE=8 ./run.sh

# Dry run (print YAML without applying)
DRY_RUN=1 ./run.sh

# Delete everything
./run.sh delete
```

### Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `NAMESPACE` | `hwoo` | K8s namespace |
| `MODEL_NAME` | `Qwen/Qwen3-0.6B` | Model name |
| `TP_SIZE` | `2` | Tensor parallel size (= number of GPUs) |
| `ENABLE_EP` | auto-detected | Expert parallelism (`1`/`0`). Auto-enables for MoE models |
| `IMAGE_TAG` | `gms-vllm-runtime-...` | Engine container image tag |
| `MX_IMAGE_TAG` | `vllm-runtime-...` | MX client container image tag |
| `VLLM_EXTRA_ARGS` | auto-detected | Extra engine args (e.g. Dynamo reasoning/tool parsers) |
| `MX_SERVER` | `modelexpress-server:8001` | MX Server address |
| `SHM_SIZE` | `64Gi` | Shared memory size |
| `DRY_RUN` | `0` | Print YAML without applying |

`ENABLE_EP` and `VLLM_EXTRA_ARGS` are auto-detected from `MODEL_NAME`:

| Model | `ENABLE_EP` | `VLLM_EXTRA_ARGS` |
|-------|-------------|-----|
| `moonshotai/Kimi-K2.5` | `1` | `--dyn-reasoning-parser kimi_k25 --dyn-tool-call-parser kimi_k2` |
| `deepseek-ai/DeepSeek-V3` | `1` | (none) |
| Others | `0` | (none) |

All auto-detected values can be overridden: `ENABLE_EP=0 MODEL_NAME=moonshotai/Kimi-K2.5 ./run.sh`

### Pod Layout

Each VllmWorker pod has three containers sharing GPUs:

| Container | Role | Image |
|-----------|------|-------|
| `gms-weights` | Per-GPU memory manager (CUDA VMM), UDS sockets | Dynamo runtime (operator-managed) |
| `mx-client` | Load weights into GMS, register NIXL, stay alive | `modelexpress-client` |
| `engine-0/1` | Read weights from GMS, serve inference | Dynamo runtime |

Startup order: `gms-weights` -> `mx-client` -> engine.

The Dynamo operator auto-injects `TMPDIR=/shared` and the shared volume into the main container and `gms-weights`, but **not** into custom init containers like `mx-client`. The DGD template handles this explicitly.

### Building Images

```bash
# MX client image with GMS support
docker build -f container/Dockerfile.client \
    --build-arg ENABLE_GMS=true \
    --build-arg VLLM_VERSION=0.15.1 \
    -t nvcr.io/nvidian/dynamo-dev/modelexpress-client:vllm-runtime-latest-cuda-12.9.1 .

docker push nvcr.io/nvidian/dynamo-dev/modelexpress-client:vllm-runtime-latest-cuda-12.9.1
```

### Monitoring

```bash
# Watch pods
kubectl get pods -n $NAMESPACE -w

# MX client logs
kubectl logs -n $NAMESPACE -l component=vllmworker -c mx-client --tail=50

# GMS server logs
kubectl logs -n $NAMESPACE -l component=vllmworker -c gms-weights --tail=50

# Engine logs
kubectl logs -n $NAMESPACE -l component=vllmworker -c engine-0 --tail=50

# MX Server logs
kubectl logs -n $NAMESPACE deploy/modelexpress-server --tail=50

# Check Redis state
kubectl exec -n $NAMESPACE deploy/modelexpress-server -c redis -- redis-cli KEYS '*'
```

For local Docker-based testing, see [docs/gms.md](../../docs/gms.md#docker).
