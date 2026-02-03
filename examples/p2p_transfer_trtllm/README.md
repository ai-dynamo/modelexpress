# TensorRT-LLM P2P Transfer Example

This example demonstrates GPU-to-GPU model weight transfer for TensorRT-LLM using ModelExpress and NIXL over RDMA.

## Overview

TensorRT-LLM uses a three-phase workflow: **Convert → Build → Run**. This example enables P2P transfer at the checkpoint level:

1. **Source**: Loads TRT-LLM checkpoint to GPU, registers with NIXL, publishes metadata
2. **Target**: Receives checkpoint via RDMA, saves locally, builds TensorRT engine

```
Source Node                                    Target Node
┌─────────────────────────┐                   ┌─────────────────────────┐
│ TRT-LLM Checkpoint      │                   │ MxTrtllmTargetLoader    │
│ ├── config.json         │                   │ ├── Query MX server     │
│ ├── rank0.safetensors   │  ═══ RDMA ══════► │ ├── Receive checkpoint  │
│ └── rank1.safetensors   │                   │ ├── Save locally        │
│                         │                   │ └── Build engine        │
│ MxTrtllmSourcePublisher │                   │                         │
│ └── Publish to MX server│                   │ TRT-LLM Engine          │
└─────────────────────────┘                   └─────────────────────────┘
```

## Prerequisites

1. **Kubernetes cluster** with GPU nodes and InfiniBand/RoCE networking
2. **ModelExpress server** deployed (see `../p2p_transfer_k8s/modelexpress-server.yaml`)
3. **TRT-LLM checkpoint** prepared on a shared PVC or source node
4. **RDMA resources** configured (`rdma/ib` resource)

## Preparing a TRT-LLM Checkpoint

First, convert a HuggingFace model to TRT-LLM checkpoint format:

```bash
# Clone TensorRT-LLM
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM/examples/llama

# Convert HuggingFace model to TRT-LLM checkpoint
python convert_checkpoint.py \
    --model_dir /path/to/llama-70b-hf \
    --output_dir /path/to/llama-70b/trtllm-checkpoint \
    --dtype float16 \
    --tp_size 8

# This creates:
# /path/to/llama-70b/trtllm-checkpoint/
# ├── config.json
# ├── rank0.safetensors
# ├── rank1.safetensors
# └── ... (rank2-7.safetensors)
```

## Deployment

### 1. Deploy ModelExpress Server

```bash
kubectl apply -f ../p2p_transfer_k8s/modelexpress-server.yaml
```

### 2. Deploy TRT-LLM Source

Edit `trtllm-source.yaml` to set:
- `MODEL_NAME`: Your model identifier
- `CHECKPOINT_DIR`: Path to TRT-LLM checkpoint
- PVC name if using shared storage

```bash
kubectl apply -f trtllm-source.yaml
```

### 3. Deploy TRT-LLM Target

Edit `trtllm-target.yaml` to set:
- `MODEL_NAME`: Same as source
- Build configuration (batch size, sequence length, etc.)

```bash
kubectl apply -f trtllm-target.yaml
```

## Configuration

### Source Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `llama-70b` | Model identifier for coordination |
| `CHECKPOINT_DIR` | `/models/checkpoint` | Path to TRT-LLM checkpoint |
| `MODEL_EXPRESS_URL` | `modelexpress-server:8001` | MX server address |
| `NIXL_LOG_LEVEL` | `INFO` | NIXL logging level |
| `UCX_LOG_LEVEL` | `WARN` | UCX logging level |

### Target Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `llama-70b` | Model identifier (must match source) |
| `OUTPUT_DIR` | `/tmp/mx_trtllm` | Local output directory |
| `MODEL_EXPRESS_URL` | `modelexpress-server:8001` | MX server address |
| `TRTLLM_MAX_BATCH_SIZE` | `8` | Engine max batch size |
| `TRTLLM_MAX_INPUT_LEN` | `2048` | Engine max input length |
| `TRTLLM_MAX_SEQ_LEN` | `4096` | Engine max sequence length |

## Python API

### Source Side

```python
from modelexpress import MxTrtllmSourcePublisher

# Initialize and publish checkpoint
publisher = MxTrtllmSourcePublisher(
    checkpoint_dir="/path/to/checkpoint",
    model_name="llama-70b",
    mx_server="modelexpress-server:8001"
)
publisher.initialize()

# Keep running to serve weights
# ... (weights remain in GPU memory)

# Cleanup when done
publisher.shutdown()
```

### Target Side

```python
from modelexpress import MxTrtllmTargetLoader, create_trtllm_from_mx

# Option 1: Full control
loader = MxTrtllmTargetLoader(
    model_name="llama-70b",
    mx_server="modelexpress-server:8001",
    output_dir="/tmp/mx_trtllm",
    build_config={
        "gemm_plugin": "auto",
        "max_batch_size": "8",
        "max_input_len": "2048",
        "max_seq_len": "4096",
    }
)
engine_dir = loader.load()

# Option 2: Convenience function
engine_dir = create_trtllm_from_mx(
    model_name="llama-70b",
    mx_server="modelexpress-server:8001"
)

# Use with TRT-LLM
from tensorrt_llm import LLM
llm = LLM(model=engine_dir)
output = llm.generate("Hello, world!")
```

### Skip Engine Build

If you only need the checkpoint (e.g., to build engine separately):

```python
loader = MxTrtllmTargetLoader(model_name="llama-70b", ...)
checkpoint_dir = loader.load(skip_build=True)

# Build engine manually later
# trtllm-build --checkpoint_dir {checkpoint_dir} --output_dir {engine_dir}
```

## Monitoring

### Check Source Status

```bash
kubectl logs -f deploy/trtllm-source

# Expected output:
# Starting TRT-LLM source publisher
# Loading rank 0 weights from /models/checkpoint/rank0.safetensors
# Rank 0: 512 tensors, 17.5 GB
# ...
# Published metadata for 8 workers to ModelExpress server
# Source publisher ready - serving weights via P2P
```

### Check Target Status

```bash
kubectl logs -f deploy/trtllm-target

# Expected output:
# Starting TRT-LLM target loader
# Querying for source model: llama-70b
# Found source with 8 workers
# Receiving rank 0: 512 tensors
# Rank 0: Received 17.5 GB in 2.1s (66.7 Gbps)
# ...
# Transfer complete: 140.0 GB total in 18.5s (60.5 Gbps)
# Building TRT-LLM engine from checkpoint...
# Engine built in 45.2s at /tmp/mx_trtllm/engine
```

### Check ModelExpress Server

```bash
# Check registered models
kubectl exec deploy/modelexpress-server -c redis -- redis-cli KEYS '*'

# View model metadata
kubectl exec deploy/modelexpress-server -c redis -- redis-cli GET 'mx:model:llama-70b'
```

## Performance

### Expected Transfer Times

| Model | Size | Transfer Time | Notes |
|-------|------|---------------|-------|
| Llama-7B (TP=1) | 14 GB | ~2s | Single GPU |
| Llama-70B (TP=8) | 140 GB | ~20s | 8 GPUs parallel |
| DeepSeek-V3 (TP=8) | 681 GB | ~80s | FP8, 8 GPUs |

### Total Time Breakdown

For Llama-70B with TP=8:
- P2P Transfer: ~20s
- Engine Build: ~15-30min (depends on hardware)
- **Total: ~15-30min** (vs ~5min load + ~15-30min build from NVMe)

The main benefit is when source is already running - targets can receive weights while source serves inference.

## Troubleshooting

### Source Not Found

```
TimeoutError: Source not found for llama-70b after 3600s
```

- Verify source pod is running: `kubectl get pods -l app=trtllm-source`
- Check source logs for errors: `kubectl logs deploy/trtllm-source`
- Verify MODEL_NAME matches between source and target

### NIXL Transfer Failed

```
RuntimeError: Transfer failed with status ERR
```

- Check RDMA connectivity between nodes
- Verify UCX environment variables are set
- Enable debug logging: `UCX_LOG_LEVEL=DEBUG`

### Engine Build Failed

```
RuntimeError: trtllm-build failed with code 1
```

- Check target has enough disk space in OUTPUT_DIR
- Verify checkpoint was transferred completely
- Check trtllm-build output for specific errors

## Architecture Differences from vLLM

| Aspect | vLLM P2P | TRT-LLM P2P |
|--------|----------|-------------|
| **Transfer Level** | Runtime tensors | Checkpoint files |
| **Integration** | Custom model loader | Separate publisher/loader |
| **Build Step** | Not needed | Required after transfer |
| **FP8 Handling** | Pre-process transfer | Handled in checkpoint |
| **Use Case** | Sub-second replication | Checkpoint distribution |

## See Also

- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [vLLM P2P Example](../p2p_transfer_k8s/README.md)
- [ModelExpress Architecture](../../docs/TRTLLM_MX.md)
