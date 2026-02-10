# ModelExpress - AI Assistant Context

This file provides context for AI assistants (Claude, Cursor, Copilot) working on the ModelExpress codebase. It summarizes the project architecture, key concepts, and common development patterns.

## Project Overview

**ModelExpress** enables high-performance GPU-to-GPU model weight transfers between vLLM instances using NVIDIA NIXL over RDMA/InfiniBand. Instead of each vLLM instance loading weights from storage, one "source" instance loads the model and transfers weights directly to "target" instances via GPU memory.

### Key Value Proposition

- **Speed**: Transfer 681GB (DeepSeek-V3) in ~15 seconds vs. ~25 minutes from NVMe storage
- **Efficiency**: GPU-to-GPU transfers via GPUDirect RDMA bypass CPU entirely (zero-copy)
- **Scalability**: Coordinate transfers across multiple vLLM instances in a cluster

### Current Status

| Model | Status | Transfer Time | Notes |
|-------|--------|---------------|-------|
| DeepSeek-V3 (671B, FP8) | Working | ~15s | 681GB across 8 GPUs @ ~112 Gbps per link |
| Llama 3.3 70B | Working | ~5s | 140GB across 8 GPUs @ ~112 Gbps |

---

## Architecture

```
   Node A (Source)                           Node B (Target)
   +---------------------------+             +---------------------------+
   | vLLM + MxSourceModelLoader|             | vLLM + MxTargetModelLoader|
   | - Load weights from disk  |             | - Create dummy weights    |
   | - Register with NIXL      | === RDMA ==>| - Receive via RDMA        |
   | - Publish to server       |             | - Run FP8 processing      |
   +-------------+-------------+             +-------------+-------------+
                 |                                         |
                 v                                         v
   +---------------------------------------------------------------+
   |                    ModelExpress Server (Rust)                  |
   |              Redis: model_name -> worker metadata              |
   +---------------------------------------------------------------+
```

### Components

| Component | Language | Location | Purpose |
|-----------|----------|----------|---------|
| ModelExpress Server | Rust | `modelexpress_server/` | gRPC coordinator, stores metadata in Redis |
| Python Client | Python | `modelexpress_client/python/` | vLLM loaders, NIXL transfer manager |
| Rust Client | Rust | `modelexpress_client/src/` | CLI tools, test utilities |
| Common | Rust | `modelexpress_common/` | Protobuf definitions, shared types |

---

## NIXL Integration

### What is NIXL?

NIXL (NVIDIA Interconnect eXchange Library) is a library for zero-copy GPU-to-GPU RDMA transfers. It sits on top of UCX (Unified Communication X) and provides:

- **Agent-based architecture**: Each GPU worker creates a NIXL agent
- **Memory registration**: GPU memory must be registered for RDMA access
- **Transfer descriptors**: NIXL uses descriptors to track source/destination addresses
- **Backend support**: UCX for InfiniBand RDMA, GDS for storage, etc.

### How ModelExpress Uses NIXL

```python
# 1. Create NIXL agent (one per GPU worker)
from nixl._api import nixl_agent, nixl_agent_config

config = nixl_agent_config(backends=["UCX"])
agent = nixl_agent("worker-0", config)

# 2. Register GPU tensors for RDMA access
tensors = [(tensor.data_ptr(), tensor.numel() * tensor.element_size(), device_id, "cuda")]
agent.register_memory(tensors, "VRAM")

# 3. Get metadata for remote agent connection
metadata = agent.get_local_md()  # Share this with target

# 4. On target: connect to source and transfer
agent.add_remote_agent("source-worker-0", source_metadata)

# Prepare transfer descriptors
src_descs = agent.prep_xfer_dlist("source-worker-0", source_tensors, "cuda", ["UCX"])
dst_descs = agent.prep_xfer_dlist("", local_tensors, "cuda", ["UCX"])

# Execute RDMA read
handle = agent.make_prepped_xfer("READ", dst_descs, indices, src_descs, indices, ["UCX"])
agent.transfer(handle)

# Wait for completion
while agent.check_xfer_state(handle) not in ("DONE", "SUCCESS"):
    time.sleep(0.001)
agent.release_xfer_handle(handle)
```

### Key NIXL Concepts

| Concept | Description |
|---------|-------------|
| **Agent** | NIXL instance managing one GPU's memory registrations and transfers |
| **Memory Registration** | Must register GPU memory before RDMA access; generates rkeys |
| **Metadata** | Serialized agent info (address, rkeys) shared between source/target |
| **Transfer Descriptor** | Prepared list of (addr, size, device) for efficient bulk transfer |
| **rkey** | Remote key - RDMA authorization token for remote memory access |

### NIXL/UCX Environment Variables

```yaml
# Transport layers
UCX_TLS: "rc_x,rc,dc_x,dc,cuda_copy"  # InfiniBand + CUDA copy

# Zero-copy RDMA
UCX_RNDV_SCHEME: "get_zcopy"  # Use RDMA read for zero-copy
UCX_RNDV_THRESH: "0"          # Force rendezvous for all messages

# Logging
NIXL_LOG_LEVEL: "INFO"        # DEBUG for troubleshooting
UCX_LOG_LEVEL: "WARN"         # DEBUG for troubleshooting
```

---

## Repository Structure

```
modelexpress/
├── CLAUDE.md                 # THIS FILE (project root) - AI assistant context
├── modelexpress_server/      # Rust gRPC server
│   └── src/
│       ├── main.rs
│       ├── p2p_service.rs    # PublishMetadata, GetMetadata RPCs
│       └── state.rs          # Redis-backed storage
├── modelexpress_client/
│   └── python/
│       └── modelexpress/
│           ├── vllm_loader.py    # MxSourceModelLoader, MxTargetModelLoader
│           ├── nixl_transfer.py  # NixlTransferManager
│           ├── types.py          # TensorDescriptor, WorkerMetadata
│           └── p2p_pb2*.py       # Generated gRPC stubs
├── modelexpress_common/
│   └── proto/
│       └── p2p.proto         # Protobuf service definitions
├── examples/
│   └── p2p_transfer_k8s/
│       ├── vllm-source.yaml  # Source Kubernetes deployment
│       ├── vllm-target.yaml  # Target Kubernetes deployment
│       └── modelexpress-server.yaml
└── docs/
    ├── CONTEXT.md            # Detailed engineering context
    ├── OPTIMIZATION_PLAN.md  # Performance optimization roadmap
    └── CONTIGUOUS_CONTEXT.md # Contiguous region debugging context
```

---

## Key Files

### `modelexpress_client/python/modelexpress/vllm_loader.py`

Contains custom vLLM model loaders:

- **`MxSourceModelLoader`**: Loads weights from disk, registers with NIXL, publishes metadata
- **`MxTargetModelLoader`**: Creates dummy weights, receives via RDMA, applies FP8 processing

```python
class MxSourceModelLoader(DefaultModelLoader):
    """Loads model from disk and publishes for RDMA transfer."""
    
    def load_model(self, vllm_config, model_config):
        model = initialize_model(...)
        self.load_weights(model, model_config)  # Load from disk
        
        # CRITICAL: Register BEFORE FP8 processing
        self._register_raw_tensors(model, device)
        
        process_weights_after_loading(...)  # FP8 transform
        return model.eval()
```

### `modelexpress_client/python/modelexpress/nixl_transfer.py`

Contains `NixlTransferManager`:

- **Agent lifecycle**: Create, initialize, destroy NIXL agents
- **Memory registration**: Register tensors or contiguous regions
- **Transfer execution**: Prepare descriptors, execute RDMA, wait for completion
- **Contiguous regions**: Coalesce adjacent tensors for reduced overhead

```python
class NixlTransferManager:
    """Manages NIXL agent and RDMA transfers for a single GPU worker."""
    
    def register_tensors(self, tensors: dict[str, torch.Tensor]) -> bytes:
        """Register tensors with NIXL for RDMA access."""
        
    def receive_from_source(self, source_tensors, source_metadata, ...):
        """Execute RDMA transfer from source to local tensors."""
```

### `modelexpress_server/src/p2p_service.rs`

Rust gRPC service implementation:

- **`PublishMetadata`**: Source workers publish NIXL metadata + tensor descriptors
- **`GetMetadata`**: Target workers query for source with matching model name
- **Worker merging**: Server merges workers by rank (critical for multi-GPU)

---

## Common Development Tasks

### Building Docker Image

```bash
cd path/to/modelexpress

# Build client image
docker build -f examples/p2p_transfer_k8s/Dockerfile.client \
  -t nvcr.io/nvidian/dynamo-dev/IMAGE_NAME:YOUR_TAG .

docker push nvcr.io/nvidian/dynamo-dev/IMAGE_NAME:YOUR_TAG
```

### Deploying to Kubernetes

```bash
# Namespace
NAMESPACE=<your-namespace>

# 1. Flush Redis (clear stale metadata)
microk8s kubectl -n $NAMESPACE exec deploy/modelexpress-server -c redis -- redis-cli FLUSHALL

# 2. Delete existing deployments
microk8s kubectl -n $NAMESPACE delete deployment mx-source mx-target

# 3. Deploy fresh
microk8s kubectl -n $NAMESPACE apply -f examples/p2p_transfer_k8s/vllm-source.yaml
microk8s kubectl -n $NAMESPACE apply -f examples/p2p_transfer_k8s/vllm-target.yaml

# 4. Monitor
watch microk8s kubectl -n $NAMESPACE get pods -l 'app in (mx-source, mx-target)'
```

### Debugging

```bash
# Stream logs
kubectl -n $NAMESPACE logs -f deploy/mx-source
kubectl -n $NAMESPACE logs -f deploy/mx-target

# Check Redis state
kubectl -n $NAMESPACE exec deploy/modelexpress-server -c redis -- redis-cli KEYS '*'

# Test inference
kubectl -n $NAMESPACE exec deploy/mx-target -- curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-ai/DeepSeek-V3", "prompt": "Hello", "max_tokens": 10}'
```

---

## Environment Variables

### ModelExpress Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MX_REGISTER_LOADERS` | `1` | Auto-register mx-source/mx-target loaders with vLLM |
| `MODEL_EXPRESS_URL` | `localhost:8001` | gRPC server address (also reads `MX_SERVER_ADDRESS` for compat) |
| `MX_CONTIGUOUS_REG` | `0` | Enable contiguous region registration (experimental) |
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

---

## FP8 Model Handling (DeepSeek-V3)

### The Problem

DeepSeek-V3 uses FP8 quantization with scale factors. vLLM's `process_weights_after_loading()`:
1. Renames `weight_scale_inv` → `weight_scale`
2. Transforms the scale data format
3. Deletes the original `weight_scale_inv` parameter

If we transfer AFTER processing, source has `weight_scale` but target expects `weight_scale_inv` → **mismatch!**

### The Solution

Transfer RAW tensors BEFORE `process_weights_after_loading()` runs:

```
Source:                              Target:
┌─────────────────────┐              ┌─────────────────────┐
│ Load weight_scale_inv│              │ Dummy weight_scale_inv│
│ from safetensors    │              │                     │
├─────────────────────┤              ├─────────────────────┤
│ Register raw tensors│───RDMA──────>│ Receive raw tensors │
│ with NIXL           │              │ into dummy memory   │
├─────────────────────┤              ├─────────────────────┤
│ process_weights:    │              │ process_weights:    │
│ scale_inv → scale   │              │ scale_inv → scale   │
│ (identical)         │              │ (identical)         │
└─────────────────────┘              └─────────────────────┘
         ↓                                    ↓
   Identical weights!                  Identical weights!
```

---

## Known Issues

### Issue 1: NIXL_ERR_REMOTE_DISCONNECT

**Symptom**: Target fails with `Remote access error on mlx5_X:1/IB`

**Common Causes**:
- Source crashed/restarted during transfer (stale rkeys)
- UCX transport misconfiguration
- Premature target connection attempts during source warmup

**Solutions**:
- Use robust NIXL ready coordination (implemented in vllm-source.yaml)
- Check source pod for restarts
- Enable UCX_LOG_LEVEL=DEBUG for diagnostics

### Issue 2: Contiguous Region Failures

**Status**: BLOCKED - See `docs/CONTIGUOUS_CONTEXT.md`

When `MX_CONTIGUOUS_REG=1`, transfers fail with Remote access error even when source is stable. The issue is fundamental to how contiguous regions are registered vs accessed.

**Current Workaround**: Use baseline mode (`MX_CONTIGUOUS_REG=0`)

### Issue 3: Long Source Warmup

DeepSeek-V3 takes ~40 minutes to fully warm up (loading + DeepGemm + CUDA graphs). Target must wait via Redis coordination.

---

## Coordination Protocol

### Redis Keys

| Key Pattern | Purpose |
|-------------|---------|
| `mx:nixl_ready:{model}:worker:{id}` | Source stability signal (published after warmup) |
| `mx:model:{model}` | Model metadata (via gRPC, not directly in Redis) |

### Flow

1. **Source starts**: Loads weights, registers with NIXL, publishes metadata to gRPC server
2. **Source warmup**: DeepGemm compilation, CUDA graph capture (~13 min)
3. **Source publishes NIXL ready**: Background script waits for health + test inference, then publishes Redis flag
4. **Target waits**: Polls Redis for `mx:nixl_ready` flag
5. **Target transfers**: Executes RDMA reads from source
6. **Target warmup**: Same DeepGemm + CUDA graph as source

---

## Performance

### Current Numbers

| Metric | Value |
|--------|-------|
| Model | DeepSeek-V3 (671B, FP8) |
| Total Data | 681 GB (8 workers × 85 GB) |
| Transfer Time | ~15 seconds (8 parallel RDMA streams @ 112 Gbps each) |
| Per-Worker Speed | 60-112 Gbps |
| Theoretical Max | 400 Gbps per NIC |

### Optimization Opportunities

See `docs/OPTIMIZATION_PLAN.md` for detailed analysis:

1. **Contiguous regions** (`MX_CONTIGUOUS_REG=1`): BLOCKED - needs investigation
2. **Warm source pool**: Keep source always running
3. **Kernel caching**: Cache DeepGemm compiled kernels
4. **Multi-rail RDMA**: `UCX_IB_NUM_PATHS=2` if dual NICs available

---

## Related Documentation

| Document | Purpose |
|----------|---------|
| `docs/CONTEXT.md` | Detailed engineering context, debugging commands |
| `docs/OPTIMIZATION_PLAN.md` | Performance analysis and optimization roadmap |
| `docs/CONTIGUOUS_CONTEXT.md` | Contiguous region debugging history |
| `docs/CLI.md` | CLI tool documentation |
| `docs/QUICK_START.md` | Getting started guide |

---

## Tips for AI Assistants

1. **Always read before editing**: Use the Read tool to understand context
2. **Check pod status first**: Many issues are caused by pod restarts
3. **Flush Redis on redeploy**: Stale metadata causes transfer failures
4. **Use baseline mode**: `MX_CONTIGUOUS_REG=0` until contiguous is fixed
5. **Long startup times are normal**: DeepSeek-V3 takes ~40 min to warm up
6. **UCX errors need DEBUG logging**: Set `UCX_LOG_LEVEL=DEBUG` for diagnostics
7. **NIXL agents must match ranks**: Source rank 0 → Target rank 0
