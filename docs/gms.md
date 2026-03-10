# MX GMS Integration

ModelExpress (MX) GMS loads model weights into GPU Memory Service (GMS) for zero-copy sharing with inference engines, and registers tensors with NIXL for RDMA transfers to remote nodes.

## How It Works

The MX GMS process runs as a sidecar alongside the inference engine, sharing GPUs. It:

1. Connects to GMS and acquires a GPU memory pool
2. Loads model weights into GMS-managed GPU memory (with TP/EP sharding via vLLM)
3. Registers raw tensors with NIXL for RDMA transfers to target nodes
4. Runs post-processing (FP8 scale transforms, MLA absorption)
5. Commits the final post-processed state to GMS
6. Publishes metadata to the MX Server for coordination
7. Stays alive (`signal.pause()`) so NIXL agents and GPU memory remain valid

The engine then connects to GMS in read-only mode and imports the weights zero-copy.

```
MX Client                           GMS Server          Engine
    |                                   |                   |
    |-- connect(RW) ------------------->|                   |
    |-- allocate via mem pool --------->|                   |
    |-- load weights (disk/GDS/S3)      |                   |
    |-- register NIXL (raw tensors)     |                   |
    |-- post-process (FP8/MLA)          |                   |
    |-- commit + switch to RO --------->|                   |
    |-- publish metadata to MX Server   |                   |
    |-- signal.pause() (stay alive)     |                   |
    |                                   |<-- connect(RO) ---|
    |                                   |<-- import weights |
    |                                   |    (zero-copy)    |
    |                                   |                   |-- serve inference
```

## GMS Overview

GPU Memory Service (GMS) is an out-of-process GPU memory manager that enables zero-copy sharing of model weights across processes via CUDA VMM. Each GPU runs its own GMS server, communicating with clients over Unix Domain Sockets.

```
┌──────────────────┐     Unix Domain Socket     ┌──────────────────┐
│ Client (Sidecar  │ <------------------------> │   GMS Server     │
│  or vLLM engine) │  /tmp/gms_<GPU-UUID>.sock  │ (per-GPU daemon) │
└──────────────────┘                            └──────────────────┘
        |                                               |
        |  CUDA VMM (cuMemMap) + FD passing             |
        v                                               v
   GPU Memory (client VA)                    GPU Memory (server allocs)
```

Clients connect with either an **RW** lock (exclusive, for writing weights) or **RO** lock (shared, for reading). The MX client connects RW, loads weights, commits, then switches to RO. The engine connects RO and imports weights zero-copy.

For multi-GPU, each GPU has its own independent GMS server. Workers derive their socket path from the GPU UUID — no cross-GPU coordination needed.

For more details on GMS internals, see the [GMS codebase](https://github.com/ai-dynamo/dynamo/tree/main/lib/gpu_memory_service).

## MX GMS Code Structure

```
modelexpress/gms/
├── __init__.py
├── __main__.py          # python -m modelexpress.gms
├── config.py            # GmsConfig, MxConfig, enums
├── main.py              # CLI dispatcher
├── mx_hooks.py          # Shared hooks: GMS connect/commit, NIXL registration, MX Server publish
├── launchers/
│   └── vllm.py          # vLLM launcher: mp.spawn workers, distributed init, weight loading
└── weight_sources/
    ├── disk.py           # Default disk loading (via vLLM's DefaultModelLoader)
    ├── gds.py            # GPUDirect Storage (stub)
    └── s3.py             # S3 streaming (stub)
```

### Key Design Decisions

**GMS memory pool**: All tensor allocations during model init, weight loading, and post-processing go through `use_mem_pool(pool)`. This is required because `register_module_tensors()` only recognizes tensors allocated through GMS's pool.

**Hook ordering**: NIXL registers raw (pre-post-processing) tensors so targets can receive them via RDMA and run post-processing locally. GMS commits the final post-processed state so the engine reads the correct weights.

**Process lifecycle**: Each worker calls `signal.pause()` after loading. This keeps NIXL agents registered and GPU memory allocated. RDMA reads are one-sided (target NIC reads directly from source GPU memory without CPU involvement). The process exits on SIGTERM from kubelet.

**GMS finalize flow**: After `commit()`, the client calls `disconnect()` then `connect(RO)` (matching the reference GMS loader in Dynamo).

## CLI

```bash
python -m modelexpress.gms \
    --model <model_name_or_path> \
    --engine vllm \
    --mode source \
    --tp-size 8 \
    --mx-server modelexpress-server:8001 \
    --trust-remote-code \
    --enable-expert-parallel  # for MoE models
```

## Container Image

Built from `container/Dockerfile.client`:

```bash
docker build -f container/Dockerfile.client \
    --build-arg ENABLE_GMS=true \
    --build-arg VLLM_VERSION=0.15.1 \
    -t modelexpress-client:latest .
```

| Build Arg | Default | Description |
|-----------|---------|-------------|
| `ENABLE_GMS` | `false` | Install `gpu-memory-service` and GMS dependencies |
| `VLLM_VERSION` | `0.15.1` | Pin vLLM version |
| `CUDA_VERSION` | `12.9.1` | CUDA toolkit version |

Uses multi-stage build with `uv` for fast dependency resolution. Layer caching: changing Python source only rebuilds the final `pip install --no-deps` layer.

## Installation (Development)

```bash
cd modelexpress_client/python
pip install -e ".[gms]"
```

This installs `gpu-memory-service` from the Dynamo repo, plus `cuda-python`, `pynvml`, and `vllm`.

Verify:

```bash
python -c "from gpu_memory_service import get_or_create_gms_client_memory_manager; print('OK')"
python -c "from modelexpress.gms.main import main; print('OK')"
```

## Quick Start (Local)

### Bare Metal

```bash
# Terminal 1: GMS server
python -m gpu_memory_service --device 0

# Terminal 2: MX client (load weights into GMS + register NIXL)
python -m modelexpress.gms --model Qwen/Qwen3-0.6B --engine vllm --mode source --tp-size 1

# Terminal 3: vLLM engine (reads from GMS)
python -m dynamo.vllm --model Qwen/Qwen3-0.6B --load-format gms
```

### Docker

Build the MX server and client images first (see [container/README.md](../container/README.md)).
The GMS server and engine images come from the [Dynamo container builds](https://github.com/ai-dynamo/dynamo/tree/main/container).

```bash
TP_SIZE=1
MODEL=Qwen/Qwen3-0.6B
MX_SERVER_IMAGE=modelexpress-server:latest
MX_CLIENT_IMAGE=modelexpress-client:gms
DYNAMO_IMAGE=dynamo-vllm:latest  # from Dynamo repo

# Shared dir for GMS UDS sockets
mkdir -p /tmp/mx-test

# 1. Redis + MX Server
docker run -d --name mx-redis --network host redis:7-alpine
docker run -d --name mx-server --network host \
    -e REDIS_URL=redis://localhost:6379 \
    ${MX_SERVER_IMAGE}

# 2. GMS server (one process per GPU)
docker run -d --name gms-server --network host \
    --gpus all --ipc=host \
    -e TMPDIR=/shared \
    -v /tmp/mx-test:/shared \
    ${DYNAMO_IMAGE} \
    bash -c '
        for dev in $(seq 0 $((TP_SIZE - 1))); do
            python3 -m gpu_memory_service --device "$dev" &
        done
        wait -n
    '

# Wait for sockets
watch "ls /tmp/mx-test/gms_*.sock 2>/dev/null | wc -l"

# 3. MX client (load weights into GMS + register NIXL)
docker run --name mx-client --network host \
    --gpus all --ipc=host --cap-add IPC_LOCK \
    -e TMPDIR=/shared \
    -v /tmp/mx-test:/shared \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    ${MX_CLIENT_IMAGE} \
    python3 -m modelexpress.gms \
        --model ${MODEL} \
        --engine vllm \
        --mode source \
        --tp-size ${TP_SIZE} \
        --mx-server localhost:8001

# 4. Verify
ls -la /tmp/mx-test/gms_*.sock
docker logs mx-client --tail=30

# 5. Cleanup
docker rm -f mx-client gms-server mx-server mx-redis
rm -rf /tmp/mx-test
```
