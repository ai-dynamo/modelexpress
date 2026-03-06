# TRT-LLM Phase 3: Live Model P2P Transfer (GPU Param → GPU Param)

**Status**: **VALIDATED** — Qwen 0.5B (TP=1) and Llama 3.1 70B (TP=8)
**Date**: February 19, 2026
**Prerequisite**: Phase 2 complete (PRESHARDED validated for TP=1 and TP=8)

---

## Table of Contents

1. [Summary](#1-summary)
2. [Motivation — Why Phase 3](#2-motivation--why-phase-3)
3. [Architecture Overview](#3-architecture-overview)
4. [Detailed Data Flow](#4-detailed-data-flow)
5. [Implementation: Source Side](#5-implementation-source-side)
6. [Implementation: Target Side](#6-implementation-target-side)
7. [TRT-LLM Upstream Changes](#7-trt-llm-upstream-changes)
8. [Docker Image Build](#8-docker-image-build)
9. [Kubernetes Deployment](#9-kubernetes-deployment)
10. [Name Matching: How Source and Target Params Align](#10-name-matching-how-source-and-target-params-align)
11. [Performance Results](#11-performance-results)
12. [Comparison: Phase 2 vs Phase 3](#12-comparison-phase-2-vs-phase-3)
13. [Issues Encountered and Resolved](#13-issues-encountered-and-resolved)
14. [MX Server Behavior for TP>1](#14-mx-server-behavior-for-tp1)
15. [Observability: Per-Rank Transfer Logging](#15-observability-per-rank-transfer-logging)
16. [Open Items and Future Work](#16-open-items-and-future-work)

---

## 1. Summary

Phase 3 achieves **true zero-copy GPU-to-GPU model weight transfer** between TensorRT-LLM
instances. A running TRT-LLM model's GPU parameter buffers are registered directly with
NIXL on the source, and the target receives via RDMA directly into its own model parameter
buffers. There is no disk I/O, no CPU round-trip, no HuggingFace-to-TRT-LLM format
conversion, no weight mapper fusing, and no intermediate buffer allocation.

**Key results (Llama 3.1 70B, TP=8, BF16, cross-node InfiniBand):**

| Metric | Value |
|--------|-------|
| Total model size | 141.1 GB (8 × 17.64 GB) |
| Params per rank | 483 |
| Match rate | 483/483 (100%) |
| RDMA transfer time (wall-clock) | **~3.08 seconds** |
| Per-rank bandwidth | 45.4–46.2 Gbps |
| Aggregate bandwidth | **~368 Gbps** |
| Total LLM load time | 141.6s (includes engine compilation + KV cache) |
| RDMA as % of total load | **~2%** |

---

## 2. Motivation — Why Phase 3

### Phase 2 Recap

Phase 2 introduced `LoadFormat.PRESHARDED` to skip TP re-slicing on the target, enabling
zero-disk loading for TP>1 models. However, it still had unnecessary overhead:

```
Phase 2 data path:
  Source: HF safetensors on disk
       → CPU memory (load all shards)
       → Per-rank TP slicing (HF format, unfused q/k/v)
       → NIXL register (GPU or CPU buffers)
       → gRPC publish metadata

  Target: gRPC query metadata
       → NIXL RDMA receive (HF-format tensors)
       → .cpu() copy (NIXL shutdown invalidates GPU buffers)
       → HfWeightMapper fuse (q+k+v → qkv, gate+up → gate_up)
       → copy_weight() into model params
```

The source reads from disk and slices into HF-format tensors. The target must reverse the
format: fuse separate q/k/v into qkv, map HF names to TRT-LLM names, and copy into the
model. This is backwards — both ends are TRT-LLM instances with identical parameter layouts.

### Phase 3 Insight

If the source is a **running TRT-LLM model**, its parameters are already:
- **Fused**: `q_proj + k_proj + v_proj` → single `qkv_proj.weight`
- **TP-sharded**: each rank holds only its shard
- **On GPU**: at known `data_ptr()` addresses
- **In TRT-LLM naming**: `transformer.layers.0.attention.qkv_proj.weight`

The target creates an identical model structure (same architecture, same TP config), so its
`named_parameters()` produces the **exact same names and shapes**. We can simply RDMA the
bytes from source param buffers to target param buffers, matched by name. No conversion
of any kind.

```
Phase 3 data path:
  Source GPU params → RDMA → Target GPU params
  Done.
```

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│ SOURCE NODE (8× H100, InfiniBand)                                │
│                                                                   │
│  mpirun -np 8 source_tp8.py                                      │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Per rank (parallel via MPI):                             │    │
│  │    HfConfigLoader.load(path, mapping=Mapping(tp=8,rank))  │    │
│  │    AutoModelForCausalLM.from_config(config)               │    │
│  │    HfCheckpointLoader → weight_mapper → model.load_weights│    │
│  │                                                           │    │
│  │  Model params now on GPU: fused, TP-sharded, TRT-LLM fmt │    │
│  └──────────────────────────────────────────────────────────┘    │
│                         │                                         │
│                         ▼                                         │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  MxLiveSource.publish()  (per rank, parallel)             │    │
│  │    model.named_parameters() → 483 params on this GPU      │    │
│  │    NIXL register_tensors(param GPU buffers)                │    │
│  │    gRPC PublishMetadata → MX server                        │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  GPU 0–7: params live in GPU memory, registered with NIXL         │
└────────────────────────────┬─────────────────────────────────────┘
                             │  gRPC (metadata: names, addrs, shapes,
                             │         NIXL agent info, config files)
                             ▼
                  ┌─────────────────────┐
                  │   ModelExpress       │
                  │   Server (Redis)     │
                  │                      │
                  │  model_name →        │
                  │    8 WorkerMetadata  │
                  │    × 483 TensorDesc  │
                  │    + model_files     │
                  └─────────────────────┘
                             │
                             │  gRPC (metadata query)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ TARGET NODE (8× H100, InfiniBand)                                │
│                                                                   │
│  LLM(model=..., tp=8,                                            │
│      checkpoint_loader=MxLiveCheckpointLoader(),                  │
│      load_format=LoadFormat.PRESHARDED)                           │
│                                                                   │
│  TRT-LLM internally spawns 8 MPI workers, each:                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  1. MxConfigLoader.load_config()                          │    │
│  │     → fetch config.json from MX server (not from disk)    │    │
│  │  2. AutoModelForCausalLM.from_config(config)              │    │
│  │     → empty model on GPU (meta init → CUDA)               │    │
│  │  3. PRESHARDED: set _weights_presharded=True on Linears   │    │
│  │  4. MxLiveWeightLoader.load_weights(model=model)          │    │
│  │     → RDMA source params into model params (see §6)       │    │
│  │     → return {}                                           │    │
│  │  5. model_loader sees {} → skip model.load_weights()      │    │
│  │  6. post_load_weights() → finalize                        │    │
│  │  7. KV cache allocation + engine compilation              │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  Model ready for inference                                        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Detailed Data Flow

### 4.1 Source Lifecycle

1. **Model download** (one-time): `snapshot_download()` resolves HuggingFace model to local
   PVC path. Rank 0 downloads, broadcasts path to all ranks via `MPI.COMM_WORLD.bcast()`.

2. **Model loading** (per rank, parallel via MPI):
   - `HfConfigLoader().load(local_path, mapping=mapping, trust_remote_code=True)` creates
     a `ModelConfig` with the correct TP mapping for this rank.
   - `AutoModelForCausalLM.from_config(config)` creates the model with TP-sharded layer
     sizes (e.g., QKV weight is `[num_heads/tp_size * head_dim * 3, hidden_size]`).
   - Meta-init + materialize on CUDA: `MetaInitMode()` → `model._apply(init_meta)` →
     `model.to(f'cuda:{rank}')`.
   - `HfCheckpointLoader().load_weights()` returns HF-format weights.
   - `weight_mapper` maps HF → TRT-LLM names and fuses q+k+v → qkv.
   - `model.load_weights(weights, weight_mapper=weight_mapper)` finalizes.
   - After this, `model.named_parameters()` returns TRT-LLM-format params, fused and
     TP-sharded, all on `cuda:{rank}`.

3. **NIXL registration** (per rank):
   - `MxLiveSource(model, model_name, mx_server).publish()` iterates
     `model.named_parameters()`, filters by `param.device.index == device_id`.
   - Creates a `NixlTransferManager` per rank, calls `register_tensors(param_tensors)`.
     This registers the GPU memory regions (via `data_ptr()`) with NIXL's UCX backend.
   - No memory copy — NIXL pins the existing parameter buffers in-place.

4. **Metadata publication** (per rank):
   - Builds `TensorDescriptor` protos: name, GPU address, byte size, dtype, shape.
   - Builds `WorkerMetadata`: worker_rank, NIXL agent metadata (serialized UCX endpoint),
     tensor descriptors.
   - Sends `PublishMetadataRequest` to MX server via gRPC.
   - Also includes `model_files` (config.json, tokenizer files) so the target can load
     config without needing a PVC.

5. **MPI barrier**: All 8 ranks call `comm.Barrier()` after publishing.

6. **Serve**: All ranks enter `sleep(3600)` loop. The model params stay in GPU memory,
   NIXL agents stay active, and the source is ready to serve RDMA reads indefinitely.

### 4.2 Target Lifecycle

1. **Loader creation**: `MxLiveCheckpointLoader()` is instantiated in the main process.
   It bundles `MxLiveWeightLoader` (for RDMA) and `MxConfigLoader` (for config from MX).

2. **LLM initialization**: `LLM(model=model_name, checkpoint_loader=loader, load_format=LoadFormat.PRESHARDED, tensor_parallel_size=8)`.
   TRT-LLM spawns 8 MPI workers internally. The loader is pickled and sent to each worker.

3. **Config loading** (in each MPI worker):
   - `MxConfigLoader.load_config(checkpoint_dir)` is called.
   - It queries the MX server for `model_files` (published by the source).
   - Writes `config.json`, `tokenizer.json`, etc. to a temp directory.
   - Delegates to `HfConfigLoader().load(temp_dir, mapping=mapping)` for actual parsing.
   - Returns a `ModelConfig` with correct TP mapping for this worker's rank.

4. **Model creation** (in each MPI worker):
   - `AutoModelForCausalLM.from_config(config)` creates the model with TP-sharded shapes.
   - Meta-init → materialize on CUDA → `model.to("cuda")`.
   - At this point, the model has **empty (random) parameter buffers** on GPU.

5. **PRESHARDED flag**: `model_loader.py` sets `m._weights_presharded = True` on all
   `Linear` modules. (Not used in Phase 3 since we return `{}`, but set for consistency.)

6. **Weight transfer** (in each MPI worker):
   - `MxLiveWeightLoader.load_weights(checkpoint_dir, mapping=mapping, model=model)`:
     - Gets `device_id = torch.cuda.current_device()` (the worker's GPU).
     - Queries MX server for source metadata.
     - Finds the source worker with `worker_rank == device_id`.
     - Iterates `model.named_parameters()`, building `target_params` dict.
     - Matches source and target tensors by name, verifying size equality.
     - Initializes NIXL agent, registers target param buffers.
     - Executes RDMA read: for each matched tensor, NIXL reads from source GPU address
       into target GPU address.
     - Returns `{}`.

7. **Empty dict handling**: `model_loader.py` PRESHARDED branch checks `if weights:`.
   Since `{}` is falsy, it skips `model.load_weights()` entirely and logs
   `"PRESHARDED: weights already in model params (direct injection)"`.

8. **Post-processing**: `post_load_weights()` runs on all modules (finalizes weight layouts).
   KV cache is allocated. Engine is compiled. Model is ready for inference.

### 4.3 The RDMA Transfer (Step 6 Detail)

For each of the 8 ranks, running in parallel:

```
  Target rank N                                Source rank N
  ┌────────────────┐                          ┌────────────────┐
  │ NIXL Agent      │   1. loadRemoteAgent()  │ NIXL Agent      │
  │ (UCX endpoint)  │◄─────────────────────── │ (UCX endpoint)  │
  │                 │                          │                 │
  │ For each of     │   2. RDMA GET (×483)    │                 │
  │ 483 params:     │◄═══════════════════════ │ 483 param bufs  │
  │  dst.data_ptr() │   InfiniBand zero-copy  │  src.data_ptr() │
  │                 │                          │                 │
  │ 17.64 GB total  │   3.08s @ 45.8 Gbps     │ 17.64 GB total  │
  └────────────────┘                          └────────────────┘
```

All 8 ranks transfer concurrently. The aggregate data movement is 141.1 GB in ~3.1s
wall-clock time, with each rank independently achieving ~46 Gbps on its InfiniBand link.

---

## 5. Implementation: Source Side

### 5.1 `MxLiveSource` Class

**File**: `modelexpress_client/python/modelexpress/trtllm_live_transfer.py`

```python
class MxLiveSource:
    def __init__(self, model, model_name, mx_server, model_path=None):
        self._model = model           # torch.nn.Module (TRT-LLM model)
        self._model_name = model_name  # HF model ID for MX server key
        self._mx_server = mx_server    # gRPC endpoint
        self._model_path = model_path  # Local HF snapshot path (for config files)
        self._nixl_managers = {}
        self._published = False
```

Key methods:

- **`publish()`**: Main entry point. Calls `_get_torch_model()` to extract the underlying
  `nn.Module`, iterates `named_parameters()`, registers with NIXL, publishes to MX server.

- **`_get_torch_model()`**: Handles both direct `nn.Module` instances and `LLM` wrapper
  objects. Checks `hasattr(model, 'named_parameters')` first (works for direct models),
  then tries wrapper attributes `['_model', 'model', '_executor', '_engine']`.

- **`_collect_model_files()`**: Gathers `config.json`, `tokenizer.json`, etc. for the
  target's config loading. If `model_path` is set, reads from that directory directly
  (reliable). Otherwise falls back to searching the HF cache with glob (fragile when
  multiple models are cached — see §13.3).

### 5.2 Source Deployment for TP>1 (MPI)

For TP=8, the source must create 8 model shards, each on a different GPU. TRT-LLM's
model creation uses CUDA IPC for AllReduce workspace allocation, which requires all ranks
to participate simultaneously. A single-process sequential approach fails with
`cudaErrorDeviceUninitialized` because CUDA IPC expects peer processes.

**Solution**: Use `mpirun -np 8` to spawn 8 MPI workers in parallel:

```bash
mpirun --allow-run-as-root -np 8 python3 /tmp/source_tp8.py
```

Each worker:
1. `rank = MPI.COMM_WORLD.Get_rank()`
2. `torch.cuda.set_device(rank)`
3. `mapping = Mapping(world_size=8, tp_size=8, rank=rank)`
4. `config = HfConfigLoader().load(local_path, mapping=mapping, trust_remote_code=True)`
5. Creates model, loads weights via `HfCheckpointLoader` + weight mapper
6. `MxLiveSource(model, model_name, mx_server).publish()`
7. `comm.Barrier()` — wait for all ranks
8. Sleep forever (params must stay in GPU memory for RDMA)

### 5.3 Why MPI is Required for Source but Not Target

- **Source**: We create the model using TRT-LLM's internal classes
  (`AutoModelForCausalLM.from_config`), which include `AllReduce` modules that require
  multi-process CUDA IPC during construction. Without MPI, CUDA IPC handshake fails.

- **Target**: Uses `LLM()` wrapper, which handles MPI spawning internally. We pass the
  `MxLiveCheckpointLoader` as a parameter, and TRT-LLM takes care of the rest.

### 5.4 Source TP=1 (Simpler Path)

For TP=1, no MPI is needed. The source script uses `python3 -c` directly:

```python
config = ModelConfig.from_pretrained(local_path)
model = AutoModelForCausalLM.from_config(config)
model.to("cuda")
weights = HfCheckpointLoader().load_weights(local_path, mapping=Mapping(1,1,0))
model.load_weights(weights, weight_mapper=...)
source = MxLiveSource(model, model_name, mx_server)
source.publish()
```

---

## 6. Implementation: Target Side

### 6.1 `MxLiveWeightLoader` Class

**File**: `modelexpress_client/python/modelexpress/trtllm_live_transfer.py`

The core of Phase 3. Called inside each TRT-LLM MPI worker during model loading.

**`load_weights(checkpoint_dir, mapping, model, **kwargs)`**:

1. **Validate model reference**: `model` must be provided (via PRESHARDED path in
   `model_loader.py`). Raises `RuntimeError` if `None`.

2. **Set up file logging**: MPI workers' stdout is swallowed by TRT-LLM's process
   management. Writes per-rank logs to `/tmp/mx_logs/rank{N}.log` so transfer stats
   can be retrieved after loading (see §15).

3. **Query source metadata**: `_query_source(mx_server, model_name, timeout=600)` polls
   the MX server every 5 seconds until source metadata is found. Returns the full
   `GetMetadataResponse` including all workers' tensor descriptors and NIXL metadata.

4. **Find matching source worker**: Filters `source_meta.workers` by
   `worker_rank == device_id`. Each target rank connects only to its corresponding
   source rank.

5. **Build target param map**: Iterates `model.named_parameters()`, filters by
   `param.device.index == device_id`. Builds `{name: param.data}` dict.

6. **Name matching**: For each source tensor descriptor, checks if the name exists in
   the target param map and sizes match. Builds `matched` list of
   `(name, source_descriptor, target_param)` tuples.

7. **NIXL setup**: Creates `NixlTransferManager` for this rank, registers all target
   param buffers. Adds the source's remote NIXL agent using its serialized metadata.

8. **RDMA transfer**: Calls `nixl_mgr.receive_from_source()` which:
   - Creates NIXL transfer descriptors mapping source addresses to target addresses
   - Issues RDMA GET operations (one per tensor, or coalesced)
   - Waits for all transfers to complete
   - Returns `(bytes_transferred, n_tensors, elapsed)`

9. **Cleanup**: Shuts down NIXL manager, removes file handler.

10. **Return `{}`**: Empty dict signals to `model_loader.py` that weights are already
    in model params.

### 6.2 `MxLiveCheckpointLoader` Class

Wraps `MxLiveWeightLoader` (for weights) and `MxConfigLoader` (for config) into a
single object compatible with TRT-LLM's `checkpoint_loader` interface.

```python
class MxLiveCheckpointLoader:
    def __init__(self):
        self._weight_loader = MxLiveWeightLoader()
        self._config_loader = None   # Lazy-init MxConfigLoader
        self._weight_mapper = None
        self._checkpoint_format = "mx-p2p"

    def load_config(self, checkpoint_dir, **kwargs):
        return self.config_loader.load(checkpoint_dir, **kwargs)

    def load_weights(self, checkpoint_dir, mapping=None, model=None, **kwargs):
        return self._weight_loader.load_weights(
            checkpoint_dir, mapping=mapping, model=model, **kwargs
        )

    def get_initialized_weight_mapper(self, model, config):
        # Still needed even for empty weights — model_loader.py uses it
        weight_mapper = AutoCheckpointMapper.get("HF", model_arch)
        weight_mapper.init_model_and_config(model, config)
        return weight_mapper
```

**Pickling requirement**: TRT-LLM pickles the `checkpoint_loader` object and sends it to
MPI workers. Both `MxLiveCheckpointLoader` and `MxLiveWeightLoader` must be defined at
**module level** (not inside a function) to be picklable. They have no unpicklable
attributes (no sockets, no file handles at init time).

### 6.3 `MxConfigLoader` (Reused from Phase 2)

**File**: `modelexpress_client/python/modelexpress/trtllm_checkpoint_loader.py`

Fetches model configuration files from the MX server instead of from disk. This
eliminates the need for a PVC or init container on the target node.

1. Queries MX server `GetMetadata` for `model_files` field
2. Writes `config.json`, `tokenizer.json`, etc. to a temp directory
3. Delegates to `HfConfigLoader().load(temp_dir, **kwargs)` for parsing

Falls back to local HF path if the MX server has no `model_files`.

---

## 7. TRT-LLM Upstream Changes

Phase 3 requires 3 patched TRT-LLM files. These are applied as Docker overlay copies
(see §8) and are intended for upstream contribution.

### 7.1 `llm_args.py` — `LoadFormat.PRESHARDED` Enum

**File**: `tensorrt_llm/llmapi/llm_args.py`

```python
class LoadFormat(Enum):
    AUTO = 0
    DUMMY = 1
    VISION_ONLY = 2
    PRESHARDED = 3   # ← NEW
```

Signals that weights are already per-rank sharded. Used by both Phase 2 (HF-format
pre-sharded) and Phase 3 (TRT-LLM-format direct injection).

### 7.2 `model_loader.py` — PRESHARDED Branch + Empty Dict Handling

**File**: `tensorrt_llm/_torch/pyexecutor/model_loader.py`

The `ModelLoader.load()` method gains a `PRESHARDED` branch:

```python
elif load_format == LoadFormat.PRESHARDED:
    from tensorrt_llm._torch.modules.linear import Linear
    for m in model.modules():
        if isinstance(m, Linear):
            m._weights_presharded = True

    weights = checkpoint_loader.load_weights(
        ckpt_dir, mapping=self.mapping, model=model)  # ← passes model ref

    if weights:
        # Phase 2: HF-format pre-sharded weights — mapper still fuses
        self.weight_mapper = checkpoint_loader.get_initialized_weight_mapper(model, config)
        self._call_load_weights(model.load_weights, weights, self.weight_mapper)
    else:
        # Phase 3: Empty dict — weights injected directly via RDMA
        logger.info("PRESHARDED: weights already in model params (direct injection)")
        self.weight_mapper = checkpoint_loader.get_initialized_weight_mapper(model, config)
```

Key changes vs `AUTO`:
1. **Sets `_weights_presharded = True`** on all Linear modules (Phase 2 feature)
2. **Passes `model=model`** to `load_weights()` so the loader can access target params
3. **Handles empty dict**: If loader returns `{}`, skips `model.load_weights()` entirely

### 7.3 `linear.py` — Pre-sharded TP Skip

**File**: `tensorrt_llm/_torch/modules/linear.py`

Three weight loading helpers check `_weights_presharded` and pass `tp_size=1` to
`load_weight_shard()`, which makes the existing early-return path kick in (TP slicing
is skipped when `tp_size <= 1`):

```python
# In load_weights_vanilla_helper, load_weights_fused_qkv_helper,
# load_weights_fused_gate_up_helper:
tp_size = 1 if getattr(module, '_weights_presharded', False) else module.tp_size
```

This change is only active for Phase 2 (when the loader returns non-empty HF-format
pre-sharded weights). In Phase 3, the loader returns `{}` so these helpers are never
called. However, the flag is still set for forward compatibility.

### 7.4 Diff Summary

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `llm_args.py` | +4 | Add `PRESHARDED = 3` enum |
| `model_loader.py` | +20 | PRESHARDED branch, model ref, empty dict |
| `linear.py` | +6 (×3 helpers) | Skip TP slicing when `_weights_presharded` |
| **Total** | **~30 lines** | Fully backward compatible |

See `docs/TRTLLM_UPSTREAM_DIFF.md` for the complete annotated diff.

---

## 8. Docker Image Build

### 8.1 Base Image

```dockerfile
FROM nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6
```

The TRT-LLM NGC release image provides:
- CUDA 13 runtime
- TensorRT 10.x
- OpenMPI + HPC-X (with UCX)
- System NIXL at `/opt/nvidia/nvda_nixl/`
- Python 3.12

### 8.2 NIXL Installation (System NIXL, Not Pip)

```dockerfile
RUN pip install --no-cache-dir --no-deps nixl && \
    ln -sf /opt/nvidia/nvda_nixl/lib/python3/dist-packages/nixl_cu13 \
           /usr/local/lib/python3.12/dist-packages/nixl_cu13 && \
    echo "/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu" > /etc/ld.so.conf.d/nixl-system.conf && \
    ldconfig
```

**Critical**: We use the **system NIXL** that ships with the TRT-LLM image, NOT
`pip install nixl[cu13]`. The pip version bundles its own UCX libraries which conflict
with HPC-X UCX used by MPI. This conflict causes `NIXL_ERR_REMOTE_DISCONNECT` and
segfaults in `uct_md_query_tl_resources`.

The `pip install --no-deps nixl` installs only the Python wrapper. The symlink points
the CUDA 13 native extension to the system-installed version. The `ldconfig` ensures
the system NIXL shared libraries are discoverable.

### 8.3 ModelExpress Client

```dockerfile
COPY modelexpress_common/proto /opt/modelexpress/proto
COPY modelexpress_client/python/modelexpress /opt/modelexpress/client/modelexpress
COPY modelexpress_client/python/pyproject.toml /opt/modelexpress/client/

RUN python3 -m grpc_tools.protoc \
        -I/opt/modelexpress/proto \
        --python_out=/opt/modelexpress/client/modelexpress \
        --grpc_python_out=/opt/modelexpress/client/modelexpress \
        /opt/modelexpress/proto/p2p.proto && \
    sed -i 's/^import p2p_pb2/from . import p2p_pb2/' \
        /opt/modelexpress/client/modelexpress/p2p_pb2_grpc.py

RUN pip install --no-deps .
```

This installs the full ModelExpress Python client, including:
- `trtllm_live_transfer.py` — Phase 3 classes (`MxLiveSource`, `MxLiveWeightLoader`, `MxLiveCheckpointLoader`)
- `trtllm_checkpoint_loader.py` — Phase 2 classes (`MxCheckpointLoader`, `MxConfigLoader`)
- `trtllm_loader.py` — Phase 1 POC classes
- `nixl_transfer.py` — NIXL transfer manager
- `client.py` — Base MX client
- `p2p_pb2.py`, `p2p_pb2_grpc.py` — Generated gRPC stubs

### 8.4 TRT-LLM Patches

```dockerfile
COPY trtllm_patches/llm_args.py    /usr/local/lib/python3.12/.../tensorrt_llm/llmapi/llm_args.py
COPY trtllm_patches/model_loader.py /usr/local/lib/python3.12/.../tensorrt_llm/_torch/pyexecutor/model_loader.py
COPY trtllm_patches/linear.py      /usr/local/lib/python3.12/.../tensorrt_llm/_torch/modules/linear.py
```

These overlay the 3 patched files directly into the installed TRT-LLM package. This
approach avoids forking TRT-LLM and allows easy diffing against the upstream version.

### 8.5 Build Command

```bash
cd /path/to/modelexpress
docker build -t nvcr.io/nvidian/dynamo-dev/modelexpress-trtllm-client:ph3 \
    -f examples/p2p_transfer_trtllm/Dockerfile.trtllm .
docker push nvcr.io/nvidian/dynamo-dev/modelexpress-trtllm-client:ph3
```

Build context is the modelexpress repo root. Build time is ~2 minutes (most layers cached).

---

## 9. Kubernetes Deployment

### 9.1 Source Deployment (Llama 70B, TP=8)

**File**: `examples/p2p_transfer_trtllm/deploy/trtllm-source-70b-ph3.yaml`

Key configuration:
- **Image**: `modelexpress-trtllm-client:ph3`
- **Resources**: 8 GPUs, 8 RDMA devices, 200Gi memory, 16 CPUs
- **Volumes**: `/dev/shm` (64Gi tmpfs), `/models` (PVC for HF cache)
- **Command**: `bash -c` → writes Python script to `/tmp/source_tp8.py`, then
  `mpirun --allow-run-as-root -np 8 python3 /tmp/source_tp8.py`

Environment variables:
| Variable | Value | Purpose |
|----------|-------|---------|
| `MODEL_NAME` | `meta-llama/Llama-3.1-70B-Instruct` | HF model ID |
| `MODEL_EXPRESS_URL` | `modelexpress-server:8001` | MX server gRPC |
| `HF_HUB_CACHE` | `/models` | HF cache on PVC |
| `TP_SIZE` | `8` | Number of MPI ranks |
| `UCX_TLS` | `rc_x,rc,dc_x,dc,cuda_copy` | UCX transports |
| `UCX_RNDV_SCHEME` | `get_zcopy` | Zero-copy rendezvous |
| `UCX_RNDV_THRESH` | `0` | Always use rendezvous |

### 9.2 Target Deployment (Llama 70B, TP=8)

**File**: `examples/p2p_transfer_trtllm/deploy/trtllm-target-70b-ph3.yaml`

Key configuration:
- **Image**: Same `modelexpress-trtllm-client:ph3`
- **Resources**: 8 GPUs, 8 RDMA devices, 200Gi memory, 16 CPUs
- **No PVC needed**: Config comes from MX server, weights come via RDMA
- **Pod anti-affinity**: `topologyKey: kubernetes.io/hostname` against
  `app: trtllm-source-70b` — ensures source and target are on different nodes
  (meaningful cross-node RDMA, not intra-node loopback)

The target script is simple:
```python
from modelexpress.trtllm_live_transfer import MxLiveCheckpointLoader
loader = MxLiveCheckpointLoader()

llm = LLM(
    model=model_name,
    checkpoint_loader=loader,
    load_format=LoadFormat.PRESHARDED,
    tensor_parallel_size=8,
)
```

### 9.3 TP=1 Deployments

Simpler variants exist for Qwen 0.5B (TP=1):
- `trtllm-source-ph3.yaml` — single GPU, `python3 -c` (no MPI needed)
- `trtllm-target-ph3.yaml` — single GPU, same `MxLiveCheckpointLoader`

---

## 10. Name Matching: How Source and Target Params Align

Both source and target create models via `AutoModelForCausalLM.from_config(config)` with
identical architecture and TP mapping. TRT-LLM produces identical `named_parameters()`
names and shapes.

### 10.1 Example: Llama 70B TP=8 (Per Rank)

```
Name                                                 Shape           Bytes
transformer.vocab_embedding.weight                   [16000, 8192]   262.1M
transformer.layers.0.input_layernorm.weight          [8192]          16.4K
transformer.layers.0.attention.qkv_proj.weight       [1280, 8192]    21.0M
transformer.layers.0.attention.dense.weight          [8192, 1024]    16.8M
transformer.layers.0.post_layernorm.weight           [8192]          16.4K
transformer.layers.0.mlp.fc.weight                   [7168, 8192]    117.4M
transformer.layers.0.mlp.proj.weight                 [8192, 3584]    58.7M
...  (×80 layers)
transformer.ln_f.weight                              [8192]          16.4K
lm_head.weight                                       [16000, 8192]   262.1M
─────────────────────────────────────────────────────────────────────────
Total: 483 params, 17.64 GB per rank
```

### 10.2 Why 100% Match Rate

- Same `AutoModelForCausalLM` factory
- Same `ModelConfig` (from identical `config.json`)
- Same `Mapping(world_size=8, tp_size=8, rank=N)` for each rank pair
- Same TRT-LLM version (both use `1.2.0rc6`)

The name matching is a simple dict lookup:
```python
source_descs = {t.name: t for t in source_worker.tensors}
for src_name, src_desc in source_descs.items():
    if src_name in target_params:
        matched.append(...)
```

No fuzzy matching, no regex, no HF↔TRT-LLM translation needed.

---

## 11. Performance Results

### 11.1 Llama 3.1 70B Instruct (TP=8, BF16, Cross-Node IB)

Measured February 19, 2026 on two DGX H100 nodes connected via InfiniBand.

**Per-rank RDMA transfer:**

| Rank | Params | Size | Transfer Time | Bandwidth |
|------|--------|------|---------------|-----------|
| 0 | 483 | 17.64 GB | 3.06s | 46.2 Gbps |
| 1 | 483 | 17.64 GB | 3.06s | 46.1 Gbps |
| 2 | 483 | 17.64 GB | 3.10s | 45.5 Gbps |
| 3 | 483 | 17.64 GB | 3.11s | 45.4 Gbps |
| 4 | 483 | 17.64 GB | 3.06s | 46.0 Gbps |
| 5 | 483 | 17.64 GB | 3.07s | 46.0 Gbps |
| 6 | 483 | 17.64 GB | 3.08s | 45.8 Gbps |
| 7 | 483 | 17.64 GB | 3.08s | 45.8 Gbps |

**Aggregate:**

| Metric | Value |
|--------|-------|
| Total data | 141.1 GB |
| Wall-clock RDMA time | ~3.08s (all ranks parallel) |
| Aggregate bandwidth | ~368 Gbps |
| Per-rank avg bandwidth | ~45.9 Gbps |
| Source publish time | 17.0s (MPI model load + NIXL register) |
| Target total load time | 141.6s |
| RDMA as % of target load | ~2.2% |

**Target load time breakdown (approximate):**

| Phase | Time | Notes |
|-------|------|-------|
| Config loading + model creation | ~10s | MxConfigLoader + meta init + CUDA |
| NIXL setup + metadata query | ~3s | Agent init, register, remote agent |
| **RDMA transfer** | **~3.1s** | 141.1 GB across 8 ranks |
| NIXL teardown | ~1s | Deregister, shutdown |
| post_load_weights() | ~1s | Weight finalization |
| KV cache allocation | ~30s | 102.86 GiB across 8 GPUs |
| Engine compilation / warmup | ~90s | TRT-LLM PyTorch engine init |
| **Total** | **~141.6s** | |

The RDMA transfer is not the bottleneck — engine compilation dominates.

### 11.2 Qwen 2.5 0.5B (TP=1, BF16, Cross-Node IB)

| Metric | Value |
|--------|-------|
| Total data | 1.26 GB |
| Params | 171 |
| RDMA transfer time | 0.10s |
| Bandwidth | 102.4 Gbps |
| Total load time | 31.13s |

Higher per-rank bandwidth than 70B because fewer concurrent NIXL connections (1 vs 8).

### 11.3 Inference Validation

Both models produced coherent inference output after Phase 3 transfer:

**Llama 70B**: *"Meta AI, and I can provide information and entertainment, but I can't
currently take actions on your behalf. For example, I can plan a custom travel itinerary,
but I can't buy tickets or book hotels. I can write you an email,"*

**Qwen 0.5B**: Generated text (expected lower quality from 0.5B model at temperature=0.7).

---

## 12. Comparison: Phase 2 vs Phase 3

### 12.1 Data Path Comparison

```
Phase 2 (PRESHARDED + HF format):
  Source:  Disk (HF safetensors)
              → CPU memory (load all shards)
              → Per-rank TP slicing (tensor.narrow)
              → NIXL register GPU buffers
              → gRPC publish

  Target:  gRPC query
              → NIXL RDMA receive (HF-format tensors)
              → .cpu() copy (workaround: NIXL shutdown invalidates GPU)
              → HfWeightMapper fuse (q+k+v → qkv, gate+up → gate_up)
              → copy_weight() into model params

Phase 3 (live model P2P):
  Source:  model.named_parameters()
              → NIXL register existing GPU buffers (zero-copy)
              → gRPC publish

  Target:  gRPC query
              → NIXL RDMA directly into model.named_parameters() buffers
              → return {} — done
```

### 12.2 What's Eliminated

| Step | Phase 2 | Phase 3 |
|------|---------|---------|
| Source reads HF files from disk | Yes (~30s for 70B) | **No** (live model) |
| Source shards tensors | Yes (`tensor.narrow`) | **No** (already sharded) |
| HF → TRT-LLM name mapping | Yes (`HfWeightMapper`) | **No** (same names) |
| Fuse q+k+v → qkv | Yes (`torch.cat`) | **No** (already fused) |
| Intermediate GPU buffer allocation | Yes (~21 GB/rank) | **No** (direct to params) |
| `.cpu()` round-trip | Yes (~0.5s) | **No** |
| `copy_weight()` | Yes (pointer swap) | **No** |
| `model.load_weights()` call | Yes (full pass) | **No** (skipped, empty dict) |
| RDMA transfer | Yes (~3s/rank) | Yes (~3s/rank) — network bound |

### 12.3 What's Still Required

- NIXL agent initialization and tensor registration (~3s total)
- MX server metadata query (~0.5s)
- `post_load_weights()` (~1s)
- KV cache allocation (~30s)
- Engine compilation (~90s)

The last two dominate total load time and are TRT-LLM-internal costs unrelated to
weight transfer.

---

## 13. Issues Encountered and Resolved

### 13.1 Frozen `ModelConfig` — Cannot Set `mapping` Attribute

**Symptom**: `AttributeError: Cannot modify ModelConfig.'mapping' - instance is frozen`

**Root cause**: TRT-LLM's `ModelConfig` is a frozen dataclass. Setting `config.mapping`
directly after `ModelConfig.from_pretrained()` is not allowed.

**Fix**: Use `HfConfigLoader().load(local_path, mapping=mapping, trust_remote_code=True)`
which creates the `ModelConfig` with the mapping already set during construction.

### 13.2 CUDA IPC Error with Single-Process TP>1

**Symptom**: `RuntimeError: CUDA Runtime API error: cudaErrorDeviceUninitialized: 201`
when creating a TP=8 model in a single process.

**Root cause**: TRT-LLM's `AllReduce` module allocates IPC shared memory
(`IpcMemory.open_ipc_memory`) that requires all TP ranks to participate simultaneously
from separate processes. A single process cannot satisfy the multi-process handshake.

**Fix**: Use `mpirun -np 8` for the source, so all 8 ranks create their models in parallel.
Each rank is a separate process, satisfying CUDA IPC requirements.

### 13.3 Wrong `config.json` from HF Cache Glob

**Symptom**: Target creates a **Qwen** model instead of Llama. Error:
`AssertionError: num_heads % tp_size == 0` from `modeling_qwen.py`.

**Root cause**: `MxLiveSource._collect_model_files()` used `glob.glob()` to find
`config.json` in the HF cache, taking `matches[0]`. When multiple models were cached
(Qwen from TP=1 test + Llama from TP=8 test), the glob returned Qwen's config first
(alphabetical ordering: `Q` < `m`).

**Fix**: Added `model_path` parameter to `MxLiveSource.__init__()`. When set, reads
config files directly from the specific model directory instead of globbing the entire
cache. In the deployment YAML, a monkey-patch overrides `_collect_model_files()` to
use the known `local_path`:

```python
source = MxLiveSource(model, model_name, mx_server)
def _collect_from_path():
    model_files = {}
    for fname in ["config.json", "tokenizer.json", ...]:
        fpath = os.path.join(local_path, fname)
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                model_files[fname] = f.read()
    return model_files
source._collect_model_files = _collect_from_path
```

### 13.4 MPI Worker Logs Swallowed by TRT-LLM

**Symptom**: `kubectl logs` showed only the main process output (10 lines). All per-rank
transfer stats (bandwidth, timing) were invisible.

**Root cause**: TRT-LLM's `LLM()` spawns MPI workers as child processes. Their stdout/
stderr is not forwarded to the main process's stdout. `kubectl logs` only captures the
main container's stdout.

**Fix**: Added per-rank file logging in `MxLiveWeightLoader.load_weights()`:
```python
log_dir = os.environ.get("MX_TRANSFER_LOG_DIR", "/tmp/mx_logs")
os.makedirs(log_dir, exist_ok=True)
rank_log = os.path.join(log_dir, f"rank{device_id}.log")
fh = logging.FileHandler(rank_log, mode="w")
logging.getLogger("modelexpress").addHandler(fh)
```

The target YAML reads and prints these files after model loading:
```python
for log_file in sorted(glob.glob(f"{log_dir}/rank*.log")):
    with open(log_file) as f:
        for line in f:
            print(line.rstrip())
```

Log files can also be retrieved via `kubectl exec`:
```bash
kubectl exec -n kavin deploy/trtllm-target-70b -- cat /tmp/mx_logs/rank0.log
```

### 13.5 UCX Library Conflict (Inherited from Phase 1/2)

**Symptom**: `NIXL_ERR_REMOTE_DISCONNECT` or segfault in `uct_md_query_tl_resources`.

**Root cause**: Pip-installed NIXL bundles its own UCX. TRT-LLM's MPI uses HPC-X UCX.
Two UCX installations in the same process conflict.

**Fix**: Use system NIXL from the NGC image (same UCX as MPI). See §8.2.

---

## 14. MX Server Behavior for TP>1

### 14.1 Atomic Worker Merge

When 8 source ranks publish metadata concurrently, the MX server uses a **Redis Lua
script** for atomic read-modify-write:

1. Read existing workers for this model
2. For each new worker, find existing worker with same `worker_rank`:
   - If found: update in place
   - If not found: append
3. Sort workers by rank
4. Write back atomically

This ensures no lost updates when multiple ranks publish simultaneously.

### 14.2 Config File Merge

The `model_files` field (config.json, tokenizer files) is also merged atomically.
All 8 ranks publish the same config files; subsequent writes overwrite per-key,
resulting in the correct final state.

### 14.3 In-Memory vs Redis Backend

The MX server supports multiple backends:
- **In-memory** (default): Fast, non-persistent. Source metadata lost on server restart.
- **Redis**: Persistent, supports the Lua atomic merge script.
- **Layered**: In-memory cache + Redis write-through.

For Phase 3 testing, we use the layered backend with Redis.

---

## 15. Observability: Per-Rank Transfer Logging

### 15.1 Log Files

Each MPI worker writes to `/tmp/mx_logs/rank{N}.log`. Example content:

```
2026-02-19 01:04:09,607 modelexpress.trtllm_live_transfer INFO Live transfer: loading 'meta-llama/Llama-3.1-70B-Instruct' into model on GPU 0
2026-02-19 01:04:10,118 modelexpress.trtllm_live_transfer INFO Found source: 8 workers
2026-02-19 01:04:10,120 modelexpress.trtllm_live_transfer INFO Target has 483 params on GPU 0
2026-02-19 01:04:10,121 modelexpress.trtllm_live_transfer INFO Matched 483/483 params for direct RDMA transfer
2026-02-19 01:04:10,662 modelexpress.nixl_transfer INFO NIXL agent 'trtllm-live-target-rank0-286' created on device 0
2026-02-19 01:04:12,441 modelexpress.nixl_transfer INFO Registered 483 individual tensors with NIXL
2026-02-19 01:04:12,449 modelexpress.nixl_transfer INFO Added remote agent b'trtllm-live-source-rank0-77'
2026-02-19 01:04:12,449 modelexpress.nixl_transfer INFO [Coalesce] DISABLED - transferring 483 individual tensors
2026-02-19 01:04:15,498 modelexpress.nixl_transfer INFO Transfer complete: 483 tensors, 17.64 GB in 3.06s (46.2 Gbps)
2026-02-19 01:04:15,499 modelexpress.trtllm_live_transfer INFO Rank 0: transferred 483 params (17.64 GB) in 3.06s (46.2 Gbps) — DIRECT into model params
2026-02-19 01:04:16,150 modelexpress.nixl_transfer INFO NixlTransferManager shutdown complete
```

### 15.2 Retrieving Logs

```bash
# From the running target pod:
for r in 0 1 2 3 4 5 6 7; do
    echo "=== RANK $r ==="
    kubectl exec -n kavin deploy/trtllm-target-70b -- \
        grep "transferred" /tmp/mx_logs/rank${r}.log
done
```

### 15.3 Log Directory Override

Set `MX_TRANSFER_LOG_DIR` environment variable to change the log directory:
```yaml
env:
  - name: MX_TRANSFER_LOG_DIR
    value: "/var/log/mx_transfer"
```

---

## 16. Open Items and Future Work

### 16.1 Resolved (from Original Plan)

| Question | Resolution |
|----------|------------|
| Does `model.load_weights({})` work? | Yes — PRESHARDED branch checks `if weights:` and skips entirely |
| Does NIXL registration affect inference? | No — source continues serving; weights are read-only during inference |
| Empty dict assertions? | None — handled by the PRESHARDED empty-dict branch |
| Mixed source modes (HF vs live)? | Separate loaders: `MxCheckpointLoader` (Phase 2) vs `MxLiveCheckpointLoader` (Phase 3) |

### 16.2 Active Items

1. **Source from running `LLM()` wrapper**: Currently the source uses
   `AutoModelForCausalLM.from_config()` + `HfCheckpointLoader` (loads HF weights into
   TRT-LLM model). Ideally, the source would be an existing running inference instance.
   Extracting the torch model from `LLM()` wrapper is non-trivial — `_get_torch_model()`
   tries several attribute paths but the internal API is not stable.

2. **Upstream PR**: The 3 TRT-LLM patches should be submitted as an upstream PR. The
   changes are backward compatible and total ~30 lines.

3. **Descriptor coalescing**: Currently disabled (`MX_COALESCE_TRANSFERS=0`) due to
   `NIXL_ERR_NOT_FOUND` from `prepXferDlist` when enabled. Individual tensor transfers
   still achieve high bandwidth (46 Gbps/rank). Enabling coalescing could reduce NIXL
   overhead for models with many small tensors.

4. **Memory pinning lifecycle**: NIXL pins model param memory for RDMA. Need to verify
   that TRT-LLM never reallocates or moves parameter buffers after model creation. Current
   testing suggests this is safe (params are allocated once and never moved).

5. **Stale metadata**: If the source restarts, old metadata (with invalid GPU addresses)
   remains in the MX server. The target may connect to stale addresses, causing
   `NIXL_ERR_REMOTE_DISCONNECT`. A session-based invalidation mechanism is needed.

6. **First-instance bootstrap**: The first TRT-LLM instance in a cluster has no source.
   It must load from disk (NVMe/PVC) normally, then become a live source for subsequent
   instances. This "auto-config" pattern requires orchestration (e.g., Dynamo scheduler).

7. **KV cache transfer**: Phase 3 transfers model weights only. For inference-state
   migration (e.g., live request draining), KV cache transfer would also be needed.

### 16.3 Performance Optimization Opportunities

1. **Parallel model creation + RDMA**: Currently model creation (meta init → CUDA) happens
   before RDMA. If the model were created incrementally (layer by layer), RDMA could start
   for completed layers while later layers are still being created.

2. **Warm NIXL agents**: Currently NIXL agents are created and destroyed per transfer.
   Persistent agents would eliminate the ~3s setup overhead.

3. **Reduce engine compilation**: The ~90s engine compilation dominates load time. TRT-LLM's
   `engine_cache` feature could cache compiled engines, reducing subsequent loads to seconds.

---

## Appendix A: File Map

| File | Purpose |
|------|---------|
| `modelexpress_client/python/modelexpress/trtllm_live_transfer.py` | Phase 3 classes: `MxLiveSource`, `MxLiveWeightLoader`, `MxLiveCheckpointLoader` |
| `modelexpress_client/python/modelexpress/trtllm_checkpoint_loader.py` | Phase 2 classes: `MxCheckpointLoader`, `MxConfigLoader`, `MxWeightLoader` |
| `modelexpress_client/python/modelexpress/nixl_transfer.py` | `NixlTransferManager` — NIXL agent lifecycle, tensor registration, RDMA |
| `modelexpress_client/python/modelexpress/trtllm_loader.py` | Phase 1 POC: `MxTrtllmSourcePublisher`, `MxTrtllmTargetLoader` |
| `trtllm_patches/llm_args.py` | Patched `LoadFormat` enum (adds `PRESHARDED`) |
| `trtllm_patches/model_loader.py` | Patched `ModelLoader.load()` (PRESHARDED branch) |
| `trtllm_patches/linear.py` | Patched weight helpers (skip TP slicing) |
| `examples/p2p_transfer_trtllm/Dockerfile.trtllm` | Docker image build |
| `examples/p2p_transfer_trtllm/deploy/trtllm-source-ph3.yaml` | Source: Qwen 0.5B TP=1 |
| `examples/p2p_transfer_trtllm/deploy/trtllm-target-ph3.yaml` | Target: Qwen 0.5B TP=1 |
| `examples/p2p_transfer_trtllm/deploy/trtllm-source-70b-ph3.yaml` | Source: Llama 70B TP=8 (MPI) |
| `examples/p2p_transfer_trtllm/deploy/trtllm-target-70b-ph3.yaml` | Target: Llama 70B TP=8 |
| `modelexpress_common/proto/p2p.proto` | gRPC proto (TensorDescriptor, WorkerMetadata, model_files) |
| `modelexpress_server/src/state.rs` | MX server state: atomic worker merge (Lua script) |
| `modelexpress_server/src/p2p_service.rs` | gRPC service: PublishMetadata, GetMetadata |
| `docs/TRTLLM_UPSTREAM_DIFF.md` | Annotated diff of the 3 TRT-LLM patches |

## Appendix B: Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_NAME` | (required) | HuggingFace model ID |
| `MODEL_EXPRESS_URL` | `localhost:8001` | MX server gRPC endpoint |
| `TP_SIZE` | `8` | Tensor parallel size (source MPI) |
| `HF_HUB_CACHE` | `/models` | HuggingFace cache directory |
| `HF_TOKEN` | (from secret) | HuggingFace auth token |
| `MX_TRANSFER_LOG_DIR` | `/tmp/mx_logs` | Per-rank log file directory |
| `MX_COALESCE_TRANSFERS` | `0` | Enable NIXL descriptor coalescing |
| `NIXL_LOG_LEVEL` | `INFO` | NIXL library log level |
| `UCX_LOG_LEVEL` | `WARN` | UCX library log level |
| `UCX_TLS` | `rc_x,rc,dc_x,dc,cuda_copy` | UCX transport selection |
| `UCX_RNDV_SCHEME` | `get_zcopy` | UCX rendezvous protocol |
| `UCX_RNDV_THRESH` | `0` | UCX rendezvous threshold (0 = always) |

## Appendix C: How to Reproduce

### Prerequisites

- Two nodes with H100 GPUs and InfiniBand (for cross-node RDMA)
- MicroK8s or Kubernetes cluster with GPU operator and RDMA device plugin
- Namespace `kavin` (or modify YAMLs)
- `nvcr-imagepullsecret` for pulling from `nvcr.io/nvidian/`
- `hf-token-secret` with `HF_TOKEN` key for HuggingFace model access
- PVC `model-weights-storage` with Llama 70B pre-downloaded (source needs this)
- MX server + Redis already deployed

### Step 1: Build and Push the Docker Image

```bash
cd /path/to/modelexpress
docker build -t nvcr.io/nvidian/dynamo-dev/modelexpress-trtllm-client:ph3 \
    -f examples/p2p_transfer_trtllm/Dockerfile.trtllm .
docker push nvcr.io/nvidian/dynamo-dev/modelexpress-trtllm-client:ph3
```

### Step 2: Deploy Source (Llama 70B, TP=8)

```bash
kubectl apply -f examples/p2p_transfer_trtllm/deploy/trtllm-source-70b-ph3.yaml -n kavin
# Wait ~2–3 minutes for all 8 ranks to load and publish
kubectl logs -n kavin -l app=trtllm-source-70b --tail=5
# Should see: "All 8 ranks published in XXs — serving model params via NIXL"
```

### Step 3: Deploy Target (Llama 70B, TP=8)

```bash
kubectl apply -f examples/p2p_transfer_trtllm/deploy/trtllm-target-70b-ph3.yaml -n kavin
# Wait ~3 minutes for RDMA transfer + engine compilation
kubectl logs -n kavin -l app=trtllm-target-70b --tail=10
# Should see: "SUCCESS: Llama 70B Phase 3 (live model P2P, direct GPU injection, TP=8)!"
```

### Step 4: Read Per-Rank Transfer Logs

```bash
for r in 0 1 2 3 4 5 6 7; do
    echo "=== RANK $r ==="
    kubectl exec -n kavin deploy/trtllm-target-70b -- \
        grep "transferred" /tmp/mx_logs/rank${r}.log
done
```

### For TP=1 (Qwen 0.5B)

```bash
kubectl apply -f examples/p2p_transfer_trtllm/deploy/trtllm-source-ph3.yaml -n kavin
# Wait ~1 minute
kubectl apply -f examples/p2p_transfer_trtllm/deploy/trtllm-target-ph3.yaml -n kavin
```

## Appendix D: Inference Validation Output

### Llama 3.1 70B Instruct (TP=8)

```
Prompt:  "Hello, I am a large language model trained by"
Output:  " Meta AI, and I can provide information and entertainment, but I can't
          currently take actions on your behalf. For example, I can plan a custom
          travel itinerary, but I can't buy tickets or book hotels. I can write
          you an email,"
```

Coherent, factually grounded, stylistically consistent with Llama 3.1 70B Instruct.

### Qwen 2.5 0.5B (TP=1)

```
Prompt:  "Hello, I am a large language model"
Output:  (lower quality expected from 0.5B model at temperature=0.7)
```

## Appendix E: Git State and Branch Context

**Branch**: `kavink/trtllm`
**Base**: `main` (after PR #135 merge)

### Uncommitted Files (at time of Phase 3 validation)

**New files (untracked):**
- `docs/TRTLLM_PLAN.md` — Master engineering plan (all phases)
- `docs/TRTLLM_PLAN_PH1.md` — Phase 1 progress tracker (CLOSED)
- `docs/TRTLLM_PLAN_PHASE2.md` — Phase 2 design and results
- `docs/TRTLLM_PLAN_PHASE3.md` — This document
- `docs/TRTLLM_QUESTIONS.md` — Questions for TRT-LLM team
- `docs/TRTLLM_UPSTREAM_DIFF.md` — Annotated diff of 3 TRT-LLM patches
- `modelexpress_client/python/modelexpress/trtllm_checkpoint_loader.py` — Phase 2 loader
- `modelexpress_client/python/modelexpress/trtllm_live_transfer.py` — Phase 3 loader
- `trtllm_patches/` — 3 patched TRT-LLM files (llm_args.py, model_loader.py, linear.py)
- `examples/p2p_transfer_trtllm/deploy/` — All K8s deployment YAMLs
- `examples/p2p_transfer_k8s/deploy/` — Persistence backend YAMLs

**Modified files (tracked):**
- `modelexpress_common/proto/p2p.proto` — Added `model_files` field
- `modelexpress_server/src/state.rs` — Added `model_files` storage + Lua merge
- `modelexpress_server/src/p2p_service.rs` — Added `model_files` handling
- `modelexpress_server/Cargo.toml` — Added `base64` dependency
- `modelexpress_client/python/modelexpress/trtllm_loader.py` — Phase 1 POC updates
- `modelexpress_client/python/modelexpress/p2p_pb2.py` / `p2p_pb2_grpc.py` — Regenerated
- `examples/p2p_transfer_trtllm/Dockerfile.trtllm` — Updated for Phase 3
- `Cargo.lock` — Dependency update

### Related Documents

- `docs/TRTLLM_PLAN.md` — Overall roadmap (Phases 1–4), TRT-LLM plugin system analysis
- `docs/TRTLLM_PLAN_PH1.md` — Phase 1 results: Qwen 0.5B TP=1, 151 Gbps; Llama 70B TP=8 RDMA works but disk still needed
- `docs/TRTLLM_PLAN_PHASE2.md` — Phase 2 results: PRESHARDED validated for TP=8 zero-disk
- `docs/TRTLLM_QUESTIONS.md` — Discussion items for TRT-LLM team (UCX resolved, PRESHARDED = P0)
- `docs/TRTLLM_UPSTREAM_DIFF.md` — The exact 3-file diff for the upstream PR

### Docker Image Tags

| Tag | Contents | Used By |
|-----|----------|---------|
| `modelexpress-trtllm-client:ph2` | Phase 2 loader + TRT-LLM patches | Phase 2 70B test |
| `modelexpress-trtllm-client:ph3` | Phase 3 loader + per-rank logging + TRT-LLM patches | Phase 3 tests |

### Kubernetes Deployments Active (at time of writing)

```
NAMESPACE  NAME                  READY  STATUS   AGE
kavin      modelexpress-server   1/1    Running  5d
kavin      redis                 1/1    Running  6d
kavin      trtllm-source-70b     1/1    Running  ~40m  (Phase 3 source, 8 ranks)
kavin      trtllm-target-70b     1/1    Running  ~35m  (Phase 3 target, inference validated)
```
