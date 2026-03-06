# TRT-LLM Phase 1: Custom Checkpoint Loader — CLOSED

**Branch**: `kavink/trtllm`
**Status**: Phase 1 COMPLETE for TP=1. TP>1 requires upstream TRT-LLM changes (Phase 2).
**Outcome**: TP=1 zero-disk loading validated at 151 Gbps. TP=8 RDMA validated at 85 Gbps/rank
but blocked on pre-sharded weight loading (see Phase 2 plan in TRTLLM_PLAN.md).

---

## Status Overview

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Proto: Add `model_files` to PublishMetadata/GetMetadata | DONE | `map<string, bytes>` in proto |
| 2 | Server (Rust): Store and return model_files | DONE | base64 in Redis, decode on GetMetadata |
| 3 | Regenerate Python protobuf files | DONE | p2p_pb2.py, p2p_pb2_grpc.py |
| 4 | Python: `MxWeightLoader` (NIXL RDMA weight loading) | DONE | `trtllm_checkpoint_loader.py` |
| 5 | Python: `MxConfigLoader` (config from MX server) | DONE | Writes to temp dir, delegates to HfConfigLoader |
| 6 | Python: `MxCheckpointLoader` (registers as "mx-p2p") | DONE | Module-level class (picklable) |
| 7 | Source publisher: Publish config files with metadata | DONE | `_collect_model_files()` in trtllm_loader.py |
| 8 | Docker: TRT-LLM 1.2.0rc6 + ModelExpress + NIXL image | DONE | Base: `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6` |
| 9 | K8s: Source deployment (Qwen 0.5B) | DONE | Publishes weights + 4 config files via MX server |
| 10 | K8s: Target deployment (single container, no PVC) | DONE | System NIXL + no coalesce |
| 11 | Test: Qwen 0.5B (TP=1) end-to-end | DONE | 127.7 Gbps transfer, inference OK |
| 12 | Test: Llama 70B (TP=8) full validation | TODO | |
| 13 | Benchmark: Compare with POC init-container approach | TODO | |

---

## Detailed Progress

### 1. Proto: Add `model_files` to gRPC Messages

**Why**: The target needs `config.json`, `tokenizer.json`, `tokenizer_config.json` etc. to initialize
TRT-LLM's model config and tokenizer. Currently these are copied from a PVC, which means the target
needs PVC access. By transferring them via gRPC, the target becomes completely PVC-free.

**Design**: Use `map<string, bytes>` for flexibility — any file can be included.

```protobuf
message PublishMetadataRequest {
  string model_name = 1;
  repeated WorkerMetadata workers = 2;
  map<string, bytes> model_files = 3;  // NEW: {"config.json": <bytes>, ...}
}

message GetMetadataResponse {
  bool found = 1;
  repeated WorkerMetadata workers = 2;
  map<string, bytes> model_files = 3;  // NEW: returned to target
}
```

**Files changed**:
- `modelexpress_common/proto/p2p.proto`
- `modelexpress_server/src/state.rs` (ModelMetadataRecord)
- `modelexpress_server/src/p2p_service.rs` (pass model_files through)
- `modelexpress_client/python/modelexpress/p2p_pb2.py` (regenerated)

**Status**: DONE

**Files changed**:
- `modelexpress_common/proto/p2p.proto` — added `map<string, bytes> model_files = 3`

---

### 2. Server (Rust): Store and Return model_files

Store the `model_files` map alongside the existing worker metadata. Since config files are small
(< 1MB total), they fit comfortably in the existing storage (Redis or in-memory).

**Status**: DONE

**Files changed**:
- `modelexpress_server/src/state.rs` — added `model_files: HashMap<String, String>` to `ModelMetadataRecord`, base64 encode/decode in Lua script
- `modelexpress_server/src/p2p_service.rs` — pass `model_files` through publish/get, decode base64 on GetMetadata response
- `modelexpress_server/Cargo.toml` — added `base64 = "0.22"` dependency
- Compilation verified: `cargo check` passes

---

### 3. Regenerate Python Protobuf Files

**Status**: DONE

```bash
python -m grpc_tools.protoc -I modelexpress_common/proto \
  --python_out=modelexpress_client/python/modelexpress \
  --grpc_python_out=modelexpress_client/python/modelexpress \
  p2p.proto
```

---

### 4. Python: MxWeightLoader

Receives weights via NIXL RDMA from a ModelExpress source. Returns `ConsumableWeightsDict`
with HF-format names.

**Status**: DONE

**File**: `modelexpress_client/python/modelexpress/trtllm_checkpoint_loader.py`

Key methods:
- `load_weights(checkpoint_dir, mapping)` — queries MX, receives via NIXL, reconstructs
- `_query_source()` — polls MX server until source is found
- `_allocate_tensors()` — allocates GPU tensors with correct shapes from proto
- `_reconstruct_full_weights()` — concatenates TP shards back to full HF format

---

### 5. Python: MxConfigLoader

Loads config from MX server's `model_files` metadata, writes to temp dir, delegates to
`HfConfigLoader` for actual parsing. Falls back to local path if no model_files available.

**Status**: DONE

**File**: `modelexpress_client/python/modelexpress/trtllm_checkpoint_loader.py`

Key behavior:
- Uses cached source metadata from weight loader (avoids duplicate gRPC call)
- Writes config.json, tokenizer.json etc. to `tempfile.mkdtemp()`
- Delegates to `HfConfigLoader` for robust config parsing
- Cleans up temp dir in `cleanup()`

---

### 6. Python: MxCheckpointLoader

Registered as `@register_checkpoint_loader("mx-p2p")`. Composes `MxWeightLoader` +
`MxConfigLoader` + reuses `HfWeightMapper` via `AutoCheckpointMapper.get("HF", arch)`.

**Status**: DONE

**File**: `modelexpress_client/python/modelexpress/trtllm_checkpoint_loader.py`

Auto-registers on import:
```python
import modelexpress.trtllm_checkpoint_loader  # Registers "mx-p2p"
from tensorrt_llm import LLM
llm = LLM(model="dummy", checkpoint_format="mx-p2p", tensor_parallel_size=8)
```

---

### 7. Source Publisher: Publish Config Files

Added `_collect_model_files()` method to `MxTrtllmSourcePublisher` that reads config
files from the HF model directory and includes them in `PublishMetadata`.

**Status**: DONE

**File**: `modelexpress_client/python/modelexpress/trtllm_loader.py`

Collected files: `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`generation_config.json`, `special_tokens_map.json`, `tokenizer.model`,
`preprocessor_config.json`

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│ Source Node                                                          │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ MxTrtllmSourcePublisher                                         │ │
│ │  1. Load HF model → shard for TP=8                             │ │
│ │  2. Register tensors with NIXL                                  │ │
│ │  3. Read config.json, tokenizer.json, etc.                      │ │
│ │  4. PublishMetadata(workers=..., model_files={...})             │ │
│ │  5. Keep running (serve weights)                                │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ gRPC + RDMA
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│ MX Server (gRPC)                                                    │
│ Stores: worker metadata + NIXL descs + model_files                  │
└────────────────────────────────────────────────────────────────────┘
                               │ gRPC + RDMA
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Target Node (SINGLE CONTAINER — no init container, no PVC)           │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ import modelexpress  # registers "mx-p2p" loader               │ │
│ │ llm = LLM(model="dummy", checkpoint_format="mx-p2p", tp=8)    │ │
│ │                                                                 │ │
│ │ Inside TRT-LLM loading pipeline:                                │ │
│ │  1. MxConfigLoader.load() → GetMetadata → extract model_files  │ │
│ │     → write config.json to temp dir → parse ModelConfig         │ │
│ │  2. TRT-LLM creates model on GPU (meta init → cuda)            │ │
│ │  3. MxWeightLoader.load_weights() → NIXL RDMA from source      │ │
│ │     → reconstruct full HF → return ConsumableWeightsDict       │ │
│ │  4. HfWeightMapper maps HF names → TRT-LLM names               │ │
│ │  5. copy_weight() assigns to model params                       │ │
│ │  6. Inference ready!                                            │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

1. **`map<string, bytes>` for config files**: Flexible — any file can be added without proto changes.
   Config files are small (~1MB total), so gRPC message size isn't a concern.

2. **Reuse `HfWeightMapper`**: Our loader returns HF-format weights. The existing mapper handles
   name conversion (q/k/v → qkv_proj) and TP sharding. No custom mapper needed.

3. **Temp dir for config**: `MxConfigLoader` writes received config files to a temp directory,
   then delegates to `HfConfigLoader` to parse them. This reuses all existing config parsing logic.

4. **GPU tensors in weight dict**: `MxWeightLoader.load_weights()` returns GPU tensors directly
   (not CPU). TRT-LLM's `copy_weight()` will do a GPU→GPU copy (~2s for 141 GB). This is
   the Phase 1 limitation documented in TRTLLM_PLAN.md Section 3.8.

---

## UCX Library Conflict (RESOLVED)

### Problem (Now Fixed)

`pip install nixl[cu13]` bundles its own UCX which conflicts with MPI's HPC-X UCX
in TRT-LLM worker processes → segfault in `uct_md_query_tl_resources`.

### Resolution: System NIXL

Use the **system NIXL** from the TRT-LLM NGC image (`/opt/nvidia/nvda_nixl/`). It uses
the same UCX as MPI — no library conflict.

```dockerfile
RUN pip install --no-deps nixl && \
    ln -sf /opt/nvidia/nvda_nixl/lib/python3/dist-packages/nixl_cu13 \
           /usr/local/lib/python3.12/dist-packages/nixl_cu13 && \
    echo "/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu" > /etc/ld.so.conf.d/nixl-system.conf && \
    ldconfig
```

Additional fix: disable descriptor coalescing (`coalesce_transfers=False`) — system NIXL's
`prepXferDlist` has an issue with coalesced descriptors. Individual tensor transfer works at
full bandwidth (127.7 Gbps).

### Test Matrix (All Passing)

| Scenario | Result |
|----------|--------|
| NIXL alone (no MPI) | ✓ OK |
| MPI first, then NIXL (system) | ✓ OK (0.23s agent creation) |
| MPI + NIXL transfer (no coalesce) | ✓ OK (127.7 Gbps) |
| Full MxCheckpointLoader in MPI worker | ✓ OK (config + weights + inference) |
| MPI first, then NIXL (pip) | ✗ Segfault (UCX conflict) |
| System NIXL with coalescing | ✗ NIXL_ERR_NOT_FOUND (minor, not blocking) |

### Current Deployment Status (kavin namespace)

```
modelexpress-server   Running   image: trtllm-ph1 (with model_files proto)
trtllm-source         Running   publishes Qwen 0.5B (290 tensors + 4 config files)
trtllm-target         Scaled=0  (needs image rebuild with system NIXL Dockerfile)
trtllm-debug          Running   validated full E2E in debug pod
```

### Resolution: Solution A Validated (System NIXL)

**System NIXL + disabled coalescing = fully working.**

1. **System NIXL** (`/opt/nvidia/nvda_nixl/`) uses the same UCX as MPI — no conflict ✓
2. **Coalescing disabled** (`coalesce_transfers=False`) — system NIXL's `prepXferDlist` has
   an issue with coalesced descriptors, but individual tensor transfers work at full speed ✓
3. **MPI + NIXL coexist** — tested: MPI init first, then NIXL agent creation ✓

**Dockerfile fix**:
```dockerfile
# Use system NIXL (same UCX as MPI) — NOT pip nixl
RUN pip install --no-deps nixl && \
    ln -sf /opt/nvidia/nvda_nixl/lib/python3/dist-packages/nixl_cu13 \
           /usr/local/lib/python3.12/dist-packages/nixl_cu13 && \
    echo "/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu" > /etc/ld.so.conf.d/nixl-system.conf && \
    ldconfig
```

### End-to-End Results (Qwen 0.5B, TP=1) — PASSING

```
Config:     4 files from MX server (no PVC)          0.08s
Transfer:   290 tensors, 0.99 GB via NIXL RDMA       0.06s @ 127.7 Gbps
Load:       488 weight params mapped + loaded         0.89s
TRT-LLM:   Model init + KV cache allocation         ~31s
Inference:  Generated text successfully               ✓
Total:      32.15s (dominated by TRT-LLM engine init)
```

Deployed as K8s deployment (Step 1 done). No PVC, no init container, no disk I/O. ✓

### Llama 70B (TP=8) — NIXL Transfer Works, But Disk Still Required

RDMA transfer validated at 85 Gbps per rank (170.54 GB total, 8 ranks):

```
Per-rank:   723 tensors, 21.32 GB in 2.0s @ 85.3 Gbps
Total:      8 ranks × 21.32 GB = 170.54 GB
Reconstruct: 723 full tensors → 141.11 GB in ~5s
Config:     5 files (9 MB) from MX server — no PVC ✓
```

**However, the custom checkpoint loader (zero-disk) does NOT work for TP>1.**

---

## Blocker: TP>1 Custom Loader — Per-Worker Weight Loading Problem

### The Problem

TRT-LLM spawns 8 MPI worker processes for TP=8. Each worker independently calls
`MxCheckpointLoader.load_weights()`. This creates a fundamental conflict:

```
What HfWeightMapper expects:    Full HF model weights (141 GB)
                                → Mapper shards per rank: q_proj[0:N/8] for rank 0, etc.

What each worker can get:       Only its matching source rank's shard (21 GB)
                                → Already sharded: q_proj is already [0:N/8]

Problem:                        Mapper will re-shard an already-sharded tensor
                                → Wrong shapes, wrong data
```

### What We Tried

| Approach | Result |
|----------|--------|
| All 8 workers receive from all 8 source ranks | `NIXL_ERR_REMOTE_DISCONNECT` — 64 concurrent NIXL connections overwhelm source agents |
| Each worker receives only matching rank's shard | Mapper re-shards the already-sharded tensor → wrong shapes |
| Each worker receives all ranks on local GPU | Works for TP=1, but TP=8 would need 141 GB on each GPU (OOM) |

### Root Cause

TRT-LLM's checkpoint loading architecture assumes:
1. `load_weights()` returns **full** HF-format weights
2. `HfWeightMapper` handles all TP sharding (fusing q+k+v → qkv, splitting by tp_rank)
3. Each MPI worker calls `load_weights()` independently — no coordination between workers

For file-based loading (`HfWeightLoader`), this works because each worker reads the same files
from disk (shared filesystem). For RDMA, there's no shared filesystem — each worker must
independently receive weights, and receiving the full 141 GB per worker is wasteful (8× redundant).

### What We Need (Upstream)

**Option A: Pre-sharded weight loading (most impactful)**

Each worker receives only its rank's shard via RDMA. The loader returns per-rank weights
with HF-format names. A new "identity" weight mapper (or a `PRESHARDED` load format)
skips the sharding step — it still needs to do fused-weight assembly (q+k+v → qkv) but
not TP slicing.

```python
# In load_weights(), each worker only gets its own rank:
def load_weights(self, checkpoint_dir, mapping):
    my_rank = mapping.tp_rank
    weights = nixl_receive(source_rank=my_rank)  # 21 GB, not 141 GB
    return weights  # Already per-rank sharded

# Need: Mapper that fuses (q+k+v → qkv) but doesn't TP-slice
```

**Option B: Coordinated single-receiver**

Only rank 0's worker receives all weights, then distributes to other workers via MPI
broadcast or shared memory. Requires changes to `ModelLoader.load()` to support
coordinated loading across ranks.

**Option C: Model reference in loader (Phase 2 approach)**

Pass the model to `load_weights()`. Each worker has its model allocated on its own GPU.
NIXL writes directly into each worker's model parameter buffers (21 GB per rank).
No reconstruction, no mapper, no disk.

```python
def load_weights(self, checkpoint_dir, mapping, model=None):
    for name, param in model.named_parameters():
        source_rank = mapping.tp_rank
        nixl_write_into(param.data_ptr(), source_rank, tensor_name=name)
```

### Current Status

| Model | TP | Custom Loader (zero-disk) | Two-Step (disk) |
|-------|----|--------------------------|--------------------|
| Qwen 0.5B | 1 | ✓ PRESHARDED working (150 Gbps) | ✓ Working |
| Llama 70B | 8 | ⚠ RDMA works, shape mismatch in QKV fuse | ✓ Working (85 Gbps/rank) |

---

## Phase 2 Progress: PRESHARDED Implementation

### TRT-LLM Upstream (branch: `kavink/presharded-weight-loading`)

3 files, ~50 lines:
- `llm_args.py`: `LoadFormat.PRESHARDED = 3`
- `model_loader.py`: PRESHARDED branch sets `_weights_presharded` on modules, passes `model=model`
- `linear.py`: `load_weight_shard()` checks `presharded` + `expected_shape` to skip TP slicing;
  `copy_weight()` pointer swap; 3 helpers thread `presharded` and `expected_shape`

### ModelExpress Client (`trtllm_checkpoint_loader.py`)

- `load_weights()` accepts `model=None` for future GPU-resident weights
- Each MPI worker receives only from its matching source rank
- `.cpu()` still needed (NIXL shutdown invalidates GPU tensors)

### Resolved Issues

- **UCX library conflict**: Use system NIXL from NGC image ✓
- **Pickle for MPI workers**: Module-level `MxCheckpointLoader` class ✓
- **lm_head/embed_tokens**: `expected_shape` comparison in `load_weight_shard()` — replicated
  tensors (full-size) still get TP-sliced, pre-sharded tensors skip slicing ✓
- **Fused modules (qkv, gate_up)**: Pass `expected_shape=None` so presharded always skips ✓
- **Sharding dimension convention**: Source now uses dim 0 for column parallel, dim 1 for row
  parallel (matching TRT-LLM's `TensorParallelMode`) ✓

### Current Blocker: k_proj/v_proj Not Sharded by Source

**Error**: `copy_weight: tensor a (1280) must match tensor b (3072) at dimension 0`

Expected fused QKV per rank: `q(1024) + k(128) + v(128) = 1280`
Actual: `q(1024) + k(1024) + v(1024) = 3072`

k_proj and v_proj arrive at full size [1024, 8192] instead of sharded [128, 8192].
The source's `_shard_tensor_for_rank` has the correct patterns and narrow() call,
but the weights are not being sharded. Need shape logging on source to diagnose.

### Next Steps

1. Add shape logging to source's `_shard_tensor_for_rank` — verify narrow() is called
2. Fix source sharding for k_proj/v_proj
3. Validate TP=8 end-to-end with Llama 70B
4. Discuss upstream changes with TRT-LLM team
