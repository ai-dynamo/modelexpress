# Questions for TRT-LLM Team — P2P Weight Transfer Integration

**Date**: Feb 17, 2026 (updated with Phase 1 E2E results)
**From**: ModelExpress / Dynamo team
**Context**: We're building GPU-to-GPU model weight transfers via NIXL/RDMA for TRT-LLM,
similar to what we have working for vLLM today. This document covers what we've tried,
what works, what doesn't, and what we need from TRT-LLM upstream.

---

## Background: What We're Building

ModelExpress enables **sub-second model weight replication** between inference instances
over RDMA. A "source" instance with a loaded model serves its GPU-resident weights to
"target" instances that need the same model — bypassing NVMe/network storage entirely.

**Working today with vLLM:**
- Custom `--load-format mx-target` loader registered via `@register_model_loader`
- NIXL RDMA transfers weights directly to GPU, vLLM uses them in-place
- Llama 70B (TP=8): 170 GB transferred in ~1.5s at 112 Gbps
- Zero disk I/O on the target — single container, no PVC needed

**Goal for TRT-LLM:** Achieve the same — fast P2P weight loading with no disk round-trip.

---

## What We've Tried

### 1. Custom Checkpoint Loader (`@register_checkpoint_loader`)

We discovered TRT-LLM (v1.2+) has a checkpoint loader plugin system similar to vLLM:

```python
@register_checkpoint_loader("mx-p2p")
class MxCheckpointLoader(BaseCheckpointLoader):
    def load_config(self, checkpoint_dir, **kwargs): ...   # Gets config from MX server
    def load_weights(self, checkpoint_dir, mapping): ...   # Gets weights via NIXL RDMA
    def get_initialized_weight_mapper(self, model, config): ...  # Reuses HfWeightMapper
```

**What works:**
- Registration: `@register_checkpoint_loader("mx-p2p")` ✓
- Config loading from remote source (via gRPC) ✓
- Weight mapper reuse (`HfWeightMapper`) ✓
- `checkpoint_loader=MxCheckpointLoader()` instance passed to `LLM()` ✓
- Picklable loader class (module-level, not inner function) ✓

**What initially broke (now resolved):**
- TRT-LLM spawns executor workers as **MPI processes**
- `pip install nixl[cu13]` bundles its own UCX which conflicts with MPI's HPC-X UCX → segfault
- **Fix**: Use **system NIXL** from TRT-LLM NGC image (`/opt/nvidia/nvda_nixl/`) — same UCX as MPI
- Also: descriptor coalescing fails with system NIXL; disabling it works (individual tensors still hit 127 Gbps)

**End-to-end validated (Qwen 0.5B, TP=1):**
```
Config:     4 files from MX server (no PVC)          0.08s
Transfer:   290 tensors, 0.99 GB via NIXL RDMA       0.06s @ 127.7 Gbps
Load:       488 weight params mapped + loaded         0.89s
TRT-LLM:   Model init + KV cache allocation         ~31s
Inference:  Generated text successfully               ✓
Total:      32.15s (dominated by TRT-LLM engine init)
```

### 2. Two-Step Approach (NIXL before TRT-LLM) — Superseded

We also tested a two-step workaround (NIXL in main process, save to tmpfs, TRT-LLM loads from there).
This works but has a disk round-trip. The custom checkpoint loader approach above is preferred
since it eliminates the disk write/read entirely.

---

## High-Level Needs

### Need 1: NIXL + MPI Coexistence (RESOLVED)

**Status: Resolved.** Using the system NIXL from the TRT-LLM NGC image (`/opt/nvidia/nvda_nixl/`)
instead of `pip install nixl[cu13]` eliminates the UCX conflict. System NIXL shares the same
UCX as MPI (HPC-X), so NIXL agents can be created inside MPI worker processes.

**Remaining question for TRT-LLM team:**
- Is the NIXL at `/opt/nvidia/nvda_nixl/` intended for external use, or only for KV cache?
- Will it continue to ship in future NGC releases?
- Is there a plan to make the Python NIXL API (`nixl_cu13`) a first-class supported interface?

### Need 2: Run Custom Loader in MPI Worker Context (RESOLVED)

**Status: Resolved.** The custom `MxCheckpointLoader` runs successfully inside MPI workers
when using system NIXL. The loader:
1. Is defined at module level (picklable for multiprocessing)
2. Is passed as an instance via `LLM(checkpoint_loader=MxCheckpointLoader())`
3. Registers all three registries (`checkpoint_loader`, `weight_loader`, `config_loader`)

**Remaining question:**
- The `_construct_checkpoint_loader()` function requires all three separate registries
  (`checkpoint_weight_loader`, `config_loader`, `checkpoint_loader`) when using `checkpoint_format=`.
  Is there a plan to simplify this to a single registration?
- Could the `checkpoint_loader` instance path skip the `_construct_checkpoint_loader()` lookup
  entirely? (Currently it does, but only when passed as `checkpoint_loader=`, not `checkpoint_format=`.)

### Need 3: Direct GPU Tensor Injection (Skip `copy_weight`)

### Need 3: Direct GPU Tensor Injection (Skip `copy_weight`)

Currently `load_weights()` returns a weight dict, and TRT-LLM does `param.data.copy_(w)`
for every parameter. If our weights are already on GPU (from RDMA), this is a wasted
GPU→GPU copy.

**What we want:**
```python
# Option A: Pointer swap instead of copy
param.data = rdma_tensor  # Zero-copy, O(1)

# Option B: load_weights() receives the model so we can write directly into params
def load_weights(self, checkpoint_dir, mapping, model=None):
    for name, param in model.named_parameters():
        nixl_write_into(param.data_ptr(), ...)  # RDMA directly into param buffer
```

**Question for TRT-LLM team:**
- Could `copy_weight()` support `param.data = src` (pointer swap) when src is already on
  the correct device with matching shape/dtype?
- Could the `model` object be passed to `load_weights()` (backward-compatible, `model=None`
  default)? This would let custom loaders do direct injection.
- Is there a reason `copy_weight()` must always copy? (e.g., memory layout requirements,
  TensorRT engine constraints, etc.)

### Need 4: TP-Aware Pre-Sharded Weight Loading (CRITICAL — P0 for TP>1)

**This is now our #1 blocker.** The custom checkpoint loader works perfectly for TP=1 but
breaks for TP>1 because of how TRT-LLM handles per-worker weight loading.

**The problem in detail:**

TRT-LLM spawns 8 MPI workers for TP=8. Each worker independently calls `load_weights()`.
The `HfWeightMapper` expects **full** HF weights and shards them per rank. But with RDMA:

```
Approach A: All 8 workers receive all 8 source ranks
            → 64 NIXL connections → NIXL_ERR_REMOTE_DISCONNECT (connection storm)

Approach B: Each worker receives only its matching rank's shard (21 GB, not 141 GB)
            → Mapper re-shards already-sharded weights → wrong shapes

Approach C: Each worker receives all 8 ranks on its local GPU
            → 141 GB per GPU → OOM (GPU has ~80 GB)
```

The RDMA transfer **works** (85 Gbps per rank, validated). The issue is entirely in
TRT-LLM's assumption that `load_weights()` returns full HF weights.

**What we need — one of:**

1. **Pre-sharded load mode**: `load_weights()` returns per-rank weights (already TP-split).
   The mapper should fuse (q+k+v → qkv) but NOT TP-slice:

   ```python
   def load_weights(self, checkpoint_dir, mapping):
       my_rank = mapping.tp_rank
       weights = nixl_receive(source_rank=my_rank)  # 21 GB per rank
       return weights  # HF names, but per-rank sizes
   
   # Mapper: fuses q+k+v weights but skips load_weight_shard() TP slicing
   ```

2. **Coordinated loading across workers**: `load_weights()` is called once (not per-worker)
   with the full model, then weights are distributed to workers via MPI/shared memory.

3. **Model reference in loader**: Each worker's `load_weights(model=model)` gets the model
   with params already allocated on the correct GPU. NIXL writes directly into each param
   buffer (21 GB per rank, no mapper needed).

**Question for TRT-LLM team:**
- Is there a `LoadFormat` or mapper mode that accepts pre-sharded weights (already split
  for the current rank)?
- The `HfWeightMapper` does TWO things: (a) name mapping + fusing, (b) TP sharding.
  Can these be separated? We need (a) but not (b).
- Could `load_weight_shard()` in `linear.py` detect that the input weight is already
  the correct shard size and skip the slicing?
- Is there a way to coordinate `load_weights()` across MPI workers (e.g., only rank 0
  loads, then broadcast)?
- Could the `mapping` parameter be used to tell the mapper "these weights are already
  sharded for my rank, skip TP slicing"?

### Need 5: Collective Weight Update / Hot Reload

For production use cases, we want to update model weights on a running TRT-LLM instance
without restarting. This requires:

1. **Coordinated update across TP ranks** — all ranks must update simultaneously
2. **Quiesce in-flight requests** — drain the batch before swapping weights
3. **Atomic swap** — switch from old weights to new weights without serving partial state

**Question for TRT-LLM team:**
- Does TRT-LLM have any weight update / hot reload mechanism today?
- For PyTorch backend: could we directly update `model.named_parameters()` between batches?
- For TensorRT engine backend: does `IRefitter` work at runtime while the engine is serving?
  Can it be done without stopping the executor?
- Is there a "pause inference" API we could use to safely swap weights?
- How does the KV cache disaggregated serving handle similar coordination across ranks?

### Need 6: Reuse TRT-LLM's NIXL Infrastructure

TRT-LLM already has NIXL for KV cache transfer in disaggregated serving:
- `NixlTransferAgent` in `cpp/tensorrt_llm/executor/cache_transmission/nixl_utils/`
- Python bindings in `tensorrt_llm/_torch/disaggregation/nixl/`
- Uses system UCX (no conflict with MPI)

**Question for TRT-LLM team:**
- Can we reuse `NixlTransferAgent` for weight transfers (not just KV cache)?
- Is the Python binding (`_agent_py.py`, `_agent_cpp.py`) stable enough for external use?
- Is there a plan to expose NIXL as a general-purpose transfer API in TRT-LLM?

---

## The UCX Library Conflict (Resolved)

### What Happened

```
pip install nixl[cu13]  →  bundles libucp-*.so, libuct-*.so (NIXL's own UCX)
TRT-LLM MPI workers     →  use /opt/hpcx/ucx/ (system UCX)
Both in same process     →  segfault in uct_md_query_tl_resources
```

### Resolution

Use the **system NIXL** from the TRT-LLM NGC image (`/opt/nvidia/nvda_nixl/`). It links
against the same HPC-X UCX as MPI — no conflict.

```dockerfile
# Dockerfile fix: use system NIXL, NOT pip nixl
RUN pip install --no-deps nixl && \
    ln -sf /opt/nvidia/nvda_nixl/lib/python3/dist-packages/nixl_cu13 \
           /usr/local/lib/python3.12/dist-packages/nixl_cu13 && \
    echo "/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu" > /etc/ld.so.conf.d/nixl-system.conf && \
    ldconfig
```

**Note**: Descriptor coalescing (`_coalesce_transfers`) doesn't work with system NIXL
(`prepXferDlist` fails with `NIXL_ERR_NOT_FOUND`). Disabling coalescing works fine —
individual tensor transfers still hit 127.7 Gbps.

**Remaining question for TRT-LLM team:**
- Is the NIXL at `/opt/nvidia/nvda_nixl/` intended for external use or only for KV cache?
- Will it continue to be included in future NGC releases?
- Any known issues with the system NIXL's `prepXferDlist` and coalesced descriptors?

---

## What Exists Today vs What We Need

| Capability | TRT-LLM Today | What We Need | Status |
|-----------|---------------|-------------|--------|
| Custom checkpoint loader | `@register_checkpoint_loader` ✓ | Same | **Working** ✓ |
| Custom weight loader | `@register_checkpoint_weight_loader` ✓ | Same | **Working** ✓ |
| Custom config loader | `@register_config_loader` ✓ | Same | **Working** ✓ |
| `checkpoint_loader` instance | `LLM(checkpoint_loader=...)` ✓ | Same | **Working** ✓ |
| Loader runs inside MPI worker | Yes (runs in worker) | NIXL inside worker | **Working** ✓ (system NIXL) |
| NIXL compatible with MPI | System NIXL yes, pip NIXL no | System NIXL | **Working** ✓ (resolved) |
| TP=1 zero-disk loading | Via custom loader | Same | **Working** ✓ (151 Gbps) |
| **TP>1 zero-disk loading** | **Not possible** | Pre-sharded weights | **P0 BLOCKER** |
| Per-worker coordination | None (each worker independent) | Coordinated or pre-sharded | **Upstream needed** |
| Model passed to loader | No (`load_weights` gets dir + mapping) | `model=` param | **Upstream needed** |
| Pointer swap in copy_weight | No (always `.copy_()`) | `param.data = src` | **Upstream needed** |
| Hot weight reload | No | Coordinated update + quiesce | **Upstream needed** |
| NIXL Python API for weights | KV cache only | General-purpose | **Clarification** |

---

## POC Results (Validated End-to-End)

### Phase 1 Complete: Custom Checkpoint Loader (Qwen 0.5B, TP=1)

| Step | Time | Detail |
|------|------|--------|
| Config from MX server | 0.08s | 4 files (7 MB) via gRPC — no PVC |
| NIXL RDMA transfer | 0.06s | 290 tensors, 0.99 GB @ **127.7 Gbps** |
| Weight mapping + loading | 0.89s | HfWeightMapper: 290→488 params |
| TRT-LLM engine init | ~31s | PyTorch backend, KV cache allocation |
| Inference | ✓ | Generated text correctly |
| **Total** | **32.15s** | Dominated by TRT-LLM init (not transfer) |

### Earlier POC: Init Container Approach (Llama 70B, TP=8)

| Metric | Value |
|--------|-------|
| NIXL Transfer | 170 GB in ~12s (112 Gbps) |
| Reconstruction | 723 tensors → 141 GB (~5s) |
| Disk write + TRT-LLM reload | ~35s |
| TRT-LLM inference | 15.9 TPS |

### Key Achievements and Gaps

**Working (TP=1):** Custom checkpoint loader eliminates init container, PVC, and disk I/O.
Weights go RDMA → CPU dict → TRT-LLM model params at 151 Gbps.

**Working (TP=8 transfer):** NIXL RDMA at 85 Gbps per rank (170 GB total in ~16s).
Config from MX server (no PVC).

**Blocked (TP>1 zero-disk):** Each MPI worker independently calls `load_weights()`.
Workers can't coordinate RDMA downloads, and the mapper re-shards already-sharded weights.
Currently falls back to disk (RDMA → save safetensors → TRT-LLM reads).

---

## Summary of Asks (Priority Order)

### Resolved (No Upstream Needed)

- ~~**UCX/NIXL guidance**~~: **Resolved.** Use system NIXL from NGC image, not pip nixl.
- ~~**Custom loader in MPI worker**~~: **Resolved.** System NIXL + module-level class + instance passing.

### Remaining Asks

1. **P0 — Pre-sharded weight loading (TP>1 BLOCKER)**: Accept per-rank weights that are
   already TP-sharded, with a mapper that fuses (q+k+v → qkv) but skips TP slicing.
   **Without this, TP>1 cannot bypass disk.** This is our #1 ask. Options:
   - New `LoadFormat.PRESHARDED` that tells the mapper weights are already per-rank
   - `load_weight_shard()` auto-detects that input size == expected shard size (skip slicing)
   - Separate the mapper's fusing (needed) from sharding (not needed for pre-sharded)
   - Coordinated loading: only rank 0 calls `load_weights()`, broadcasts to other ranks

2. **P0 — System NIXL support**: Confirm `/opt/nvidia/nvda_nixl/` is a supported external
   interface and will continue to ship in NGC releases.

3. **P1 — Model reference in loader**: Pass `model` to `load_weights()` so custom loaders
   can read `param.data_ptr()` and have NIXL write directly into parameter buffers.
   This would also solve the TP>1 problem (each worker has its model on the right GPU).

4. **P1 — Pointer swap in copy_weight**: Use `param.data = src` instead of `.copy_()` when
   source tensor is already on the correct device with matching shape.

5. **P2 — Coalesced descriptor support**: System NIXL's `prepXferDlist` fails with coalesced
   descriptors. Individual tensor transfer works at full bandwidth.

6. **P2 — Hot weight reload**: API to update weights on a running instance with coordinated
   TP-rank synchronization and request draining.
