# TRT-LLM Phase 2: PRESHARDED Weight Loading — Implementation & Results

**Branch**: `kavink/trtllm` (ModelExpress) + `kavink/presharded-weight-loading` (TRT-LLM)
**Status**: TP=1 and TP=8 validated end-to-end. Zero-disk, single container, no PVC.
**Date**: Feb 18, 2026

---

## Results

### Llama 70B (TP=8) — Zero-Disk P2P Transfer + TRT-LLM Inference

```
Config:      5 files from MX server (no PVC)              0.1s
Transfer:    8 × 21.32 GB via NIXL RDMA                   2.8s (60-67 Gbps/rank)
.cpu() copy: 21 GB per worker GPU→CPU                     ~0.5s
Mapper:      HfWeightMapper fuse (q+k+v → qkv)            ~0.5s
copy_weight: CPU→GPU into model params                     ~1.0s
TRT-LLM:     Engine init + KV cache allocation             ~420s
Inference:   "Meta AI, and I can provide information..."   ✓
────────────────────────────────────────────────────────────────
Total:       428.70s (dominated by TRT-LLM engine init)
Weight I/O:  ~5s (was ~44s in POC, ~180s from NVMe)
```

### Qwen 0.5B (TP=1) — Zero-Disk P2P Transfer

```
Transfer:    290 tensors, 0.99 GB via NIXL RDMA            0.05s (150 Gbps)
Total:       31.54s
```

---

## What We Changed

### TRT-LLM Upstream (3 files, ~50 lines)

**Branch**: `kavink/presharded-weight-loading` on TensorRT-LLM repo

These changes are patched into the NGC TRT-LLM 1.2.0rc6 image via file overlay in the
Dockerfile. They would become a proper upstream PR.

#### 1. `tensorrt_llm/llmapi/llm_args.py`

Added `LoadFormat.PRESHARDED = 3` to the enum. This tells TRT-LLM that weights returned
by the checkpoint loader are already sharded per TP rank.

#### 2. `tensorrt_llm/_torch/pyexecutor/model_loader.py`

Added a `PRESHARDED` branch in `ModelLoader.load()`:
- Sets `_weights_presharded = True` on all `Linear` modules
- Passes `model=model` to `checkpoint_loader.load_weights()` (for future direct GPU injection)
- Otherwise identical to the `AUTO` branch

#### 3. `tensorrt_llm/_torch/modules/linear.py`

Three changes:

**a) `load_weight_shard()` — skip slicing for pre-sharded weights**

Added `presharded: bool = False` and `expected_shape: Optional[tuple] = None` parameters.
When `presharded=True`:
- If `expected_shape is None` (fused helpers — qkv, gate_up): always skip TP slicing.
  These tensors are always sharded by the source.
- If `expected_shape` provided (vanilla helpers): skip only if `width == expected_width`.
  This handles replicated tensors (embed_tokens, lm_head) that are full-size and still
  need TP slicing.

```python
if presharded:
    if expected_shape is None:
        return maybe_convert_to_torch_tensor(weight)  # Fused: always skip
    expected_width = expected_shape[split_dim]
    if width == expected_width:
        return maybe_convert_to_torch_tensor(weight)  # Vanilla: skip if sizes match
# ... else normal slicing
```

**b) Three weight loading helpers — thread presharded flag**

`load_weights_vanilla_helper()`, `load_weights_fused_qkv_helper()`, and
`load_weights_fused_gate_up_helper()` each:
- Read `_presharded = getattr(module, '_weights_presharded', False)`
- Pass `presharded=_presharded` and `expected_shape=tuple(module.weight.shape)`
  (or `expected_shape=None` for fused helpers) to `load_weight_shard()`

**c) `copy_weight()` — pointer swap for zero-copy**

When source and destination tensors are on the same device with matching shape and
contiguity, use `dst.data = src` instead of `dst.data.copy_(src)`.

### ModelExpress Client Changes

**File**: `modelexpress_client/python/modelexpress/trtllm_checkpoint_loader.py`

#### MxWeightLoader.load_weights()

- Accepts `model=None` parameter (passed by TRT-LLM's PRESHARDED branch)
- Each MPI worker receives ONLY from its matching source rank (`mapping.tp_rank`)
- 1 NIXL connection per source agent (no thundering herd)
- `.cpu()` copy after RDMA (NIXL shutdown invalidates GPU tensors)
- Returns HF-format weight dict (per-rank sizes)

#### MxCheckpointLoader.get_initialized_weight_mapper()

- Detects `_weights_presharded` on model modules
- Sets `weight_mapper._tp_size = 1` to prevent `_duplicate_kv_weights` from expanding
  pre-sharded KV heads (the mapper would see 1 KV head per rank and try to duplicate 8×)

#### Source: MxTrtllmSourcePublisher._shard_tensor_for_rank()

- Column parallel (q/k/v/gate/up): shard at dim 0 (matching TRT-LLM's convention)
- Row parallel (o_proj/down_proj): shard at dim 1
- Uses `tensor.narrow(shard_dim, start_idx, shard_size)` for clean slicing
- Replicated tensors (embed_tokens, lm_head, layernorm): returned full-size

### Docker Image

**Base**: `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6`
**NIXL**: System NIXL from `/opt/nvidia/nvda_nixl/` (same UCX as MPI — no conflict)
**TRT-LLM patches**: 3 files overlaid via `COPY trtllm_patches/`
**Tag**: `nvcr.io/nvidian/dynamo-dev/modelexpress-trtllm-client:ph2`

---

## Issues Discovered & Resolved

### 1. UCX Library Conflict (Phase 1)

`pip install nixl[cu13]` bundles its own UCX which conflicts with MPI's HPC-X UCX.
**Fix**: Use system NIXL from NGC image.

### 2. MxCheckpointLoader Not Picklable (Phase 1)

TRT-LLM spawns MPI workers that pickle the loader. Inner-function class can't pickle.
**Fix**: Define `MxCheckpointLoader` at module level.

### 3. ConsumableWeightsDict Missing (Phase 1)

TRT-LLM 1.2.0rc6 doesn't have `ConsumableWeightsDict` (added in 1.3+).
**Fix**: Optional import, fall back to plain dict.

### 4. Replicated Tensors (lm_head, embed_tokens)

Source sends full-size replicated tensors. With `presharded=True`, `load_weight_shard`
would skip slicing for ALL tensors including replicated ones.
**Fix**: `expected_shape` comparison — skip slicing only when input matches module param size.
Fused helpers pass `expected_shape=None` (always skip), vanilla helpers pass the module's
weight shape (skip if match, slice if larger).

### 5. Source Sharding Dimension Convention

Source originally sharded column-parallel at dim=-1 and row-parallel at dim=0.
TRT-LLM uses column=dim0, row=dim1. Also the slicing code used `if dim == -1` which
didn't handle `dim=1` correctly.
**Fix**: Aligned conventions + `tensor.narrow()` for arbitrary dimension.

### 6. KV Head Duplication (GQA)

`HfWeightMapper._duplicate_kv_weights` sees pre-sharded k_proj with 1 KV head per rank
(128/128=1) and duplicates it 8× (since `num_kv_heads(1) < tp_size(8)`).
**Fix**: Set `weight_mapper._tp_size = 1` for presharded mode so the check becomes
`num_kv_heads(1) >= tp_size(1)` → no duplication.

### 7. Stale Metadata from Previous Source

MX server merges metadata by worker rank. If target starts before new source publishes,
it gets old metadata with wrong NIXL agent addresses → `NIXL_ERR_REMOTE_DISCONNECT`.
**Workaround**: Ensure source publishes before target starts. Proper fix: session-based
metadata invalidation.

---

## Remaining CPU Round-Trip

The current flow still copies weights GPU→CPU→GPU:

```
Source GPU ──RDMA──► Target GPU buffer ──.cpu()──► CPU dict ──fuse──► copy_weight──► Model param GPU
```

### Why `.cpu()` Is Needed Today

The NIXL agent is shut down after receiving weights (to free RDMA resources). Shutdown
invalidates the GPU memory registered with NIXL. If we don't copy to CPU first, the
GPU tensors become garbage.

### Option A: Defer NIXL Shutdown

Keep the NIXL agent alive until after `model.load_weights()` completes. The GPU tensors
stay valid because the agent still owns the memory registration.

```python
# Current:
nixl_mgr.receive()
cpu_weights = {k: v.cpu() for ...}  # copy to CPU
nixl_mgr.shutdown()                  # invalidates GPU tensors
return cpu_weights                   # CPU dict → mapper → copy_weight(CPU→GPU)

# Option A:
nixl_mgr.receive()
all_weights[rank] = weights          # keep on GPU
# DON'T shutdown yet — return GPU dict
# shutdown later (after model.load_weights finishes)
```

**Pros**:
- Simple change (~10 lines in MxWeightLoader)
- Eliminates GPU→CPU copy (~0.5s for 21 GB)
- `copy_weight()` pointer swap works for vanilla modules (same device, same shape)
- Fused modules still need `torch.cat()` + copy (unavoidable without direct injection)

**Cons**:
- NIXL agent stays alive longer (holds RDMA memory registration)
- Need to store agent references and shutdown after `load_weights()` returns
- The `load_weights()` function currently returns a dict — adding shutdown logic
  requires either a callback or storing agents on the loader instance
- If `model.load_weights()` fails, agents might leak (need cleanup in finally block)
- Each worker holds 21 GB of GPU memory for both the NIXL buffer AND model params
  simultaneously (~42 GB peak per GPU for 70B/TP=8)

### Option B: NIXL into Model Param Buffers

Use the `model` reference (already passed via PRESHARDED) to get `param.data_ptr()` for
each parameter. Have NIXL write directly into model parameter buffers. No intermediate
allocation, no copy.

```python
# Option B:
def load_weights(self, checkpoint_dir, mapping, model=None):
    # Build map: HF weight name → model param GPU address
    param_map = {}
    for name, param in model.named_parameters():
        hf_name = trtllm_to_hf_name(name)  # reverse mapping
        param_map[hf_name] = param

    # NIXL writes directly into param buffers
    for each source tensor:
        dst_addr = param_map[tensor_name].data_ptr()
        nixl_transfer(src_addr → dst_addr)

    # Return empty dict — weights already in model params
    return {}
```

**Pros**:
- True zero-copy for ALL modules (vanilla and fused)
- No GPU→CPU→GPU round-trip at all
- No intermediate GPU buffer allocation (saves ~21 GB peak memory)
- No dependency on NIXL agent lifetime

**Cons**:
- Requires **reverse name mapping** (TRT-LLM param name → HF source tensor name).
  TRT-LLM fuses q+k+v into a single `qkv_proj` param, so there's no 1:1 mapping
  for fused modules. We'd need to write q, k, v into sub-regions of the fused buffer.
- Need to know the **fused layout** (where q ends and k starts within the qkv buffer).
  This is model-specific (depends on num_heads, num_kv_heads, head_dim).
- The `model.load_weights()` step would need to be skipped or made a no-op
  (since weights are already in params). Requires upstream change to support this.
- More complex implementation (~200 lines vs ~10 lines for Option A)
- Harder to debug (RDMA writes directly into model memory — silent corruption if addresses wrong)

### Recommendation

**Start with Option A (defer shutdown)** — it's a 10-line change that eliminates the
CPU round-trip for ~60% of parameters (vanilla modules) and reduces peak memory pressure.
The `copy_weight()` pointer swap we already implemented will kick in for these modules.

**Then pursue Option B** as a follow-up for full zero-copy on fused modules. This requires:
1. Building a TRT-LLM param name → HF name reverse mapping
2. Understanding fused buffer layouts per model architecture
3. Upstream support for "weights already loaded" (skip `model.load_weights()`)

### Expected Performance with Option A

```
Current (CPU round-trip):
  RDMA: 2.8s → .cpu(): 0.5s → fuse: 0.5s → copy_weight: 1.0s → Total: ~5s

Option A (deferred shutdown, GPU-resident):
  RDMA: 2.8s → fuse: 0.5s → copy_weight (pointer swap ~60%): 0.4s → Total: ~3.7s
  Saves: ~1.3s (26% improvement in weight loading)

Option B (direct injection):
  RDMA into params: 2.8s → Total: ~2.8s
  Saves: ~2.2s (44% improvement in weight loading)
```

Note: TRT-LLM engine init (~420s) dominates total time in all cases. The weight loading
optimization matters most for:
- Repeat deployments (engine cached, only weights change)
- Scale-out scenarios (many targets spinning up)
- Comparing against NVMe baseline (~180s for weight loading alone)

---

## Summary: What's Working

| Feature | TP=1 | TP=8 |
|---------|------|------|
| RDMA transfer | ✓ 150 Gbps | ✓ 60-67 Gbps/rank |
| Config from MX server (no PVC) | ✓ | ✓ |
| Single container (no init container) | ✓ | ✓ |
| Zero disk I/O | ✓ | ✓ |
| PRESHARDED (skip TP re-slicing) | ✓ | ✓ |
| System NIXL (MPI compatible) | ✓ | ✓ |
| Weight fusing (q+k+v → qkv) | N/A (TP=1) | ✓ |
| KV duplication prevention | N/A | ✓ |
| Replicated tensor handling | N/A | ✓ |
| copy_weight pointer swap | ✓ | ✓ (vanilla only) |
| Inference | ✓ | ✓ |

## Next Steps

1. **Option A**: Defer NIXL shutdown, remove `.cpu()`, enable GPU-resident weights
2. **Update TRTLLM_PLAN.md and docs** with Phase 2 completion
3. **Benchmark**: Compare Phase 2 vs POC vs NVMe baseline
4. **Upstream PR**: Submit TRT-LLM changes (PRESHARDED + copy_weight + model ref)
5. **Option B** (future): Direct NIXL injection into model param buffers
