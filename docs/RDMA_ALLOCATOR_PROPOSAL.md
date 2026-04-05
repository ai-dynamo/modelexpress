# Proposal: RDMA-Aware Weight Allocation for GPU Inference Engines

## Problem

GPU-to-GPU model weight transfer via RDMA (NIXL/UCX) requires registering
GPU memory with the InfiniBand HCA via `ibv_reg_mr`. This is a kernel call
into the IB driver that takes 0.1-10ms per registration depending on region
size. For large models, this dominates cold start time:

| Model | Tensors/worker | Size/worker | Registration | Transfer |
|---|---|---|---|---|
| Qwen2.5-3B (6 GB) | 255 | 6 GB | 0.06s | 0.23s |
| DeepSeek-V2-Lite FP8 (16 GB) | 539 | 16 GB | 0.29s | 0.58s |
| Hermes 405B FP8 TP8 | 1264 | 51 GB | 2.6s | 1.9-12s |
| DeepSeek-V3 FP8 TP8 EP8 | 1386 | 86 GB | 3.0s | 4.6-17s |
| Kimi K2.5 NVFP4 TP8 (Nebius) | 2644 | 77 GB | 27s | 2.2s |

All numbers are per-tensor baseline (no pool-reg or VMM). The Nebius
numbers (27s) are from a customer deployment on different hardware; the
nScale B200 numbers (2.6-3.0s) reflect faster IB NICs.

The root cause: PyTorch's caching allocator spreads tensors across
hundreds of `cudaMalloc` segments. Each segment requires a separate
`ibv_reg_mr` call. A single contiguous allocation would need exactly one.

## Two Systems, Same Problem

Two NVIDIA projects independently need control over GPU weight memory
layout for efficient external access:

**ModelExpress (MX)** - cross-node RDMA weight transfer. Source loads
model, registers GPU memory with NIXL, serves weights to target nodes
over InfiniBand. Per-tensor `ibv_reg_mr` overhead dominates target
cold start.

**dynamo GPU Memory Service (GMS)** - same-node cross-process weight
sharing. First process loads model into VMM-backed memory, exports VMM
handles via POSIX FDs, subsequent processes import the same physical
GPU pages (zero copy). Uses `CUDAPluggableAllocator` to direct all
weight allocations into a VMM arena during loading.

Both systems solve "don't repeat expensive weight loading" at different
scales. Both need the same foundation: contiguous, externally-managed
GPU memory that PyTorch tensors can alias.

### Why GMS and MX should converge

GMS already allocates weights into VMM arenas. If MX could register a
GMS VMM arena with NIXL in one `ibv_reg_mr` call, you'd get both
same-node sharing AND cross-node RDMA from the same memory:

```
Node A:
  GMS loads weights into VMM arena (one allocation)
  Local replicas import same physical pages (GMS sharing, zero copy)
  MX registers VMM range with NIXL (one ibv_reg_mr)
  RDMA serves weights to Node B

Node B:
  GMS creates VMM arena
  MX RDMA receives into GMS arena (one ibv_reg_mr)
  Local replicas import same pages (GMS sharing, zero copy)
```

One disk load total. One RDMA transfer between nodes. Zero copies within
each node. Weight data touches memory exactly twice across the entire
fleet: once from disk, once over the wire.

Today this doesn't work because NIXL can't efficiently register or
transfer sub-regions of a VMM range (see investigation results below).

## Current State and Workarounds

### Pool Registration (MX, shipping)
Discover `cudaMalloc` segment boundaries via `cuMemGetAddressRange_v2`,
register one region per segment. Reduces `ibv_reg_mr` calls by ~60-80%.
Speedup: ~2x on registration time. Limited by allocator fragmentation.

### VMM Compaction (MX, prototype)
After weight loading, compact all tensors into a single contiguous CUDA
VMM range. Handles non-contiguous tensors (FP8 scales, MLA projections)
via storage repointing. Segment-ordered freeing limits peak VRAM overhead
to one `cudaMalloc` segment (~2-256 MB).

### GMS VMM Allocator (dynamo, shipping)
`CUDAPluggableAllocator` + `use_mem_pool` directs weight allocations
into VMM during `load_weights()`. No post-hoc compaction needed.
Weights are in VMM from the start.

### Limitations shared by all approaches
- Rely on `torch._C._construct_storage_from_data_pointer` (private API)
- NIXL requires per-tensor registration even when all tensors reside
  in a single contiguous VMM range (see investigation below)
- VMM compaction (MX) is post-hoc; GMS allocator is upfront but
  requires the GMS daemon

## Investigation Results

### NIXL registration codepath

We traced the full path from Python to `ibv_reg_mr`:

```
register_memory([tensor_list])
  -> nixlLocalSection::addDescList()     // loops per descriptor
    -> for each of N tensors:
      -> nixlUcxEngine::registerMem()    // per-descriptor
        -> ucp_mem_map()                 // explicit API, NO rcache
          -> ibv_reg_mr()                // kernel call, ~2ms each
        -> ucp_rkey_pack()               // per-tensor rkey
```

Key finding: `ucp_mem_map` is UCX's explicit registration API. It does
NOT go through UCX's rcache (registration cache). The rcache only applies
to implicit registrations during zero-copy transfer operations. Every
`register_memory` call results in a fresh `ibv_reg_mr` kernel call.

### Range cache prototype (NIXL fork)

We implemented a range cache in NIXL's UCX backend
(`nicolasnoble/nixl`, branch `nnoble/range-aware-registration`):
when a new descriptor falls within an already-registered range, reuse
the existing `ucp_mem_h` and rkey instead of calling `ucp_mem_map`.

Results (Hermes 405B FP8 TP8, 1264 tensors, 51 GB/worker):

| Phase | Without cache | With cache |
|---|---|---|
| Range registration (1 call) | 0.008s | 0.008s |
| Per-tensor descriptors (1264) | 2.4s | **0.009s** |
| Total registration | 2.4s | **0.017s** |

**Registration: 140x faster.** The range cache eliminates per-tensor
`ibv_reg_mr` calls entirely. Each tensor descriptor gets a shared
reference to the range's `ucp_mem_h` + rkey.

### Transfer failure with shared memh

RDMA transfers fail with "Local protection error" when using the shared
`ucp_mem_h`. The IB HCA rejects the lkey for addresses that ARE within
the registered range. This blocks the full end-to-end optimization.

The failure needs investigation at the UCX/nvidia-peermem level. It may
be related to how VMM memory (`cuMemCreate`/`cuMemMap`) is handled by
the IB driver vs regular `cudaMalloc` memory, or a UCX assumption about
memh-to-address correspondence.

Details: `docs/NIXL_RANGE_REGISTRATION_INVESTIGATION.md`

## Benchmark Results

All benchmarks on nScale B200 cluster with InfiniBand, pinned nodes,
3 runs per configuration, steady-state medians.

### Hermes 405B FP8, TP8 (1264 tensors, 51.32 GB/worker)

| Config | Compaction | Registration | Regions | Transfer |
|---|---|---|---|---|
| Per-tensor | - | 2.6s | 1264 | 1.9-12s |
| Pool-reg | - | 1.65s | 524 | 1.9-12s |
| VMM + per-tensor reg | 0.39s | 3.0s | 1264 | 1.9-12s |
| VMM + range cache | 0.39s | **0.017s** | 1265 | **FAILS** |
| GMS + range cache (projected) | 0s | **0.017s** | 1265 | pending fix |

### DeepSeek-V3 FP8, TP8 EP8 (1386 tensors, 85.80 GB/worker)

| Config | Registration (median) | Registration (best) |
|---|---|---|
| Per-tensor | 3.0s | 2.4s |
| Pool-reg | 1.5s | 0.9s |

### Key findings

1. **Pool-reg: 1.6-2.1x registration speedup.** Consistent across models.
   Reduces `ibv_reg_mr` calls by 60-80%.

2. **VMM compaction works but needs NIXL changes to help.** Compaction
   is fast (0.39s for 51 GB) and handles non-contiguous tensors. But
   without NIXL range support, registration time is unchanged.

3. **Range cache: 140x registration speedup, transfer broken.** The
   0.017s registration proves the approach works. The transfer failure
   is the remaining blocker.

4. **GMS eliminates compaction entirely.** If MX integrates with GMS's
   VMM allocator, the 0.39s compaction overhead disappears. Weights
   start in VMM, no post-hoc movement needed.

5. **Transfer speed is independent of registration mode.** All configs
   achieve the same RDMA bandwidth (35-215 Gbps per worker, variance
   from PCIe topology and NIC health).

## Proposed Changes

### 1. NIXL: Sub-region transfer support (CRITICAL BLOCKER)

**What:** Allow RDMA transfers using a `ucp_mem_h` that covers a
superset of the transfer address range.

**Why:** This is the single blocker preventing both VMM compaction and
GMS+MX integration from achieving O(1) registration. We've proven that
registration drops from 2.4s to 0.017s with the range cache, but
transfers fail because the shared `ucp_mem_h` triggers IB protection
errors on sub-region access.

**Scope:** The IB spec explicitly supports sub-region access within a
Memory Region (MR). The lkey from `ibv_reg_mr` is valid for any address
within `[base, base+length]`. The fix is likely in how NIXL/UCX passes
the memh to `ucp_get_nbx` or how nvidia-peermem handles VMM-backed MRs.

**Prototype:** `nicolasnoble/nixl`, branch `nnoble/range-aware-registration`.
Registration works. Transfer investigation documented in
`NIXL_RANGE_REGISTRATION_INVESTIGATION.md`.

### 2. PyTorch: Public External Memory Tensor API

**What:** A public API for creating CUDA tensors backed by externally
managed device memory.

**Why:** `_construct_storage_from_data_pointer` is used in production by
GMS and ModelExpress but has no stability guarantee.

```python
storage = torch.cuda.ExternalStorage(
    data_ptr: int,
    size_bytes: int,
    device: torch.device,
)

tensor = torch.as_tensor_from_storage(
    storage,
    shape: tuple[int, ...],
    stride: tuple[int, ...] | None = None,
    dtype: torch.dtype = torch.float32,
    storage_offset: int = 0,
)
```

**Requirements:**
- All CUDA dtypes including `float8_e4m3fn`, `bfloat16`
- Storage must NOT free underlying memory on GC
- Must work with autograd (`nn.Parameter.data` replacement)

### 3. vLLM: Weight Allocation Hook

**What:** Formalize the pattern GMS already uses (`GMSModelLoader`) as
a vLLM extension point.

```python
class WeightAllocatorHook:
    def pre_allocate(self, model, model_config) -> ContextManager:
        """Wrap load_weights + process_weights_after_loading."""
        return contextlib.nullcontext()

    def post_load(self, model, tensors: dict[str, torch.Tensor]) -> None:
        """Register memory, publish metadata, etc."""
        pass
```

GMS's vLLM integration already implements this pattern:
```python
with use_mem_pool(pool, device=target_device):
    model = initialize_model(vllm_config=vllm_config, model_config=model_config)
    default_loader.load_weights(model, model_config)
    process_weights_after_loading(model, model_config, target_device)
```

Formalizing it benefits all external weight delivery systems.

### 4. PyTorch: Scoped Allocator Context

**What:** Make `CUDAPluggableAllocator` + `use_mem_pool` a first-class
pattern with cleaner ergonomics.

GMS already uses this via a C++ extension shim (`allocator.cpp`) that
routes `malloc`/`free` through Python callbacks to the VMM manager.
A cleaner API would reduce the boilerplate and make the pattern
accessible without writing C++ extensions.

## Deployment Tiers

The proposed changes enable a tiered deployment model where each tier
is a fallback for when the tier above isn't available:

| Tier | MX + GMS | MX + VMM compaction | MX + pool-reg |
|---|---|---|---|
| Prerequisite | GMS daemon on node | MX_VMM_COMPACT=1 | (default) |
| Allocation | VMM from start | cudaMalloc, then compact | cudaMalloc |
| Registration* | 1 call, 0.008s | 1 call, 0.008s | ~500 calls, 1.5s |
| Compaction | none | 0.39s | none |
| Same-node sharing | zero copy | no | no |
| Total overhead* | 0.008s | 0.4s | 1.5s |

*Registration numbers assume NIXL sub-region transfer fix (#1).
Without it, all tiers fall back to per-tensor registration (~2.5s).

VMM compaction exists as a standalone option for deployments that want
near-optimal registration without the GMS daemon dependency. GMS+MX is
the optimal path but requires both systems running.

## Existing Prior Art: dynamo GPU Memory Service

GMS already implements most of these patterns using today's private APIs:

| Proposed API | Current Private API | Used By |
|---|---|---|
| `torch.cuda.ExternalStorage` | `torch._C._construct_storage_from_data_pointer` | GMS, MX |
| Scoped allocator context | `CUDAPluggableAllocator` + `use_mem_pool` | GMS (vLLM, SGLang) |
| Weight allocation hook | GMS `GMSModelLoader` pattern | GMS (vLLM, SGLang) |

### GMS code references

- Tensor from pointer: `lib/gpu_memory_service/client/torch/tensor.py:75`
- Pluggable allocator: `lib/gpu_memory_service/client/torch/allocator.py:57`
- C++ allocator shim: `lib/gpu_memory_service/client/torch/extensions/allocator.cpp`
- vLLM loader integration: `lib/gpu_memory_service/integrations/vllm/model_loader.py:142`
- SGLang integration: `lib/gpu_memory_service/integrations/sglang/`
- VMM utilities: `lib/gpu_memory_service/client/cuda_vmm_utils.py`
- Cross-process sharing: `cuMemImportFromShareableHandle` via DMA-buf FDs

The proposed changes codify patterns already in production across two
independent systems (GMS for same-node sharing, MX for cross-node RDMA),
demonstrating that the need is real and the APIs are proven.

## Who Benefits

- **ModelExpress** - O(1) RDMA registration, GMS integration for zero-copy local sharing
- **dynamo GPU Memory Service** - stabilizes existing private API dependencies
- **NCCL / Gloo** - any collective that needs ibv_reg_mr on torch tensors
- **vLLM** - clean extension point for custom model delivery
- **SGLang** - same benefits, GMS already integrates
- **TensorRT-LLM** - same weight loading patterns
- **Cloud providers** - faster cold start for serverless inference
