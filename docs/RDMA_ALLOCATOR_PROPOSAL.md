# Proposal: RDMA-Aware Weight Allocation for GPU Inference Engines

## Problem

GPU-to-GPU model weight transfer via RDMA (NIXL/UCX) requires registering
GPU memory with the InfiniBand HCA via `ibv_reg_mr`. This is a kernel call
into the IB driver that takes 0.1-10ms per registration depending on region
size. For large models, this dominates cold start time:

| Model | Tensors | Registration Time | Transfer Time |
|---|---|---|---|
| Qwen2.5-3B (6 GB) | 255 | 0.06s | 0.23s |
| DeepSeek-V2-Lite FP8 (16 GB) | 539 | 0.29s | 0.58s |
| Kimi K2.5 NVFP4 (77 GB) | 2644 | 27s | 2.2s |
| DeepSeek-V3 FP8 TP8 (86 GB/worker) | 1386 | 5.6s | 4.6-15.6s |

Registration takes 27 seconds for Kimi K2.5 while the actual RDMA transfer
takes 2.2 seconds. The registration overhead comes from PyTorch's memory
layout: the caching allocator spreads tensors across hundreds of `cudaMalloc`
segments, each requiring a separate `ibv_reg_mr` call.

## Current Workarounds (ModelExpress)

### Pool Registration (shipping)
Discover `cudaMalloc` segment boundaries via `cuMemGetAddressRange_v2` and
register one region per segment instead of one per tensor. Reduces
registrations by ~80% (e.g., 1386 tensors -> 298 allocations). Speedup:
~2.8x on registration. Limited by PyTorch's allocation fragmentation.

### VMM Compaction (prototype)
After weight loading, compact all tensors into a single contiguous CUDA
Virtual Memory range (`cuMemAddressReserve` / `cuMemCreate` / `cuMemMap`).
One `ibv_reg_mr` call total. Uses segment-ordered freeing to limit peak
VRAM overhead to one `cudaMalloc` segment (~2-256 MB).

**Limitations of both approaches:**
- Post-hoc: weights are loaded into fragmented memory, then moved or
  discovered. The fragmentation happens because vLLM and PyTorch don't
  know about RDMA at allocation time.
- Relies on `torch._C._construct_storage_from_data_pointer` (private API)
  to create tensors at arbitrary CUDA addresses.
- Non-contiguous tensors (FP8 TMA scales, MLA projections) require
  special handling as flat byte views.
- Any component that captures raw GPU pointers before compaction gets
  stale references.

## Proposed Changes

### 1. PyTorch: Public External Memory Tensor API

**What:** A public API for creating CUDA tensors backed by externally
managed device memory.

**Why:** `_construct_storage_from_data_pointer` is used in production by
multiple NVIDIA projects (ModelExpress, dynamo GPU Memory Service) but has
no stability guarantee. The need is legitimate and growing: RDMA frameworks,
GPU memory pools, cross-process shared memory, and custom allocators all
need to wrap external device pointers as tensors.

**API surface:**

```python
# Create a storage from a raw device pointer (does not take ownership)
storage = torch.cuda.ExternalStorage(
    data_ptr: int,
    size_bytes: int,
    device: torch.device,
)

# Create a tensor from external storage
tensor = torch.as_tensor_from_storage(
    storage,
    shape: tuple[int, ...],
    stride: tuple[int, ...] | None = None,
    dtype: torch.dtype = torch.float32,
    storage_offset: int = 0,
)
```

**Requirements:**
- Must support all CUDA dtypes including `float8_e4m3fn`, `bfloat16`
  (rules out `__cuda_array_interface__` and DLPack which lack FP8 support)
- Storage must NOT free the underlying memory on garbage collection
- Must work with autograd (tensors are used as `nn.Parameter.data`)
- Should integrate with `torch.cuda.memory_stats()` for accounting
  (external memory tracked separately from the caching allocator)

**Existing usage (private API):**
- `torch._C._construct_storage_from_data_pointer` - used by:
  - NVIDIA dynamo GPU Memory Service (`lib/gpu_memory_service/client/torch/tensor.py`)
  - NVIDIA ModelExpress VMM compaction (`modelexpress/vmm_compact.py`)
- `torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata` - dynamo GMS

### 2. PyTorch: CUDA Pluggable Allocator for Weight Loading

**What:** Extend `torch.cuda.CUDAPluggableAllocator` to support scoped
allocation contexts, enabling inference frameworks to direct weight
allocations into RDMA-friendly memory.

**Why:** The current `CUDAPluggableAllocator` replaces the allocator
globally. For RDMA weight loading, we only want to control allocations
during `load_weights()` and `process_weights_after_loading()`, not
during inference (KV cache, activations, etc.).

**API surface:**

```python
class CUDAAllocationContext:
    """Scoped context that redirects CUDA allocations to a custom allocator."""

    def __init__(
        self,
        alloc_fn: Callable[[int, int, int], int],  # (size, device, stream) -> ptr
        free_fn: Callable[[int, int, int], None],   # (ptr, size, device) -> None
    ): ...

    def __enter__(self) -> "CUDAAllocationContext": ...
    def __exit__(self, *args) -> None: ...
```

**Usage by inference frameworks:**

```python
# VMM-backed allocator
arena = VmmArena(total_model_size, granularity, device_id)

def vmm_alloc(size, device, stream):
    return arena.allocate(size)  # returns VA within the contiguous range

def vmm_free(ptr, size, device):
    pass  # VMM arena manages lifetime, not per-tensor

with CUDAAllocationContext(vmm_alloc, vmm_free):
    model.load_weights(...)       # all allocations go to VMM arena
    process_weights_after_loading(model)  # derived tensors too

# After context: model weights are in one contiguous RDMA-registerable range
# KV cache and activations use the default caching allocator
nixl_agent.register_memory([(arena.base, arena.size)], ...)  # one call
```

**Key property:** Allocations made inside the context go to the custom
allocator. Allocations outside (before/after) use the default caching
allocator. This eliminates the entire post-hoc compaction step.

### 3. vLLM: Weight Allocation Hook

**What:** A hook in vLLM's model loading pipeline that lets external
libraries control how weight memory is allocated.

**Why:** vLLM currently allocates weight memory through PyTorch's default
allocator, which fragments across hundreds of `cudaMalloc` segments.
External model delivery systems (ModelExpress, dynamo GMS) need contiguous
RDMA-friendly memory but can only intervene after the fact.

**Integration point:** The `ModelLoader` base class, specifically around
`load_weights()` and `process_weights_after_loading()`.

**API surface:**

```python
class WeightAllocatorHook:
    """Hook for controlling weight memory allocation."""

    def pre_allocate(self, model: nn.Module, model_config: ModelConfig) -> ContextManager:
        """Return a context manager that controls allocation during weight loading.

        Called before load_weights(). The returned context wraps both
        load_weights() and process_weights_after_loading() so that all
        weight tensors (including derived tensors like MLA projections)
        are allocated through the hook.

        Return nullcontext() to use default allocation.
        """
        return contextlib.nullcontext()

    def post_load(self, model: nn.Module, tensors: dict[str, torch.Tensor]) -> None:
        """Called after all weights are loaded and post-processed.

        Opportunity to register memory, publish metadata, etc.
        The tensors dict contains all weight tensors in their final state.
        """
        pass
```

**Registration:**

```python
# In vLLM config or via environment variable
vllm_config.weight_allocator_hook = ModelExpressAllocatorHook(
    server_address="mx-server:8001",
    use_vmm=True,
)
```

**vLLM implementation changes:**

```python
# In model loading pipeline (simplified)
hook = vllm_config.weight_allocator_hook or DefaultHook()

with hook.pre_allocate(model, model_config):
    loader.load_weights(model, model_config)
    process_weights_after_loading(model, model_config, device)

tensors = collect_module_tensors(model)
hook.post_load(model, tensors)
```

### 4. NIXL: Range-Based Descriptor Matching

**What:** Support sub-region transfers within a registered memory range.

**Why:** Currently, `prep_xfer_dlist` requires exact address+size matches
against registered descriptors. This forces RDMA libraries to register
every individual tensor even when they all live in a single registered
region. With range-based matching, registering one large region would
enable transfers of arbitrary sub-regions within it.

**Current behavior:**
```
register_memory([(0x1000, 4096)])     # register one 4KB region
prep_xfer_dlist([(0x1000, 4096)])     # OK - exact match
prep_xfer_dlist([(0x1000, 1024)])     # FAIL - no exact match for sub-region
prep_xfer_dlist([(0x1400, 512)])      # FAIL - no exact match
```

**Proposed behavior:**
```
register_memory([(0x1000, 4096)])     # register one 4KB region
prep_xfer_dlist([(0x1000, 4096)])     # OK - exact match
prep_xfer_dlist([(0x1000, 1024)])     # OK - within registered range
prep_xfer_dlist([(0x1400, 512)])      # OK - within registered range
```

**Impact:** This single change would make VMM compaction fully effective.
Register the one VMM range, transfer individual tensors as sub-regions.
No need for per-tensor registration. Registration goes from O(N) to O(1)
regardless of tensor count.

## Combined Flow (All Changes Integrated)

```
Source (disk load):
  1. MX creates VMM arena sized to model
  2. vLLM hook wraps load_weights() with VMM allocator context
  3. All weight tensors allocated directly in VMM arena
  4. process_weights_after_loading() derived tensors also in arena
  5. hook.post_load() registers one range with NIXL, publishes metadata
  -> Total registration: 1 ibv_reg_mr call, <1ms

Target (RDMA receive):
  1. MX creates VMM arena sized to source model
  2. Registers one range with NIXL
  3. NIXL RDMA READ of all tensors (sub-region matching)
  4. vLLM hook wraps model init with VMM allocator context
  5. Dummy weights allocated in same arena layout as source
  6. RDMA overwrites in-place, no post-processing needed
  -> Total registration: 1 ibv_reg_mr call, <1ms
  -> No compaction step, no copies, no repointing
```

## Priority Order

1. **NIXL range-based matching** (highest impact, smallest change) -
   enables single-registration VMM without per-tensor workaround
2. **PyTorch external memory API** (removes private API dependency) -
   stabilizes the foundation all other work builds on
3. **vLLM weight allocation hook** (eliminates compaction entirely) -
   the endgame, but requires #1 and #2 first
4. **PyTorch scoped allocator** (ergonomic improvement) - makes #3
   cleaner but not strictly necessary (can use VMM arena + pointer
   arithmetic directly)

## Existing Prior Art: dynamo GPU Memory Service

The dynamo inference framework's GPU Memory Service (GMS) already implements
most of these patterns using today's private APIs. This section documents
exactly what GMS depends on, to motivate why these APIs should be promoted
to first-class public interfaces.

### Private API: `_construct_storage_from_data_pointer`

GMS creates tensors from raw CUDA pointers in
`lib/gpu_memory_service/client/torch/tensor.py`:

```python
# line 75 - creates PyTorch storage aliasing a raw device pointer
storage = torch._C._construct_storage_from_data_pointer(
    data_ptr, device, storage_size_bytes
)
# line 87 - builds typed tensor with shape/stride metadata
return torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(metadata, storage)
```

This is used by `GMSTensorSpec.materialize()` to reconstruct model tensors
from GMS-managed VMM memory. Every RO-mode model load (the common case in
multi-replica deployments) flows through this path.

### CUDAPluggableAllocator for Weight Loading

GMS uses PyTorch's `CUDAPluggableAllocator` to redirect weight allocations
to VMM-managed memory during model loading. The allocator is a C++ extension
(`client/torch/extensions/allocator.cpp`) that routes `malloc`/`free` to
Python callbacks:

```python
# allocator.py - routes weight allocation to GMS VMM memory
_pluggable_alloc = CUDAPluggableAllocator(
    cumem.__file__, "my_malloc", "my_free"
)

def _gms_malloc(size, device, stream):
    return _manager.create_mapping(size=int(size), tag=_tag)  # returns VMM VA

def _gms_free(ptr, size, device, stream):
    _manager.destroy_mapping(int(ptr))
```

The scoped allocation pattern this proposal describes is already implemented
in GMS's vLLM integration (`integrations/vllm/model_loader.py`):

```python
# model_loader.py lines 142-151 - allocate weights via GMS pool
with use_mem_pool(pool, device=target_device):
    with target_device:
        model = initialize_model(vllm_config=vllm_config, model_config=model_config)
    default_loader.load_weights(model, model_config)
    process_weights_after_loading(model, model_config, target_device)
```

The `use_mem_pool` context manager directs all CUDA allocations during weight
loading to the GMS-backed VMM arena. After the context exits, subsequent
allocations (KV cache, activations) use the default caching allocator.

### CUDA VMM API

GMS wraps the full VMM lifecycle in `client/cuda_vmm_utils.py`:
- `reserve_va()` -> `cuMemAddressReserve`
- `map_to_va()` -> `cuMemMap`
- `set_access()` -> `cuMemSetAccess`
- `unmap()` -> `cuMemUnmap`
- `import_handle_from_fd()` -> `cuMemImportFromShareableHandle`
  (for cross-process memory sharing via DMA-buf file descriptors)

### SGLang Integration

GMS also integrates with SGLang (`integrations/sglang/`) using the same
patterns, demonstrating that the allocator hook approach works across
multiple inference frameworks, not just vLLM.

### What This Means for the Proposal

The proposed changes are not speculative - they codify patterns that are
already in production:

| Proposed API | Current Private API | Used By |
|---|---|---|
| `torch.cuda.ExternalStorage` | `torch._C._construct_storage_from_data_pointer` | GMS, ModelExpress |
| Scoped allocator context | `torch.cuda.memory.use_mem_pool` + `CUDAPluggableAllocator` | GMS (vLLM, SGLang) |
| Weight allocation hook | GMS `GMSModelLoader` pattern | GMS (vLLM, SGLang) |

The difference is that GMS does cross-process sharing (one process loads,
others import via shared VMM handles), while ModelExpress does cross-node
RDMA transfer. Both need the same foundation: control over where weight
memory lives so it can be efficiently registered for external access.

Promoting these to public APIs eliminates the private-API fragility for
both projects simultaneously, and opens the pattern to every RDMA/sharing
framework in the ecosystem.

## Who Benefits

- **ModelExpress** - zero-overhead RDMA registration, no post-hoc hacks
- **dynamo GPU Memory Service** - stabilizes existing production code paths
- **NCCL / Gloo** - any collective that needs ibv_reg_mr on torch tensors
- **vLLM** - clean extension point for custom model delivery (replaces ad-hoc loader patterns)
- **SGLang** - same benefits as vLLM, GMS already integrates
- **TensorRT-LLM** - same weight loading patterns
- **Cloud providers** - faster cold start for serverless inference
