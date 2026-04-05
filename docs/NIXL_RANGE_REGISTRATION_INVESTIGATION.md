# NIXL Range-Aware Registration: Investigation Report

## Summary

Per-tensor `ibv_reg_mr` calls dominate NIXL registration time for large models.
We implemented a range cache in the UCX backend that reuses a single `ucp_mem_map`
registration for multiple tensor descriptors within the same contiguous memory
region (e.g., VMM-compacted model weights). Registration time dropped from 2.4s
to 0.017s (140x). However, RDMA transfers fail with the shared `ucp_mem_h` -
a "Local protection error" from the IB HCA.

## Context

- **Repository**: ModelExpress (`ai-dynamo/modelexpress`), branch `nnoble/vmm-compaction`
- **NIXL fork**: `nicolasnoble/nixl`, branch `nnoble/range-aware-registration`
- **Model**: NousResearch/Hermes-3-Llama-3.1-405B-FP8, TP8
- **Hardware**: nScale B200 cluster, InfiniBand, pinned nodes (hx78c -> l9nsv)
- **Per-worker stats**: 1264 tensors, 51.32 GB, 524 cudaMalloc segments, 504 non-contiguous (`__storage` views)

## The Registration Problem

NIXL's `register_memory` calls `ucp_mem_map` (which calls `ibv_reg_mr`) once
per tensor descriptor. For 1264 tensors at 51 GB, this takes 2.4s. The
`ucp_mem_map` path is explicit registration - it does NOT go through UCX's
rcache (registration cache). The rcache only applies to implicit registrations
during zero-copy transfer operations.

### Codepath (NIXL 1.0.0 / main branch)

```
Python register_memory([tensor_list])
  -> nixlAgent::registerMem()           [src/core/nixl_agent.cpp:443]
    -> nixlLocalSection::addDescList()   [src/infra/nixl_memory_section.cpp:133]
      -> for each descriptor (line 152):
        -> nixlUcxEngine::registerMem()  [src/plugins/ucx/ucx_backend.cpp:958]
          -> nixlUcxContext::memReg()    [src/plugins/ucx/ucx_utils.cpp:564]
            -> ucp_mem_map()             // explicit, no rcache
              -> ibv_reg_mr()            // ~2ms per call
          -> nixlUcxContext::packRkey()
            -> ucp_rkey_pack()           // per-tensor rkey
```

Each descriptor gets its own `nixlUcxPrivateMetadata` containing:
- `nixlUcxMem mem` - holds `ucp_mem_h`, base address, size
- `nixl_blob_t rkeyStr` - packed rkey for remote access

The transfer path (`sendXferRangeBatch`, line 1220) uses per-descriptor metadata:
```cpp
ep.read(raddr, rmd->getRkey(worker_id), laddr, *lmd->mem, lsize, req)
```
Where `lmd->mem` provides the local `ucp_mem_h` passed as `UCP_OP_ATTR_FIELD_MEMH`.

### Why a single registration should work

UCX's `ucp_mem_map` on a contiguous range produces:
- One `ucp_mem_h` valid for any sub-region local operation within the range
- One packed rkey valid for any sub-region remote access within the range

The IB MR (memory region) from `ibv_reg_mr` covers `[base, base+size]`. The
associated lkey is valid for any RDMA operation targeting addresses within that
range. Sub-region access is explicitly supported by the IB spec.

## The Range Cache Fix

### Changes (nicolasnoble/nixl, branch nnoble/range-aware-registration)

**`src/plugins/ucx/ucx_backend.h`**:
- `nixlUcxPrivateMetadata::mem` changed from `nixlUcxMem` to `std::shared_ptr<nixlUcxMem>`
- Added `RegRange { shared_ptr<nixlUcxMem> mem; nixl_blob_t rkeyStr; }` struct
- Added `std::map<uintptr_t, RegRange> regRangeCache` to `nixlUcxEngine`

**`src/plugins/ucx/ucx_backend.cpp`**:
- `registerMem()`: Before calling `ucp_mem_map`, searches regRangeCache using
  `upper_bound` to find if `[addr, addr+len)` falls within an existing range.
  If yes, shares the `shared_ptr<nixlUcxMem>` and rkey string. If no, performs
  normal registration and adds to cache.
- `deregisterMem()`: Uses `shared_ptr::use_count()` for lifetime management.
  Only calls `ucp_mem_dereg` when the last descriptor sharing the range is removed.
- `sendXferRangeBatch()`: Updated `lmd->mem` to `*lmd->mem` (dereference shared_ptr).

### Registration Results (Hermes 405B FP8 TP8, nScale B200 IB)

VMM two-phase registration (register VMM range first, then per-tensor):

| Phase | Without cache | With cache |
|---|---|---|
| Range registration (1 region) | 0.008s | 0.008s |
| Per-tensor registration (1264 regions) | 2.4s | **0.009s** |
| Total `register_tensors` | 2.4s | **0.017s** |

The cache works: 1264 tensors registered in 0.017s instead of 2.4s. Each per-tensor
registration hits the range cache and shares the VMM range's `ucp_mem_h` + rkey.

## The Transfer Failure

RDMA READ fails with "Local protection error" when using the shared `ucp_mem_h`:

```
UCX ERROR Local protection error on mlx5_0:1/IB (synd 0x4 vend 0x33 hw_synd 0/157)
UCX ERROR RC QP 0xf6e wqe[1]: RDMA_READ s--
  [rva 0x78a43c000000 rkey 0x524b00]
  [va 0xc8ebe00000 len 386812080 lkey 0x15500]
  [rqpn 0xa77b dlid=1579 sl=0 port=1 src_path_bits=0]
```

- `rva` / `rkey`: Remote (source) address and rkey - from pool-reg, per-allocation. Should be valid.
- `va` / `lkey`: Local (target) address and lkey - from VMM shared registration.
- `va 0xc8ebe00000` IS within the VMM range (base ~0xbd18000000, size ~51 GB, end ~0xCA6DD60000).
- `lkey 0x15500` is from the VMM range's `ucp_mem_h`.
- `len 386812080` (~369 MB) is a `__storage` view (flat byte view of non-contiguous tensor storage).

### What works vs what doesn't

| Configuration | Registration | Transfer |
|---|---|---|
| Pool-reg (no cache, separate memh per allocation) | 1.65s | OK |
| Pool-reg (with NIXL cache, no VMM) | 3.5s* | OK |
| VMM + cache (shared memh from range) | **0.017s** | **FAILS** |
| VMM without cache (separate memh per tensor) | 2.4s | OK (Qwen 3B), FAILS (405B)** |

*Pool-reg allocations don't overlap, so the cache provides no benefit.
**The 405B transfer failure without cache may be a separate issue (gRPC timeouts on some workers).

### The question

Why does `ucp_get_nbx` with `UCP_OP_ATTR_FIELD_MEMH` fail when the memh covers
a superset of the local buffer address range? The IB spec allows sub-region access
within an MR. Possible explanations:

1. **UCX validates memh base/size against the transfer address and rejects mismatches.**
   The memh has base=VMM_start, size=51GB, but the transfer laddr is at some offset
   within that range. UCX might require exact match (base == laddr) rather than containment.

2. **The lkey derivation in UCX depends on the memh's base address.**
   UCX might compute the lkey offset from the memh base, and if the IB HCA's MR
   registration used a different base (unlikely since we control the registration),
   the offset calculation would be wrong.

3. **MULTI_SEND flag interaction.** The transfer path sets `UCP_OP_ATTR_FLAG_MULTI_SEND`
   alongside `UCP_OP_ATTR_FIELD_MEMH`. The multi-send batching might make assumptions
   about memh that break with a shared superset memh.

4. **nvidia-peermem / GPU memory type mismatch.** VMM memory (`cuMemCreate`/`cuMemMap`)
   might be registered differently by nvidia-peermem than cudaMalloc memory. The
   `ibv_reg_mr` might succeed but produce an MR with different properties.

## How to Reproduce

### Prerequisites
- nScale B200 cluster with IB
- Docker Hub access (nnoble447)
- Namespace `nnoble-mx-tensor-walk` with `model-cache` PVC (2 TB, has Hermes cached)

### Image
```bash
# Image with custom NIXL + MX VMM compaction:
nnoble447/modelexpress-client:vmm-nixl-range
# sha256:a0ead5cfb08de1b76ac844c3ed70d62b8ab9891a1a13ee46526e41f01cb08e6e
```

This image was built by:
1. Starting `nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.0` with GPU
2. `apt-get install cuda-nvcc-12-9` (for nvlink)
3. `pip install` from `nicolasnoble/nixl` branch `nnoble/range-aware-registration`
   with `-Ddisable_gds_backend=true`
4. Copying new .so files from `.nixl_cu12.mesonpy.libs/` over `/opt/nvidia/nvda_nixl/`
5. `pip install` ModelExpress client from `nnoble/vmm-compaction` branch

### Deploy source (hx78c)
```bash
# Source loads from disk (pool-reg, no VMM)
# Uses env: MX_POOL_REG=1, MX_VMM_COMPACT=0
# See tests/k8s-vmm-benchmark.yaml for full manifest
# Wait ~20 min for 405B warmup
kubectl -n nnoble-mx-tensor-walk logs hermes-source | grep "Status -> READY"
```

### Deploy target with VMM + range cache
```bash
# Target uses VMM compaction + NIXL range cache
# Env: MX_POOL_REG=1, MX_VMM_COMPACT=1
# Registration succeeds in 0.017s
# Transfer fails with Local protection error
```

### Deploy target with pool-reg only (control)
```bash
# Target uses pool-reg, no VMM
# Env: MX_POOL_REG=1, MX_VMM_COMPACT=0
# Registration: 1.65s, Transfer: OK
```

## Standalone Reproduction (smaller scale)

The two-pod VMM transfer test at `tests/test_vmm_transfer.py` transfers 1 GB
of synthetic tensors (100 tensors, float16) from VMM source to per-tensor target.
This test PASSES - suggesting the failure is scale-dependent or related to
`__storage` view tensors specifically.

```bash
# Create configmap from test script
kubectl -n nnoble-mx-tensor-walk create configmap vmm-transfer-test \
  --from-file=test_vmm_transfer.py=tests/test_vmm_transfer.py

# Apply test manifest
kubectl apply -f tests/k8s-vmm-transfer-test.yaml
```

## Benchmark Summary (all configs, Hermes 405B FP8 TP8)

| Config | Compaction | Registration | Transfer | Total Receive |
|---|---|---|---|---|
| Per-tensor baseline | - | 2.6s | 1.9-12s | 5.7-15.7s |
| Pool-reg (524 regions) | - | 1.65s | 1.9-12s | 4.7-17.2s |
| VMM + per-tensor reg | 0.39s | 3.0s | same | same |
| VMM + range cache | 0.39s | **0.017s** | **FAILS** | - |
| VMM + range cache (projected) | 0.39s | 0.017s | same | **~0.4s + transfer** |

## Files

- NIXL fix: `nicolasnoble/nixl`, branch `nnoble/range-aware-registration`
  - `src/plugins/ucx/ucx_backend.h` - shared_ptr memh, RegRange cache
  - `src/plugins/ucx/ucx_backend.cpp` - registerMem cache lookup, deregisterMem refcount
- MX VMM compaction: `ai-dynamo/modelexpress`, branch `nnoble/vmm-compaction`
  - `modelexpress_client/python/modelexpress/vmm_compact.py` - VMM arena, segment-ordered compaction
  - `modelexpress_client/python/modelexpress/nixl_transfer.py` - two-phase registration
  - `modelexpress_client/python/modelexpress/vllm_loader.py` - integration
- Proposal: `docs/RDMA_ALLOCATOR_PROPOSAL.md` - upstream changes for PyTorch/vLLM/NIXL
- Tests: `tests/test_vmm_nixl.py`, `tests/test_vmm_transfer.py`
