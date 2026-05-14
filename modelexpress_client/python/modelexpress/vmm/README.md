<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# VMM Arena

A CUDA Virtual Memory Management (VMM) arena that backs PyTorch tensor
allocations during weight load, so the entire load range can be registered
to NIXL as a **single memory region (MR)** at end-of-load. The wire path
uses one `ibv_reg_dmabuf_mr` call instead of one per tensor, which
collapses the kernel's MR setup cost and reduces RDMA fast-path overhead
on transfers that touch hundreds or thousands of tensors.

## Quick start

Set one environment variable on the worker process:

```
MX_VMM_ARENA=1
```

That is the only knob. The arena enables itself on supported devices and
falls back silently when the underlying C extension is missing.

### Required deployment flag

UCX's `cuda_copy_md` path probes allocations with `cuMemGetAddressRange`,
which returns per-handle bounds rather than the full reserve when called
against a multi-handle VMM range. Without the override, UCX truncates
transfers to a single physical chunk. Until the upstream fix lands
(openucx/ucx#11461), set:

```
UCX_CUDA_COPY_REG_WHOLE_ALLOC=off
```

## Why this exists

Without an arena, NIXL registers each tensor as its own MR. A
Hermes-405B-FP8 worker has ~3000 tensors per rank, and the kernel pays a
per-MR cost in registration time and in HCA translation table state. With
the arena, every tensor lives inside one contiguous CUDA VA range and we
register once: `cuMemGetHandleForAddressRange` produces a single dmabuf,
`ibv_reg_dmabuf_mr` consumes it, and the resulting lkey/rkey covers every
live allocation in the range.

Observed effects on Blackwell + ConnectX over InfiniBand:

- Single-MR registration regardless of tensor count.
- Lower peak VRAM during load; on Hermes 405B FP8 TP=8 the old design
  hit ~64 GiB per rank, the arena holds ~51 GiB.
- Wire rates unchanged or slightly improved (single-MR setup is closer
  to the line rate floor).

## How it works

The arena reserves a 16 TiB VA range up front via
`cuMemAddressReserve` (16 TiB has no physical cost). Each
`mx_malloc(size)` call:

1. Bump-assigns a VA slot of the requested size, aligned to the device's
   `cuMemGetAllocationGranularity` minimum (typically 2 MiB).
2. `cuMemCreate` allocates a physical handle of that size.
3. `cuMemMap` maps the handle at the assigned VA.
4. `cuMemSetAccess` grants read/write to the active device.

The C extension exposes `_mx_malloc` and `_mx_free` to PyTorch's
`CUDAPluggableAllocator`. The Python `use_arena(arena, device)` context
manager installs the allocator and stashes the arena on a thread-local so
the C shims can dispatch without holding the GIL on the hot path. `mx_free`
runs `cuMemUnmap` + `cuMemRelease` and clears the entry.

At end-of-load, `NixlTransferManager.register_arena(arena, tensors)` calls
`cuMemGetHandleForAddressRange(base, used)` to produce a dmabuf spanning
the bump range. Holes from mid-load `free` calls are tolerated: the
dmabuf attach pins the currently-mapped physical pages and the HCA
translation table survives subsequent CUDA-side unmaps.

## Files in this directory

| File | Role |
|---|---|
| `arena.py` | `VmmArena` (reservation + bump + live-allocation table). `VmmBackend` Protocol. `_StubBackend` for unit tests. |
| `backend.py` | `CudaVmmBackend` (real CUDA driver calls via cuda-python). Each allocation rolls back cleanly on partial failure. |
| `hook.py` | `use_arena` context manager. `ARENA_AVAILABLE` runtime flag. Installs the PyTorch `CUDAPluggableAllocator`. |
| `_alloc_ext.cpp` | C extension. Per-arena `ArenaState` capsule with atomic bump pointer, mutex-protected live map, in-flight counter. `Py_BEGIN_ALLOW_THREADS` around the CAS and the map ops. |
| `README.md` | This file. |

## Build

The C extension is built as **optional** by `setup.py`. If a working C++
compiler is present (`g++` or `$CXX`), the extension ships with the wheel
and `MX_VMM_ARENA=1` activates the fast path. If the build fails, the
wheel still installs (pure-Python), `ARENA_AVAILABLE` is `False`, and
`MX_VMM_ARENA=1` logs a warning and falls back to per-tensor registration.

## Runtime fallback

`MX_VMM_ARENA=1` is a request, not a guarantee. The runtime quietly drops
back to per-tensor NIXL registration when any of the following are true:

- The `_alloc_ext` C extension failed to build or import.
- The device does not support VMM (compute capability < 7.0).
- `register_arena` is called against an empty arena (used == 0).

In each case the loader logs a warning and the load completes via the
pool-reg path.

## Tests

- `tests/test_vmm_arena.py`: arena bump-allocator math against the stub
  backend (no CUDA required).
- `tests/test_vmm_backend.py`: `CudaVmmBackend` driver-call sequencing.
  CUDA-gated tests skip on hosts without a working CUDA device.
- `tests/test_vmm_hook.py`: `use_arena` context wiring, allocator install,
  fallback when the extension is absent.

## Agent handoff notes

Reading order for a fresh agent picking this up:

1. This README (the high-level mechanism).
2. `arena.py` (the bump table, the in-flight contract, and how holes are
   represented; this is the core data structure).
3. `hook.py` (the PyTorch pluggable-allocator wiring and the thread-local
   active-arena lookup; this is the seam to PyTorch).
4. `_alloc_ext.cpp` (the GIL-released hot path; only relevant when
   debugging a free-while-allocating race or a layout issue).
5. `backend.py` (CUDA driver calls; only relevant when debugging a
   `cuMem*` failure path).
6. `../engines/vllm/loader.py::_maybe_enter_vmm_arena` (how the arena is
   plumbed into vLLM's load envelope; the engine-side seam).

Key invariants:

- The arena owns the VA range. A `free` unmaps and releases the physical
  handle but does **not** free the VA. The bump pointer monotonically
  advances; reuse is not supported within an arena lifetime.
- All `_alloc_ext` callbacks return a Python integer pointer cast to
  `void*`. The C shim never holds a `torch.Tensor` reference; the arena
  is responsible for keeping handles alive.
- `register_arena` must be called once at end-of-load, before the
  receiver starts pulling tensors. Calling it on an empty arena is safe
  (warns and falls back).

Validated configurations:

- Blackwell B200 + ConnectX over InfiniBand, single-pod and P2P.
- vLLM 0.x with `CUDAPluggableAllocator` (PyTorch 2.4+).
- TP=8 ranks running concurrently against independent arenas.

Known limitations:

- VMM arena is x86_64-Linux only (CUDA driver constraint).
- Bump pointer is monotonic. Long-lived processes that load and unload
  many models within one arena lifetime will fragment the VA but never
  reclaim it. The arena is designed to be torn down between loads.
- The `_alloc_ext` extension targets Python 3.10+ with the `Py_mod_gil`
  slot for free-threaded Python (3.13+).
