# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Classic ``cudaMalloc``-backed CUDA MemPool for NIXL-registered buffers.

With ``PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`` PyTorch's caching
allocator hands out VMM-backed (``cuMemCreate`` + ``cuMemMap``) virtual ranges.
``ibv_reg_mr`` on such a range succeeds, but the mlx5 HCA's MMU walk at WRITE
time completes with ``syndrome 0x4`` ("Local protection"), because
``nvidia_peermem``'s ``get_pages`` cannot pin a VA that spans multiple
``cuMemCreate`` handles - UCX tears the endpoint down and NIXL surfaces it as
``REMOTE_DISCONNECT``.

Buffers handed to ``NixlTransferManager.register_tensors`` (i.e.
``nixl_agent.register_memory``) must therefore come from a classic, contiguous
``cudaMalloc`` block. This module exposes a ``MemPool`` backed by a
``CUDAPluggableAllocator`` that calls ``cudaMalloc`` / ``cudaFree`` directly,
plus :func:`classic_cuda_alloc` to scope specific allocations into it. Everything
else in the process keeps using the default (expandable-segments) caching
allocator.

The reshard receiver allocates its staging / capture buffers at REFIT time -
after the engine's CUDA graphs are captured - so those buffers must also stay out
of the caching allocator's graph pool; the classic pool satisfies both.

MX's ``vmm`` arena (``MX_VMM_ARENA=1``) is the more efficient single-region
alternative used by the boot-time P2P loader, but it is gated on the arena
extension; this classic pool is the always-available path with no deployment
prerequisite.
"""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# TileLang ships a libcudart stub that proxies to the real CUDA runtime via
# dlsym(RTLD_DEFAULT, ...). If the stub's own symbols are found first (nothing
# has loaded the real libcudart globally yet) its self-check fails and the stub
# aborts - which is what we hit the moment we enter the classic-cudaMalloc
# MemPool. Preloading the real library with RTLD_GLOBAL makes dlsym find it first.
try:
    ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass

import torch  # noqa: E402
from torch.utils.cpp_extension import load_inline  # noqa: E402

_SOURCE = r"""
#include <cuda_runtime.h>
#include <cstddef>
extern "C" {
void* mx_classic_alloc(ptrdiff_t size, int device, void* stream) {
    (void) stream;
    int prev = -1;
    cudaGetDevice(&prev);
    cudaSetDevice(device);
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, (size_t) size);
    if (prev >= 0) cudaSetDevice(prev);
    if (err != cudaSuccess) return nullptr;
    return ptr;
}
void mx_classic_free(void* ptr, ptrdiff_t size, int device, void* stream) {
    (void) size; (void) stream;
    int prev = -1;
    cudaGetDevice(&prev);
    cudaSetDevice(device);
    cudaFree(ptr);
    if (prev >= 0) cudaSetDevice(prev);
}
}
"""

_pools: "dict[int, torch.cuda.MemPool]" = {}
_allocator_wrapper: "torch.cuda.memory.CUDAPluggableAllocator | None" = None


def _get_allocator() -> "torch.cuda.memory.CUDAPluggableAllocator":
    global _allocator_wrapper
    if _allocator_wrapper is not None:
        return _allocator_wrapper
    module = load_inline(
        name="mx_classic_cuda_alloc",
        cpp_sources=[_SOURCE],
        functions=[],
        extra_cflags=["-O2"],
        with_cuda=True,
    )
    so_path = Path(module.__file__)
    _allocator_wrapper = torch.cuda.memory.CUDAPluggableAllocator(
        str(so_path), "mx_classic_alloc", "mx_classic_free"
    )
    return _allocator_wrapper


def _get_pool(device_index: int) -> "torch.cuda.MemPool":
    """Return the classic pool for ``device_index``, creating it on first use.

    A MemPool registers with the device current at construction and can only be
    reused there: entering it repeatedly on any other device trips the caching
    allocator's pool use-count assertion. So each device gets its own pool,
    constructed with that device current.
    """
    pool = _pools.get(device_index)
    if pool is None:
        allocator = _get_allocator()
        with torch.cuda.device(device_index):
            pool = torch.cuda.MemPool(allocator.allocator())
        _pools[device_index] = pool
    return pool


def _device_index(device: "torch.device | str | int | None") -> int:
    """Resolve ``device`` to the CUDA device index an allocation on it lands on."""
    if isinstance(device, int):
        return device
    index = None if device is None else torch.device(device).index
    return torch.cuda.current_device() if index is None else index


@contextmanager
def classic_cuda_alloc(
    device: "torch.device | str | int | None" = None,
) -> Iterator[None]:
    """Scope CUDA allocations on ``device`` into the classic-``cudaMalloc`` MemPool.

    Use when the resulting tensor's address must be a contiguous ``cudaMalloc``
    block - required for NIXL-registered buffers (see module docstring).

    The pool applies only to allocations on ``device``; ``None`` or a device
    without an index means the current CUDA device, which is where such a
    device allocates. Pass the device the buffer is allocated on whenever it
    may differ from the current one, or the allocation silently falls back to
    the default caching allocator."""
    device_index = _device_index(device)
    with torch.cuda.use_mem_pool(_get_pool(device_index), device=device_index):
        yield
