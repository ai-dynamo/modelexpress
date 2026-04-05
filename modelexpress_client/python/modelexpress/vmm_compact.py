# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VMM-based tensor compaction for NIXL registration optimization.

After model weights are loaded and post-processed, this module compacts
all tensors into a single contiguous CUDA virtual memory range. This
reduces NIXL registration from thousands of ibv_reg_mr calls to one,
eliminating the dominant startup cost (~27s for Kimi K2.5's 2644 tensors).

Uses the CUDA Virtual Memory Management API (cuMemAddressReserve /
cuMemCreate / cuMemMap) to reserve a contiguous VA range without
requiring 2x physical memory. Tensors are moved segment-by-segment:
all tensors from a single cudaMalloc segment are copied into the VMM
range, then the segment is freed via empty_cache(), making room for the
next batch. Peak overhead is one cudaMalloc segment (~2-256 MB).

Toggle: MX_VMM_COMPACT=1 (default: disabled while experimental).
Requires CUDA 10.2+ for VMM API availability.
"""

from __future__ import annotations

import ctypes
import logging
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn

logger = logging.getLogger("modelexpress.vmm_compact")

VMM_COMPACT_ENABLED = os.environ.get("MX_VMM_COMPACT", "0") == "1"

# CUDA driver constants
CU_MEM_ALLOCATION_TYPE_PINNED = 0x1
CU_MEM_LOCATION_TYPE_DEVICE = 0x1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3

_cuda = None


class _CUmemAllocationProp(ctypes.Structure):
    class _Location(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_int),
            ("id", ctypes.c_int),
        ]

    class _AllocFlags(ctypes.Structure):
        _fields_ = [
            ("compressionType", ctypes.c_ubyte),
            ("gpuDirectRDMACapable", ctypes.c_ubyte),
            ("usage", ctypes.c_ushort),
            ("reserved", ctypes.c_ubyte * 4),
        ]

    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _Location),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocFlags),
    ]


class _CUmemAccessDesc(ctypes.Structure):
    class _Location(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_int),
            ("id", ctypes.c_int),
        ]

    _fields_ = [
        ("location", _Location),
        ("flags", ctypes.c_int),
    ]


def _get_cuda() -> ctypes.CDLL:
    global _cuda
    if _cuda is None:
        _cuda = ctypes.CDLL("libcuda.so")
    return _cuda


def _check(ret: int, msg: str) -> None:
    if ret != 0:
        raise RuntimeError(f"CUDA VMM error ({ret}): {msg}")


def _get_granularity(device_id: int) -> int:
    cuda = _get_cuda()
    prop = _CUmemAllocationProp()
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id
    granularity = ctypes.c_size_t()
    ret = cuda.cuMemGetAllocationGranularity(
        ctypes.byref(granularity), ctypes.byref(prop), ctypes.c_ulonglong(0)
    )
    _check(ret, "cuMemGetAllocationGranularity")
    return granularity.value


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _find_tensor_segments(
    tensors: dict[str, torch.Tensor],
) -> dict[int, list[str]]:
    """Map each tensor to its cudaMalloc segment via cuMemGetAddressRange.

    Returns: {segment_base_addr: [tensor_name, ...]}
    """
    cuda = _get_cuda()
    segments: dict[int, list[str]] = defaultdict(list)

    for name, t in tensors.items():
        base = ctypes.c_uint64()
        size = ctypes.c_size_t()
        ret = cuda.cuMemGetAddressRange_v2(
            ctypes.byref(base),
            ctypes.byref(size),
            ctypes.c_uint64(t.data_ptr()),
        )
        _check(ret, f"cuMemGetAddressRange for '{name}'")
        segments[base.value].append(name)

    return dict(segments)


def _tensor_from_pointer(
    data_ptr: int,
    shape: list[int],
    stride: list[int],
    dtype: torch.dtype,
    device_index: int,
    storage_size_bytes: int,
) -> torch.Tensor:
    """Create a torch.Tensor aliasing a raw CUDA pointer.

    Uses torch._C._construct_storage_from_data_pointer (same approach as
    dynamo's GPU Memory Service). The tensor does NOT own the memory.
    """
    device = torch.device("cuda", device_index)
    storage = torch._C._construct_storage_from_data_pointer(
        data_ptr, device, storage_size_bytes
    )
    t = torch.empty(0, dtype=dtype, device=device)
    t.set_(storage, 0, shape, stride)
    return t


class VmmArena:
    """A contiguous CUDA VMM virtual address arena.

    Reserves a single VA range. Physical pages are mapped incrementally
    as tensors are moved in. Must stay alive for the tensor lifetime.
    """

    def __init__(self, total_size: int, granularity: int, device_id: int):
        self._device_id = device_id
        self._granularity = granularity
        self._total_size = _align_up(total_size, granularity)
        self._handles: list[ctypes.c_uint64] = []
        self._va_base = 0
        self._mapped_up_to = 0  # byte offset of next unmapped region

        cuda = _get_cuda()
        va_ptr = ctypes.c_uint64()
        ret = cuda.cuMemAddressReserve(
            ctypes.byref(va_ptr),
            ctypes.c_size_t(self._total_size),
            ctypes.c_size_t(granularity),
            ctypes.c_uint64(0),
            ctypes.c_ulonglong(0),
        )
        _check(ret, f"cuMemAddressReserve({self._total_size / 1e9:.2f} GB)")
        self._va_base = va_ptr.value

    @property
    def va_base(self) -> int:
        return self._va_base

    @property
    def total_size(self) -> int:
        return self._total_size

    def ensure_mapped(self, up_to_offset: int) -> None:
        """Map physical pages up to the given byte offset.

        Idempotent - only maps pages beyond what's already mapped.
        Mapping is done in granularity-sized chunks.
        """
        target = _align_up(up_to_offset, self._granularity)
        if target <= self._mapped_up_to:
            return

        cuda = _get_cuda()
        map_offset = self._mapped_up_to
        map_size = target - map_offset

        prop = _CUmemAllocationProp()
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self._device_id
        prop.allocFlags.gpuDirectRDMACapable = 1

        handle = ctypes.c_uint64()
        ret = cuda.cuMemCreate(
            ctypes.byref(handle),
            ctypes.c_size_t(map_size),
            ctypes.byref(prop),
            ctypes.c_ulonglong(0),
        )
        _check(ret, f"cuMemCreate({map_size / 1e6:.1f} MB)")
        self._handles.append(handle)

        target_va = self._va_base + map_offset
        ret = cuda.cuMemMap(
            ctypes.c_uint64(target_va),
            ctypes.c_size_t(map_size),
            ctypes.c_size_t(0),
            handle,
            ctypes.c_ulonglong(0),
        )
        _check(ret, f"cuMemMap at offset {map_offset}")

        access = _CUmemAccessDesc()
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = self._device_id
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        ret = cuda.cuMemSetAccess(
            ctypes.c_uint64(target_va),
            ctypes.c_size_t(map_size),
            ctypes.byref(access),
            ctypes.c_size_t(1),
        )
        _check(ret, f"cuMemSetAccess at offset {map_offset}")

        self._mapped_up_to = target

    def close(self) -> None:
        """Unmap and release all physical allocations."""
        cuda = _get_cuda()
        if self._va_base:
            cuda.cuMemUnmap(
                ctypes.c_uint64(self._va_base),
                ctypes.c_size_t(self._mapped_up_to),
            )
            cuda.cuMemAddressFree(
                ctypes.c_uint64(self._va_base),
                ctypes.c_size_t(self._total_size),
            )
            self._va_base = 0
        for h in self._handles:
            cuda.cuMemRelease(h)
        self._handles.clear()


def compact_tensors(
    model: nn.Module,
    tensors: dict[str, torch.Tensor],
    device_id: int,
) -> tuple[int, int, dict[str, torch.Tensor], VmmArena]:
    """Compact model tensors into a single contiguous CUDA VA range.

    Moves tensors segment-by-segment: discovers which tensors share each
    cudaMalloc segment, copies all tensors from one segment into the VMM
    range, then frees that segment before moving to the next. Peak memory
    overhead is one cudaMalloc segment (~2-256 MB), not the full model.

    Args:
        model: The nn.Module whose parameter .data will be repointed.
        tensors: Name -> tensor dict from _collect_module_tensors.
        device_id: CUDA device ordinal.

    Returns:
        (va_base, va_size, new_tensors, arena).
        Caller must keep the arena alive for the lifetime of the tensors.
    """
    if not tensors:
        return 0, 0, tensors, None  # type: ignore[return-value]

    # Guard: storage views require reconstructing non-contiguous views
    # after moving, which is not yet supported.
    storage_views = [n for n in tensors if n.endswith(".__storage")]
    if storage_views:
        logger.warning(
            f"VMM compaction skipped: {len(storage_views)} storage-view tensors "
            f"present (non-contiguous weights). Falling back to pool registration."
        )
        return 0, 0, tensors, None  # type: ignore[return-value]

    torch.cuda.set_device(device_id)
    total_start = time.perf_counter()

    granularity = _get_granularity(device_id)
    logger.info(
        f"VMM granularity: {granularity} bytes ({granularity / 1024:.0f} KB)"
    )

    # Phase 1: Discover cudaMalloc segments
    seg_start = time.perf_counter()
    segments = _find_tensor_segments(tensors)
    seg_time = time.perf_counter() - seg_start
    logger.info(
        f"[TIMING] Segment discovery: {seg_time:.3f}s, "
        f"{len(tensors)} tensors across {len(segments)} cudaMalloc segments"
    )

    # Phase 2: Plan dense layout - pack tensors tightly, only align the
    # total to granularity. No per-tensor alignment waste.
    # Process segments in order so all tensors from one segment are adjacent
    # in the VA range (helps with locality, though not strictly required).
    layout: dict[str, tuple[int, int]] = {}  # name -> (offset, data_bytes)
    current_offset = 0
    segment_order: list[tuple[int, list[str]]] = sorted(segments.items())

    for _seg_base, seg_tensor_names in segment_order:
        for name in seg_tensor_names:
            t = tensors[name]
            data_bytes = t.numel() * t.element_size()
            layout[name] = (current_offset, data_bytes)
            current_offset += data_bytes
        # Align to granularity at segment boundaries so each segment's
        # worth of VMM pages can be mapped as a single cuMemCreate call.
        current_offset = _align_up(current_offset, granularity)

    total_va = current_offset
    total_data = sum(t.numel() * t.element_size() for t in tensors.values())
    overhead = ((total_va - total_data) / total_data * 100) if total_data else 0

    logger.info(
        f"VMM layout: {len(tensors)} tensors, "
        f"{total_data / 1e9:.2f} GB data, "
        f"{total_va / 1e9:.2f} GB VA ({overhead:.1f}% alignment overhead, "
        f"aligned at {len(segments)} segment boundaries)"
    )

    # Phase 3: Reserve VA range (no physical memory yet)
    reserve_start = time.perf_counter()
    arena = VmmArena(total_va, granularity, device_id)
    reserve_time = time.perf_counter() - reserve_start
    logger.info(f"[TIMING] VA reserve: {reserve_time:.3f}s, base=0x{arena.va_base:x}")

    # Build reverse map: data_ptr -> model refs for repointing
    ptr_to_model_refs: dict[int, list[tuple[str, nn.Parameter | torch.Tensor]]] = {}
    for pname, param in model.named_parameters():
        ptr = param.data.data_ptr()
        ptr_to_model_refs.setdefault(ptr, []).append((pname, param))
    for bname, buf in model.named_buffers():
        ptr = buf.data_ptr()
        ptr_to_model_refs.setdefault(ptr, []).append((bname, buf))

    # Phase 4: Move tensors segment-by-segment.
    # For each cudaMalloc segment:
    #   1. Map VMM physical pages for this batch of tensors
    #   2. Copy all tensors from the segment into the VMM range
    #   3. Repoint model parameters to new locations
    #   4. Delete old tensor references
    #   5. empty_cache() to free the now-empty cudaMalloc segment
    cuda_drv = _get_cuda()
    move_start = time.perf_counter()
    new_tensors: dict[str, torch.Tensor] = {}
    moved_bytes = 0
    freed_segments = 0

    try:
        for seg_idx, (_seg_base, seg_tensor_names) in enumerate(segment_order):
            # Find the VA range this segment's tensors occupy
            max_offset = 0
            for name in seg_tensor_names:
                offset, data_bytes = layout[name]
                end = offset + data_bytes
                if end > max_offset:
                    max_offset = end

            # Map VMM physical pages up to where this segment's tensors end
            arena.ensure_mapped(_align_up(max_offset, granularity))

            # Copy and repoint each tensor in this segment
            for name in seg_tensor_names:
                old_t = tensors[name]
                old_ptr = old_t.data_ptr()
                offset, data_bytes = layout[name]
                dest_va = arena.va_base + offset

                ret = cuda_drv.cuMemcpyDtoD_v2(
                    ctypes.c_uint64(dest_va),
                    ctypes.c_uint64(old_ptr),
                    ctypes.c_size_t(data_bytes),
                )
                _check(ret, f"cuMemcpyDtoD for '{name}'")

                new_t = _tensor_from_pointer(
                    data_ptr=dest_va,
                    shape=list(old_t.shape),
                    stride=list(old_t.stride()),
                    dtype=old_t.dtype,
                    device_index=device_id,
                    storage_size_bytes=data_bytes,
                )
                new_tensors[name] = new_t

                # Repoint model parameters AND buffers. tensor.data = x
                # replaces the underlying storage in-place, so the module's
                # reference (whether in _parameters or _buffers) now points
                # at VMM memory without changing the object identity.
                if old_ptr in ptr_to_model_refs:
                    for _qname, ref in ptr_to_model_refs[old_ptr]:
                        ref.data = new_t

                moved_bytes += data_bytes

            # Release all old tensor refs from this segment, then free
            for name in seg_tensor_names:
                del tensors[name]
            torch.cuda.empty_cache()
            freed_segments += 1

    except Exception as e:
        logger.error(
            f"VMM compaction failed after {freed_segments}/{len(segment_order)} "
            f"segments ({moved_bytes / 1e9:.2f} GB moved): {e}"
        )
        logger.error("Releasing VMM arena, falling back to pool registration")
        arena.close()
        # Merge already-moved tensors back. In practice partial failure
        # means the model is broken and the worker will restart, but
        # returning a valid dict lets the caller attempt pool-reg gracefully.
        tensors.update(new_tensors)
        return 0, 0, tensors, None  # type: ignore[return-value]

    torch.cuda.synchronize(device_id)
    move_time = time.perf_counter() - move_start
    total_time = time.perf_counter() - total_start

    logger.info(
        f"[TIMING] VMM compaction: {total_time:.3f}s total "
        f"(segments={seg_time:.3f}s, reserve={reserve_time:.3f}s, "
        f"move={move_time:.3f}s), "
        f"{len(new_tensors)} tensors, {moved_bytes / 1e9:.2f} GB, "
        f"{freed_segments} segments freed -> 1 contiguous VMM region"
    )

    return arena.va_base, arena.total_size, new_tensors, arena
