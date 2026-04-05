#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone test: allocate GPU memory via CUDA VMM (cuMemAddressReserve /
cuMemCreate / cuMemMap), alias it as a torch.Tensor, then register it with
NIXL for RDMA.

Exits 0 on success, 1 on any failure.

Usage:
    python test_vmm_nixl.py [--device-id 0]
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import sys
import time

# ---------------------------------------------------------------------------
# CUDA driver API types and constants
# ---------------------------------------------------------------------------

CUresult = ctypes.c_int
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUdeviceptr = ctypes.c_uint64
CUmemGenericAllocationHandle = ctypes.c_uint64

CU_MEM_ALLOCATION_TYPE_PINNED = 0x1
CU_MEM_LOCATION_TYPE_DEVICE = 0x1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 0x1
CU_MEM_HANDLE_TYPE_NONE = 0x0

# CUmemAllocationProp - must match the driver struct layout
class CUmemLocation(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),   # CUmemLocationType
        ("id", ctypes.c_int),
    ]

class CUmemAllocationCompressAttr(ctypes.Structure):
    _fields_ = [
        ("compressibleMemory", ctypes.c_ubyte),
    ]

class _AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]

class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),              # CUmemAllocationType
        ("requestedHandleTypes", ctypes.c_int),  # CUmemAllocationHandleType
        ("location", CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocFlags),
    ]

class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [
        ("location", CUmemLocation),
        ("flags", ctypes.c_int),   # CUmemAccess_flags
    ]


def _load_cuda_driver():
    """Load libcuda.so and bind the VMM entry points we need."""
    path = ctypes.util.find_library("cuda")
    if path is None:
        # Common fallback paths
        for candidate in ("libcuda.so.1", "libcuda.so"):
            try:
                lib = ctypes.CDLL(candidate)
                return lib
            except OSError:
                continue
        raise RuntimeError("Cannot find libcuda.so - is the NVIDIA driver installed?")
    return ctypes.CDLL(path)


def _check(name: str, result: int) -> None:
    """Raise on non-zero CUDA driver result."""
    if result != 0:
        raise RuntimeError(f"{name} failed with CUresult {result}")


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Test VMM memory + NIXL registration")
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device ordinal")
    args = parser.parse_args()
    device_id = args.device_id

    alloc_size = 64 * 1024 * 1024  # 64 MB
    timings: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Phase 1: CUDA driver init and VMM allocation
    # ------------------------------------------------------------------
    print(f"[phase 1] Allocating {alloc_size // (1024*1024)} MB via CUDA VMM on device {device_id}")
    t0 = time.perf_counter()

    try:
        import torch
        # Initialize CUDA via PyTorch first - this creates the driver context.
        # We then use the driver API on top of PyTorch's context.
        torch.cuda.set_device(device_id)
        torch.cuda.init()
        # Force context creation by touching the device
        torch.empty(1, device=f"cuda:{device_id}")

        cuda = _load_cuda_driver()

        # Build allocation properties with gpuDirectRDMACapable=1
        prop = CUmemAllocationProp()
        ctypes.memset(ctypes.byref(prop), 0, ctypes.sizeof(prop))
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device_id
        prop.allocFlags.gpuDirectRDMACapable = 1

        # Query allocation granularity
        granularity = ctypes.c_size_t()
        _check(
            "cuMemGetAllocationGranularity",
            cuda.cuMemGetAllocationGranularity(
                ctypes.byref(granularity),
                ctypes.byref(prop),
                CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            ),
        )
        gran = granularity.value
        print(f"  allocation granularity: {gran} bytes ({gran // 1024} KB)")

        # Round up to granularity
        padded_size = ((alloc_size + gran - 1) // gran) * gran
        print(f"  padded allocation size: {padded_size} bytes")

        # cuMemAddressReserve - reserve a contiguous VA range
        va_ptr = CUdeviceptr()
        _check(
            "cuMemAddressReserve",
            cuda.cuMemAddressReserve(
                ctypes.byref(va_ptr),
                ctypes.c_size_t(padded_size),
                ctypes.c_size_t(gran),  # alignment
                CUdeviceptr(0),         # addr hint
                ctypes.c_uint64(0),     # flags
            ),
        )
        print(f"  VA reserved at 0x{va_ptr.value:016x}")

        # cuMemCreate - allocate physical pages
        phys_handle = CUmemGenericAllocationHandle()
        _check(
            "cuMemCreate",
            cuda.cuMemCreate(
                ctypes.byref(phys_handle),
                ctypes.c_size_t(padded_size),
                ctypes.byref(prop),
                ctypes.c_uint64(0),
            ),
        )
        print(f"  physical handle: 0x{phys_handle.value:016x}")

        # cuMemMap - map physical pages into the VA range
        _check(
            "cuMemMap",
            cuda.cuMemMap(
                va_ptr,
                ctypes.c_size_t(padded_size),
                ctypes.c_size_t(0),  # offset
                phys_handle,
                ctypes.c_uint64(0),  # flags
            ),
        )
        print("  physical pages mapped into VA")

        # cuMemSetAccess - make the range read/write accessible
        access_desc = CUmemAccessDesc()
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        access_desc.location.id = device_id
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _check(
            "cuMemSetAccess",
            cuda.cuMemSetAccess(
                va_ptr,
                ctypes.c_size_t(padded_size),
                ctypes.byref(access_desc),
                ctypes.c_size_t(1),
            ),
        )
        print("  access flags set (read/write)")

    except Exception as e:
        print(f"[FAIL] Phase 1 - VMM allocation: {e}")
        return 1

    timings["vmm_alloc"] = time.perf_counter() - t0
    print(f"[OK]   Phase 1 complete ({timings['vmm_alloc']:.3f}s)")

    # ------------------------------------------------------------------
    # Phase 2: Create a torch.Tensor aliasing the VMM memory
    # ------------------------------------------------------------------
    print("\n[phase 2] Creating torch.Tensor over VMM memory")
    t0 = time.perf_counter()

    try:
        num_elements = alloc_size // 4  # float32 = 4 bytes
        device = torch.device("cuda", device_id)

        storage = torch._C._construct_storage_from_data_pointer(
            va_ptr.value, device, alloc_size
        )
        tensor = torch.empty(0, dtype=torch.float32, device=device)
        tensor.set_(storage, 0, (num_elements,))

        print(f"  tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
        print(f"  tensor data_ptr: 0x{tensor.data_ptr():016x}")
        assert tensor.data_ptr() == va_ptr.value, (
            f"data_ptr mismatch: tensor=0x{tensor.data_ptr():016x} vs VA=0x{va_ptr.value:016x}"
        )

    except Exception as e:
        print(f"[FAIL] Phase 2 - torch.Tensor creation: {e}")
        return 1

    timings["tensor_create"] = time.perf_counter() - t0
    print(f"[OK]   Phase 2 complete ({timings['tensor_create']:.3f}s)")

    # ------------------------------------------------------------------
    # Phase 3: Fill with a known pattern
    # ------------------------------------------------------------------
    print("\n[phase 3] Filling tensor with known pattern")
    t0 = time.perf_counter()

    try:
        # Fill with a recognizable pattern: ascending integers mod 2^23
        # (fits exactly in float32 mantissa so no precision loss)
        pattern_val = 42.0
        tensor.fill_(pattern_val)
        torch.cuda.synchronize(device_id)

        # Verify
        sample = tensor[:8].cpu()
        expected = torch.full((8,), pattern_val)
        assert torch.equal(sample, expected), f"Pattern mismatch: got {sample}"
        print(f"  fill value: {pattern_val}, verified first 8 elements")

    except Exception as e:
        print(f"[FAIL] Phase 3 - pattern fill: {e}")
        return 1

    timings["pattern_fill"] = time.perf_counter() - t0
    print(f"[OK]   Phase 3 complete ({timings['pattern_fill']:.3f}s)")

    # ------------------------------------------------------------------
    # Phase 4: Register with NIXL
    # ------------------------------------------------------------------
    print("\n[phase 4] Registering VMM tensor with NIXL")
    t0 = time.perf_counter()

    try:
        from nixl._api import nixl_agent as NixlAgent

        agent = NixlAgent("vmm_test_agent")
        print("  NIXL agent created")

        agent.register_memory([tensor], backends=["UCX"])
        print("  tensor registered with NIXL (UCX backend)")

        metadata = agent.get_agent_metadata()
        print(f"  agent metadata: {len(metadata)} bytes")
        assert len(metadata) > 0, "Empty NIXL metadata after registration"

    except ImportError:
        print("[FAIL] Phase 4 - nixl not installed (import failed)")
        return 1
    except Exception as e:
        print(f"[FAIL] Phase 4 - NIXL registration: {e}")
        return 1

    timings["nixl_register"] = time.perf_counter() - t0
    print(f"[OK]   Phase 4 complete ({timings['nixl_register']:.3f}s)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ALL PHASES PASSED")
    print("=" * 60)
    print(f"  VMM allocation:      {timings['vmm_alloc']:.3f}s")
    print(f"  Tensor creation:     {timings['tensor_create']:.3f}s")
    print(f"  Pattern fill:        {timings['pattern_fill']:.3f}s")
    print(f"  NIXL registration:   {timings['nixl_register']:.3f}s")
    print(f"  Total:               {sum(timings.values()):.3f}s")
    print(f"  Allocation size:     {alloc_size // (1024*1024)} MB")
    print(f"  VA base:             0x{va_ptr.value:016x}")
    print(f"  GPUDirect RDMA:      enabled (allocFlags.gpuDirectRDMACapable=1)")

    # Cleanup note: we intentionally do NOT unmap/free here.
    # The process exit will reclaim everything, and explicit teardown
    # risks use-after-free if NIXL still holds references.

    return 0


if __name__ == "__main__":
    sys.exit(main())
