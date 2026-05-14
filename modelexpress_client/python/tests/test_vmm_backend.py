# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CudaVmmBackend.

These tests require a real CUDA-capable GPU. They skip automatically on
hosts without CUDA so the test suite stays green on dev machines.

Validation strategy: where possible we cross-check against torch.cuda
memory accounting (memory_allocated, memory_reserved). nvidia-smi-level
validation is left for manual cluster runs.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Skip-on-no-CUDA gate
# ---------------------------------------------------------------------------


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        from cuda.bindings import driver  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _cuda_available(),
    reason="requires CUDA + cuda-python",
)


@pytest.fixture(scope="module")
def cuda_init():
    """Ensure a CUDA context is bound on the current thread.

    torch.cuda.init() + set_device alone does NOT bind a primary context
    on the calling thread (cuCtxGetCurrent returns 0). A real CUDA op is
    required to force context binding. We allocate a 1-element tensor
    and free it - cheap and reliable.
    """
    import torch

    torch.cuda.init()
    torch.cuda.set_device(0)
    _ = torch.zeros(1, device="cuda:0")
    torch.cuda.synchronize()
    yield 0


# ---------------------------------------------------------------------------
# Backend construction + granularity
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_can_construct(self, cuda_init):
        from modelexpress.vmm.backend import CudaVmmBackend

        backend = CudaVmmBackend(device=cuda_init)
        assert backend.device == cuda_init
        assert backend.allocation_granularity() > 0

    def test_granularity_is_power_of_2(self, cuda_init):
        from modelexpress.vmm.backend import CudaVmmBackend

        backend = CudaVmmBackend(device=cuda_init)
        gran = backend.allocation_granularity()
        assert gran & (gran - 1) == 0  # power of 2
        # Typical values are 2 MiB on Hopper / Blackwell.
        assert gran <= 64 * 1024 * 1024

    def test_no_context_raises(self):
        """Without a CUDA context bound the constructor must raise.

        Hard to test cleanly because once a context exists in the process
        we can't destroy it without affecting other tests. Skip this
        unless we figure out an isolated subprocess approach.
        """
        pytest.skip("would need a process-isolated test harness")


# ---------------------------------------------------------------------------
# Reserve + allocate + deallocate
# ---------------------------------------------------------------------------


class TestAllocationLifecycle:
    def test_reserve_returns_nonzero_va(self, cuda_init):
        from modelexpress.vmm.backend import CudaVmmBackend

        backend = CudaVmmBackend(device=cuda_init)
        size = backend.allocation_granularity() * 4
        base = backend.reserve(size)
        try:
            assert base != 0
            # VA should be aligned to granularity.
            assert base % backend.allocation_granularity() == 0
        finally:
            backend.release_reserve(base, size)

    def test_reserve_rounds_up(self, cuda_init):
        from modelexpress.vmm.backend import CudaVmmBackend

        backend = CudaVmmBackend(device=cuda_init)
        gran = backend.allocation_granularity()
        # Request a byte more than one granule; backend must round up
        # internally so reserve + release succeeds with the rounded size.
        base = backend.reserve(gran + 1)
        try:
            assert base % gran == 0
        finally:
            backend.release_reserve(base, 2 * gran)

    def test_allocate_basic(self, cuda_init):
        import torch

        from modelexpress.vmm.backend import CudaVmmBackend

        backend = CudaVmmBackend(device=cuda_init)
        gran = backend.allocation_granularity()
        size = gran * 2
        base = backend.reserve(size)
        handle = None
        try:
            # Track GPU memory before mapping. cuMem allocations don't
            # show up in torch.cuda.memory_allocated (PyTorch's caching
            # allocator is independent), so we use mem_get_info instead.
            free_before, _ = torch.cuda.mem_get_info(cuda_init)
            handle = backend.allocate(base, gran)
            free_after, _ = torch.cuda.mem_get_info(cuda_init)
            assert handle != 0
            assert free_after < free_before
            assert free_before - free_after >= gran // 2  # generous lower bound
        finally:
            if handle is not None:
                backend.deallocate(base, gran, handle)
            backend.release_reserve(base, size)

    def test_deallocate_releases_memory(self, cuda_init):
        import torch

        from modelexpress.vmm.backend import CudaVmmBackend

        backend = CudaVmmBackend(device=cuda_init)
        gran = backend.allocation_granularity()
        size = gran * 2
        base = backend.reserve(size)
        handle = backend.allocate(base, gran)

        try:
            free_with_allocation, _ = torch.cuda.mem_get_info(cuda_init)
            backend.deallocate(base, gran, handle)
            handle = None
            free_after_deallocate, _ = torch.cuda.mem_get_info(cuda_init)
            assert free_after_deallocate > free_with_allocation
            assert free_after_deallocate - free_with_allocation >= gran // 2
        finally:
            if handle is not None:
                backend.deallocate(base, gran, handle)
            backend.release_reserve(base, size)


# ---------------------------------------------------------------------------
# Arena + real backend integration
# ---------------------------------------------------------------------------


class TestArenaWithRealBackend:
    def test_single_allocation_returns_writable_memory(self, cuda_init):
        """Confirm tensor writes/reads through the arena address."""
        import torch

        from modelexpress.vmm.arena import VmmArena
        from modelexpress.vmm.backend import CudaVmmBackend

        backend = CudaVmmBackend(device=cuda_init)
        gran = backend.allocation_granularity()
        with VmmArena(total_bytes=gran * 4, backend=backend) as arena:
            ptr = arena.malloc(1024 * 1024, alignment=256)
            # Construct a torch tensor backed by the VMM address. The
            # empty tensor must be created on the same device as the
            # storage (modern PyTorch rejects cross-device set_).
            device = torch.device(f"cuda:{cuda_init}")
            storage = torch._C._construct_storage_from_data_pointer(
                ptr, device, 1024 * 1024
            )
            tensor = torch.empty(0, dtype=torch.uint8, device=device).set_(
                storage, 0, (1024 * 1024,)
            )
            tensor.fill_(0xAB)
            torch.cuda.synchronize()
            # Read back: every byte should be 0xAB.
            assert tensor.cpu().tolist().count(0xAB) == 1024 * 1024

    def test_per_allocation_mapping(self, cuda_init):
        """Each malloc maps one physical handle and increases mapped bytes."""
        import torch

        from modelexpress.vmm.arena import VmmArena
        from modelexpress.vmm.backend import CudaVmmBackend

        backend = CudaVmmBackend(device=cuda_init)
        gran = backend.allocation_granularity()
        free_start, _ = torch.cuda.mem_get_info(cuda_init)
        with VmmArena(total_bytes=gran * 8, backend=backend) as arena:
            # Empty arena - no physical mapping yet.
            free_empty_arena, _ = torch.cuda.mem_get_info(cuda_init)
            # Reservation is VA only, no physical commit.
            assert abs(free_empty_arena - free_start) < gran  # roughly equal

            arena.malloc(1024)
            assert arena.live_allocation_count == 1
            assert arena.mapped_bytes == gran
            free_one_allocation, _ = torch.cuda.mem_get_info(cuda_init)
            assert free_start - free_one_allocation >= gran // 2

            arena.malloc(gran * 2)
            assert arena.live_allocation_count == 2
            assert arena.mapped_bytes == gran * 3
            free_two_allocations, _ = torch.cuda.mem_get_info(cuda_init)
            assert free_one_allocation - free_two_allocations >= gran

    def test_free_releases_one_allocation(self, cuda_init):
        import torch

        from modelexpress.vmm.arena import VmmArena
        from modelexpress.vmm.backend import CudaVmmBackend

        backend = CudaVmmBackend(device=cuda_init)
        gran = backend.allocation_granularity()
        with VmmArena(total_bytes=gran * 8, backend=backend) as arena:
            ptr1 = arena.malloc(1024)
            arena.malloc(gran * 2)
            used_before = arena.used_bytes
            mapped_before = arena.mapped_bytes
            free_before_free, _ = torch.cuda.mem_get_info(cuda_init)

            arena.free(ptr1)

            free_after_free, _ = torch.cuda.mem_get_info(cuda_init)
            assert arena.used_bytes == used_before
            assert arena.mapped_bytes == mapped_before - gran
            assert arena.live_allocation_count == 1
            assert free_after_free - free_before_free >= gran // 2

    def test_close_releases_all_allocations(self, cuda_init):
        import torch

        from modelexpress.vmm.arena import VmmArena
        from modelexpress.vmm.backend import CudaVmmBackend

        backend = CudaVmmBackend(device=cuda_init)
        gran = backend.allocation_granularity()

        free_start, _ = torch.cuda.mem_get_info(cuda_init)
        arena = VmmArena(total_bytes=gran * 8, backend=backend)
        for _ in range(3):
            arena.malloc(gran * 2)
        free_after_alloc, _ = torch.cuda.mem_get_info(cuda_init)
        assert free_start - free_after_alloc >= 3 * gran
        assert arena.mapped_bytes == 6 * gran

        arena.close()
        free_after_close, _ = torch.cuda.mem_get_info(cuda_init)
        # After deallocate + release, free memory should be back near baseline.
        assert free_after_close - free_after_alloc >= 3 * gran
        assert abs(free_after_close - free_start) < gran  # roughly back to start
