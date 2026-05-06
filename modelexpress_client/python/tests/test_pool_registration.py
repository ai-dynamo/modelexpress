# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for allocation discovery (cuMemGetAddressRange) and the MX_POOL_REG toggle."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from modelexpress.nixl_transfer import (
    NixlTransferManager,
    _pool_reg_enabled,
)
from modelexpress.types import TensorDescriptor


def _desc(name: str, addr: int, size: int) -> TensorDescriptor:
    return TensorDescriptor(
        name=name,
        addr=addr,
        size=size,
        device_id=0,
        dtype="torch.float16",
    )


class _FakeCuda:
    """Stand-in for ctypes.CDLL('libcuda.so').

    Maintains a list of (alloc_base, alloc_size) regions and a return-code
    override for the next cuMemGetAddressRange_v2 call. The mock identifies
    which allocation an address belongs to by linear scan, mirroring what
    cuMemGetAddressRange_v2 returns from the CUDA driver.
    """

    def __init__(self, allocations: list[tuple[int, int]], ret: int = 0) -> None:
        self._allocations = allocations
        self._ret = ret
        self.calls = 0

    def cuMemGetAddressRange_v2(self, base_ref, size_ref, addr) -> int:
        self.calls += 1
        if self._ret != 0:
            return self._ret
        addr_value = addr.value if hasattr(addr, "value") else int(addr)
        for alloc_base, alloc_size in self._allocations:
            if alloc_base <= addr_value < alloc_base + alloc_size:
                base_ref._obj.value = alloc_base
                size_ref._obj.value = alloc_size
                return 0
        return 1  # CUDA_ERROR_INVALID_VALUE


class TestPoolRegEnabled:
    def test_default_is_off(self, monkeypatch):
        monkeypatch.delenv("MX_POOL_REG", raising=False)
        assert _pool_reg_enabled() is False

    def test_explicit_zero_is_off(self, monkeypatch):
        monkeypatch.setenv("MX_POOL_REG", "0")
        assert _pool_reg_enabled() is False

    def test_one_is_on(self, monkeypatch):
        monkeypatch.setenv("MX_POOL_REG", "1")
        assert _pool_reg_enabled() is True

    def test_arbitrary_truthy_is_off(self, monkeypatch):
        # Strict "1" gate: only "1" enables, anything else (including "true",
        # "yes") leaves pool registration off.
        for value in ("true", "True", "yes", "on", "2", ""):
            monkeypatch.setenv("MX_POOL_REG", value)
            assert _pool_reg_enabled() is False, f"value={value!r} should not enable"

    def test_read_at_call_time(self, monkeypatch):
        # Set after the module has been imported; the function must observe
        # the new value rather than caching a module-level constant.
        monkeypatch.setenv("MX_POOL_REG", "1")
        assert _pool_reg_enabled() is True
        monkeypatch.setenv("MX_POOL_REG", "0")
        assert _pool_reg_enabled() is False


class TestFindCudaAllocations:
    def test_empty_returns_empty(self):
        assert NixlTransferManager._find_cuda_allocations([]) == []

    def test_single_tensor_single_allocation(self):
        # Tensor at 0x1100 inside a 4 KiB allocation starting at 0x1000.
        allocations = [(0x1000, 0x1000)]
        fake = _FakeCuda(allocations)
        with patch("ctypes.CDLL", return_value=fake):
            result = NixlTransferManager._find_cuda_allocations(
                [_desc("w", 0x1100, 64)]
            )
        assert result == [(0x1000, 0x1000)]
        assert fake.calls == 1

    def test_multiple_tensors_same_allocation_dedup(self):
        # Three tensors all inside the same 4 KiB allocation.
        allocations = [(0x1000, 0x1000)]
        fake = _FakeCuda(allocations)
        descriptors = [
            _desc("w0", 0x1000, 64),
            _desc("w1", 0x1100, 64),
            _desc("w2", 0x1200, 64),
        ]
        with patch("ctypes.CDLL", return_value=fake):
            result = NixlTransferManager._find_cuda_allocations(descriptors)
        # All three queries hit, but the result is deduplicated by alloc_base.
        assert result == [(0x1000, 0x1000)]
        assert fake.calls == 3

    def test_multiple_allocations_sorted(self):
        # Three distinct allocations in non-sorted order; result must be
        # sorted by alloc_base.
        allocations = [
            (0x3000, 0x1000),
            (0x1000, 0x1000),
            (0x2000, 0x1000),
        ]
        fake = _FakeCuda(allocations)
        descriptors = [
            _desc("w0", 0x3010, 64),
            _desc("w1", 0x1010, 64),
            _desc("w2", 0x2010, 64),
        ]
        with patch("ctypes.CDLL", return_value=fake):
            result = NixlTransferManager._find_cuda_allocations(descriptors)
        assert result == [
            (0x1000, 0x1000),
            (0x2000, 0x1000),
            (0x3000, 0x1000),
        ]

    def test_adjacent_allocations_not_merged(self):
        # Two allocations that happen to be adjacent in virtual address space
        # must remain separate. Merging them is what the (now-removed)
        # MX_CONTIGUOUS_REG path did, and it broke UCX rcache rkey lookup.
        allocations = [
            (0x1000, 0x1000),  # ends at 0x2000
            (0x2000, 0x1000),  # starts where the previous ends
        ]
        fake = _FakeCuda(allocations)
        descriptors = [
            _desc("w0", 0x1010, 64),
            _desc("w1", 0x2010, 64),
        ]
        with patch("ctypes.CDLL", return_value=fake):
            result = NixlTransferManager._find_cuda_allocations(descriptors)
        assert result == [(0x1000, 0x1000), (0x2000, 0x1000)]

    def test_driver_error_raises_runtime_error(self):
        fake = _FakeCuda(allocations=[], ret=999)  # CUDA_ERROR_UNKNOWN
        with patch("ctypes.CDLL", return_value=fake):
            with pytest.raises(RuntimeError, match="cuMemGetAddressRange_v2 failed"):
                NixlTransferManager._find_cuda_allocations(
                    [_desc("w", 0x1000, 64)]
                )

    def test_driver_error_includes_tensor_name(self):
        fake = _FakeCuda(allocations=[], ret=1)
        with patch("ctypes.CDLL", return_value=fake):
            with pytest.raises(RuntimeError, match="'w_named'"):
                NixlTransferManager._find_cuda_allocations(
                    [_desc("w_named", 0x1000, 64)]
                )
