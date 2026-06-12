# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for allocation discovery (cuMemGetAddressRange), the MX_POOL_REG toggle,
and receive_from_source manifest validation."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest
import torch

from modelexpress.nixl_transfer import (
    NIXL_ACCELERATOR_MEM_TYPE,
    NixlTransferManager,
    _pool_reg_enabled,
)
from modelexpress.types import ManifestMismatchError, TensorDescriptor


def _desc(name: str, addr: int, size: int) -> TensorDescriptor:
    return TensorDescriptor(
        name=name,
        addr=addr,
        size=size,
        device_id=0,
        dtype="torch.float16",
    )


class _FakeDriver:
    """Stand-in for cuda.bindings.driver.cuMemGetAddressRange.

    Maintains a list of (alloc_base, alloc_size) regions. On each call,
    locates the allocation containing the queried address (mirroring what
    the real CUDA driver does) and returns the (err, base, size) triple
    matching the cuda-python binding's signature.
    """

    def __init__(
        self,
        allocations: list[tuple[int, int]],
        err_override=None,
    ) -> None:
        self._allocations = allocations
        self._err_override = err_override
        self.calls = 0

    def cuMemGetAddressRange(self, addr: int):
        from cuda.bindings import driver

        self.calls += 1
        if self._err_override is not None:
            return (self._err_override, 0, 0)
        for alloc_base, alloc_size in self._allocations:
            if alloc_base <= addr < alloc_base + alloc_size:
                return (driver.CUresult.CUDA_SUCCESS, alloc_base, alloc_size)
        return (driver.CUresult.CUDA_ERROR_INVALID_VALUE, 0, 0)


@pytest.fixture
def fake_driver(monkeypatch):
    """Replace cuda.bindings.driver.cuMemGetAddressRange with a fake.

    The fake is returned so tests can inspect call counts. The real
    `CUresult` enum is preserved so `err.name` formatting in the function
    under test exercises the same code path as production.
    """
    from cuda.bindings import driver

    def _make(allocations, err_override=None):
        fake = _FakeDriver(allocations, err_override)
        monkeypatch.setattr(
            driver,
            "cuMemGetAddressRange",
            fake.cuMemGetAddressRange,
        )
        return fake

    return _make


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

    def test_single_tensor_single_allocation(self, fake_driver):
        # Tensor at 0x1100 inside a 4 KiB allocation starting at 0x1000.
        fake = fake_driver([(0x1000, 0x1000)])
        result = NixlTransferManager._find_cuda_allocations(
            [_desc("w", 0x1100, 64)]
        )
        assert result == [(0x1000, 0x1000)]
        assert fake.calls == 1

    def test_multiple_tensors_same_allocation_dedup(self, fake_driver):
        # Three tensors all inside the same 4 KiB allocation.
        fake = fake_driver([(0x1000, 0x1000)])
        descriptors = [
            _desc("w0", 0x1000, 64),
            _desc("w1", 0x1100, 64),
            _desc("w2", 0x1200, 64),
        ]
        result = NixlTransferManager._find_cuda_allocations(descriptors)
        # All three queries hit, but the result is deduplicated by alloc_base.
        assert result == [(0x1000, 0x1000)]
        assert fake.calls == 3

    def test_multiple_allocations_sorted(self, fake_driver):
        # Three distinct allocations in non-sorted order; result must be
        # sorted by alloc_base.
        fake_driver([
            (0x3000, 0x1000),
            (0x1000, 0x1000),
            (0x2000, 0x1000),
        ])
        descriptors = [
            _desc("w0", 0x3010, 64),
            _desc("w1", 0x1010, 64),
            _desc("w2", 0x2010, 64),
        ]
        result = NixlTransferManager._find_cuda_allocations(descriptors)
        assert result == [
            (0x1000, 0x1000),
            (0x2000, 0x1000),
            (0x3000, 0x1000),
        ]

    def test_adjacent_allocations_not_merged(self, fake_driver):
        # Two allocations that happen to be adjacent in virtual address space
        # must remain separate. Merging them is what the (now-removed)
        # MX_CONTIGUOUS_REG path did, and it broke UCX rcache rkey lookup.
        fake_driver([
            (0x1000, 0x1000),  # ends at 0x2000
            (0x2000, 0x1000),  # starts where the previous ends
        ])
        descriptors = [
            _desc("w0", 0x1010, 64),
            _desc("w1", 0x2010, 64),
        ]
        result = NixlTransferManager._find_cuda_allocations(descriptors)
        assert result == [(0x1000, 0x1000), (0x2000, 0x1000)]

    def test_driver_error_raises_runtime_error(self, fake_driver):
        from cuda.bindings import driver

        fake_driver(allocations=[], err_override=driver.CUresult.CUDA_ERROR_UNKNOWN)
        with pytest.raises(RuntimeError, match="cuMemGetAddressRange failed"):
            NixlTransferManager._find_cuda_allocations(
                [_desc("w", 0x1000, 64)]
            )

    def test_driver_error_includes_tensor_name(self, fake_driver):
        from cuda.bindings import driver

        fake_driver(allocations=[], err_override=driver.CUresult.CUDA_ERROR_INVALID_VALUE)
        with pytest.raises(RuntimeError, match="'w_named'"):
            NixlTransferManager._find_cuda_allocations(
                [_desc("w_named", 0x1000, 64)]
            )


class TestRawDescriptorMemType:
    def _make_manager(self) -> NixlTransferManager:
        mgr = NixlTransferManager(agent_name="test", device_id=0)
        mgr._agent = MagicMock()
        mgr._agent.get_agent_metadata.return_value = b"metadata"
        return mgr

    def test_pool_registration_uses_vram_segment(self, monkeypatch, fake_driver):
        monkeypatch.setenv("MX_POOL_REG", "1")
        tensor = torch.zeros(4, dtype=torch.float32)
        fake_driver([(tensor.data_ptr(), tensor.numel() * tensor.element_size())])

        mgr = self._make_manager()
        assert mgr.register_tensors({"w": tensor}) == b"metadata"

        mgr._agent.register_memory.assert_called_once_with(
            [(tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0, "")],
            mem_type=NIXL_ACCELERATOR_MEM_TYPE,
            backends=["UCX"],
        )

    def test_arena_registration_uses_vram_segment(self):
        class FakeArena:
            def registered_range(self):
                return 0x1000, 0x2000

        mgr = self._make_manager()
        assert mgr.register_arena(FakeArena(), {"w": torch.zeros(1)}) == b"metadata"

        mgr._agent.register_memory.assert_called_once_with(
            [(0x1000, 0x2000, 0, "")],
            mem_type=NIXL_ACCELERATOR_MEM_TYPE,
            backends=["UCX"],
        )

    def test_receive_transfer_descriptors_use_vram_segment(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "set_device", lambda *args, **kwargs: None)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)

        local = torch.zeros(4, dtype=torch.float32)
        mgr = self._make_manager()
        mgr._tensors = {"w": local}
        mgr._agent.prep_xfer_dlist.side_effect = ["src", "dst"]
        mgr._agent.make_prepped_xfer.return_value = "handle"
        mgr._agent.check_xfer_state.return_value = "DONE"

        result = mgr.receive_from_source(
            source_metadata=b"",
            source_tensors=[
                TensorDescriptor(
                    name="w",
                    addr=0x1000,
                    size=local.numel() * local.element_size(),
                    device_id=0,
                    dtype=str(local.dtype),
                )
            ],
            remote_agent_name="source",
        )

        assert result[0] == local.numel() * local.element_size()
        assert result[1] == 1
        assert mgr._agent.prep_xfer_dlist.call_args_list == [
            call(
                agent_name="source",
                xfer_list=[(0x1000, local.numel() * local.element_size(), 0)],
                mem_type=NIXL_ACCELERATOR_MEM_TYPE,
                backends=["UCX"],
            ),
            call(
                agent_name="",
                xfer_list=[
                    (local.data_ptr(), local.numel() * local.element_size(), 0),
                ],
                mem_type=NIXL_ACCELERATOR_MEM_TYPE,
                backends=["UCX"],
            ),
        ]


class TestReceiveFromSourceManifestValidation:
    """receive_from_source must reject size/dtype mismatches before building
    RDMA descriptors. Catching these here prevents silent memory corruption
    when stale source metadata or model skew sneaks past the name match.
    """

    def _make_manager(self, monkeypatch, local_tensors):
        # Bypass torch.cuda.set_device since the test runs on a CPU host.
        monkeypatch.setattr(torch.cuda, "set_device", lambda *args, **kwargs: None)
        mgr = NixlTransferManager(agent_name="test", device_id=0)
        mgr._agent = MagicMock()  # non-None so the early null check passes
        mgr._tensors = local_tensors
        return mgr

    def test_size_mismatch_raises_manifest_mismatch(self, monkeypatch):
        # Local tensor: 40 bytes (10 float32). Source claims 80 bytes.
        local = torch.zeros(10, dtype=torch.float32)
        mgr = self._make_manager(monkeypatch, {"w": local})
        bogus = TensorDescriptor(
            name="w", addr=0x1000, size=80, device_id=0, dtype=str(local.dtype),
        )
        with pytest.raises(ManifestMismatchError, match="size mismatch"):
            mgr.receive_from_source(
                source_metadata=b"",
                source_tensors=[bogus],
                remote_agent_name="dummy",
            )

    def test_dtype_mismatch_raises_manifest_mismatch(self, monkeypatch):
        # Local tensor float32 (40 bytes). Source size matches but dtype lies.
        local = torch.zeros(10, dtype=torch.float32)
        mgr = self._make_manager(monkeypatch, {"w": local})
        bogus = TensorDescriptor(
            name="w", addr=0x1000, size=40, device_id=0, dtype="torch.bfloat16",
        )
        with pytest.raises(ManifestMismatchError, match="dtype mismatch"):
            mgr.receive_from_source(
                source_metadata=b"",
                source_tensors=[bogus],
                remote_agent_name="dummy",
            )

    def test_unmatched_name_skips_silently(self, monkeypatch):
        # No matching local tensor for the source's "w". Loop should `continue`
        # without raising; the caller decides whether the empty match list is
        # an error. We just verify the validation doesn't fire spuriously.
        mgr = self._make_manager(monkeypatch, {"x": torch.zeros(1, dtype=torch.float32)})
        wrong_name = TensorDescriptor(
            name="w", addr=0x1000, size=4, device_id=0, dtype="torch.float32",
        )
        # Empty match -> early-return (0, 0, 0.0); no exception.
        result = mgr.receive_from_source(
            source_metadata=b"",
            source_tensors=[wrong_name],
            remote_agent_name="dummy",
        )
        assert result == (0, 0, 0.0)
