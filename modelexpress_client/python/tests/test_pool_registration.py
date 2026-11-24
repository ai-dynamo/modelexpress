# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for allocation discovery (cuMemGetAddressRange), the MX_POOL_REG toggle,
and receive_from_source manifest validation."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, call

import pytest
import torch

from modelexpress import nixl_transfer, p2p_pb2
from modelexpress.accelerators import CudaAcceleratorBackend, NIXL_ACCELERATOR_MEM_TYPE
from modelexpress.nixl_transfer import (
    _arena_single_mr_forced,
    NixlTransferManager,
    _pool_reg_enabled,
    _spanning_descriptors,
)
from modelexpress.types import ManifestMismatchError, TensorDescriptor


class _StandInCudaBackend(CudaAcceleratorBackend):
    """CUDA backend for CPU tensors standing in for GPU tensors."""

    def is_accel_tensor(self, tensor: torch.Tensor) -> bool:
        return True


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

    Skips on hosts without the ``cuda`` bindings (e.g. an XPU-only node), where
    these allocation-discovery tests cannot run.
    """
    driver = pytest.importorskip("cuda.bindings.driver")

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
    def test_default_is_on(self, monkeypatch):
        monkeypatch.delenv("MX_POOL_REG", raising=False)
        monkeypatch.delenv("MX_VMM_ARENA", raising=False)
        assert _pool_reg_enabled() is True

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
        result = NixlTransferManager._find_cuda_allocations([_desc("w", 0x1100, 64)])
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
        fake_driver(
            [
                (0x3000, 0x1000),
                (0x1000, 0x1000),
                (0x2000, 0x1000),
            ]
        )
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
        fake_driver(
            [
                (0x1000, 0x1000),  # ends at 0x2000
                (0x2000, 0x1000),  # starts where the previous ends
            ]
        )
        descriptors = [
            _desc("w0", 0x1010, 64),
            _desc("w1", 0x2010, 64),
        ]
        result = NixlTransferManager._find_cuda_allocations(descriptors)
        assert result == [(0x1000, 0x1000), (0x2000, 0x1000)]

    def test_tensor_past_allocation_end_raises(self, fake_driver):
        # One VA range backed by two mappings reads as two allocations.
        fake_driver([(0x1000, 0x1000), (0x2000, 0x1000)])
        with pytest.raises(RuntimeError, match="'w' at 0x1f00 .* extends past"):
            NixlTransferManager._find_cuda_allocations([_desc("w", 0x1F00, 0x200)])

    def test_driver_error_raises_runtime_error(self, fake_driver):
        from cuda.bindings import driver

        fake_driver(allocations=[], err_override=driver.CUresult.CUDA_ERROR_UNKNOWN)
        with pytest.raises(RuntimeError, match="cuMemGetAddressRange failed"):
            NixlTransferManager._find_cuda_allocations([_desc("w", 0x1000, 64)])

    def test_driver_error_includes_tensor_name(self, fake_driver):
        from cuda.bindings import driver

        fake_driver(
            allocations=[], err_override=driver.CUresult.CUDA_ERROR_INVALID_VALUE
        )
        with pytest.raises(RuntimeError, match="'w_named'"):
            NixlTransferManager._find_cuda_allocations([_desc("w_named", 0x1000, 64)])


class TestSpanningDescriptors:
    def test_tensors_inside_allocations_are_not_reported(self):
        allocations = [(0x1000, 0x1000), (0x2000, 0x1000)]
        descriptors = [_desc("w0", 0x1000, 0x1000), _desc("w1", 0x2800, 0x800)]
        assert _spanning_descriptors(descriptors, allocations) == []

    def test_tensor_past_allocation_end_is_reported_with_its_allocation(self):
        allocations = [(0x1000, 0x1000), (0x2000, 0x1000)]
        inside = _desc("inside", 0x1000, 0x100)
        spanning = _desc("spanning", 0x1f00, 0x200)
        assert _spanning_descriptors([inside, spanning], allocations) == [
            (spanning, 0x1000, 0x1000)
        ]


class TestRawDescriptorMemType:
    def _make_manager(self) -> NixlTransferManager:
        mgr = NixlTransferManager(
            agent_name="test", device_id=0, accelerator_backend=_StandInCudaBackend()
        )
        mgr._agent = MagicMock()
        mgr._agent.get_agent_metadata.return_value = b"metadata"
        return mgr

    def test_pool_registration_uses_vram_segment(self, monkeypatch, fake_driver):
        monkeypatch.setenv("MX_POOL_REG", "1")
        tensor = MagicMock(wraps=torch.zeros(4, dtype=torch.float32))
        tensor.is_cuda = True
        fake_driver([(tensor.data_ptr(), tensor.numel() * tensor.element_size())])

        mgr = self._make_manager()
        assert mgr.register_tensors({"w": tensor}) == b"metadata"

        mgr._agent.register_memory.assert_called_once_with(
            [(tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0, "")],
            mem_type=NIXL_ACCELERATOR_MEM_TYPE,
            backends=["UCX"],
        )

    def test_pool_registration_keeps_allocations_when_tensors_fit(self, monkeypatch):
        monkeypatch.setenv("MX_POOL_REG", "1")
        backing = torch.zeros(8, dtype=torch.float32)
        first = MagicMock(wraps=backing[:4])
        first.is_cuda = True
        second = MagicMock(wraps=backing[4:])
        second.is_cuda = True
        base = backing.data_ptr()
        mgr = self._make_manager()
        # Each tensor ends exactly at the end of its own allocation.
        mgr._find_cuda_allocations = MagicMock(
            return_value=[(base, 16), (base + 16, 16)]
        )
        mgr.register_tensors({"a": first, "b": second})

        mgr._agent.register_memory.assert_called_once_with(
            [(base, 16, 0, ""), (base + 16, 16, 0, "")],
            mem_type=NIXL_ACCELERATOR_MEM_TYPE,
            backends=["UCX"],
        )

    def test_pool_registration_rejects_tensor_spanning_allocations(self, monkeypatch):
        monkeypatch.setenv("MX_POOL_REG", "1")
        backing = torch.zeros(4, dtype=torch.float32)
        tensor = MagicMock(wraps=backing)
        tensor.is_cuda = True
        base = backing.data_ptr()
        mgr = self._make_manager()
        # One VA range backed by two separately mapped segments.
        mgr._find_cuda_allocations = MagicMock(return_value=[(base, 8), (base + 8, 8)])

        with pytest.raises(RuntimeError) as exc_info:
            mgr.register_tensors({"workspace": tensor})

        message = str(exc_info.value)
        assert "First: workspace" in message
        assert f"[0x{base:x}, 0x{base + 8:x})" in message
        mgr._agent.register_memory.assert_not_called()

    def test_draft_view_inside_registered_pool_reuses_main_registration(
        self, monkeypatch
    ):
        monkeypatch.setenv("MX_POOL_REG", "1")
        backing = torch.zeros(8, dtype=torch.float32)
        main = MagicMock(wraps=backing[:2])
        main.is_cuda = True
        draft = backing[4:6]
        mgr = self._make_manager()
        mgr._find_cuda_allocations = MagicMock(
            return_value=[(backing.data_ptr(), backing.numel() * backing.element_size())]
        )

        mgr.register_tensors({"main": main})
        assert mgr.register_additional_tensors({"draft": draft}) == b"metadata"

        assert [item.name for item in mgr.tensor_descriptors] == ["main", "draft"]
        mgr._agent.register_memory.assert_called_once()

    def test_host_sources_bypass_cuda_pool_discovery(self, monkeypatch):
        monkeypatch.setenv("MX_POOL_REG", "1")
        tensor = torch.zeros(4, dtype=torch.float32)
        mgr = NixlTransferManager(agent_name="test", device_id=3)
        mgr._agent = MagicMock()
        mgr._agent.get_agent_metadata.return_value = b"metadata"
        mgr._find_cuda_allocations = MagicMock(
            side_effect=AssertionError("CPU address")
        )
        assert mgr.register_tensors({"w": tensor}) == b"metadata"
        mgr._find_cuda_allocations.assert_not_called()
        mgr._agent.register_memory.assert_called_once_with(
            [(tensor.data_ptr(), 16, 0, "")], mem_type="DRAM", backends=["UCX"]
        )
        assert mgr.tensor_descriptors[0].device_id == 0

    def test_null_address_tensor_stays_in_manifest_but_is_not_registered(
        self, monkeypatch
    ):
        monkeypatch.delenv("MX_POOL_REG", raising=False)
        tensor = torch.zeros(4, dtype=torch.float32)
        unallocated = MagicMock()
        unallocated.numel.return_value = 1
        unallocated.element_size.return_value = 4
        unallocated.data_ptr.return_value = 0
        unallocated.dtype = torch.float32
        unallocated.is_contiguous.side_effect = AssertionError(
            "contiguity should not be checked without allocated storage"
        )

        mgr = self._make_manager()
        assert (
            mgr.register_tensors({"w": tensor, "unallocated": unallocated})
            == b"metadata"
        )

        mgr._agent.register_memory.assert_called_once_with(
            [tensor], backends=["UCX"]
        )
        assert list(mgr._tensors) == ["w", "unallocated"]
        assert mgr._tensors["unallocated"] is unallocated

        descriptors = {
            descriptor.name: descriptor for descriptor in mgr.tensor_descriptors
        }
        assert descriptors["unallocated"].addr == 0
        assert descriptors["unallocated"].size == 4

    def test_all_empty_tensors_skip_pool_registration(self, monkeypatch):
        monkeypatch.setenv("MX_POOL_REG", "1")
        empty = torch.empty(0, dtype=torch.float32)
        mgr = self._make_manager()
        mgr._find_cuda_allocations = MagicMock(
            side_effect=AssertionError(
                "pool discovery should not run without registrable tensors"
            )
        )

        assert mgr.register_tensors({"empty": empty}) == b"metadata"

        mgr._find_cuda_allocations.assert_not_called()
        mgr._agent.register_memory.assert_not_called()
        assert [descriptor.name for descriptor in mgr.tensor_descriptors] == [
            "empty"
        ]

    def test_arena_registration_uses_vram_segment(self):
        # The arena range must actually cover the tensor, as a real arena does.
        tensor = torch.zeros(1)
        base = tensor.data_ptr()
        used = tensor.numel() * tensor.element_size()

        class FakeArena:
            live_allocation_count = 1

            def registered_range(self):
                return base, used

        mgr = self._make_manager()
        assert mgr.register_arena(FakeArena(), {"w": tensor}) == b"metadata"

        mgr._agent.register_memory.assert_called_once_with(
            [(base, used, 0, "")],
            mem_type=NIXL_ACCELERATOR_MEM_TYPE,
            backends=["UCX"],
        )

    def test_arena_ignores_empty_tensors_in_containment_check(self):
        tensor = torch.zeros(1)
        base = tensor.data_ptr()
        used = tensor.numel() * tensor.element_size()

        class FakeArena:
            live_allocation_count = 1

            def registered_range(self):
                return base, used

        empty = torch.empty(0, dtype=torch.float32)
        unallocated = torch.empty(1, dtype=torch.float32, device="meta")
        mgr = self._make_manager()
        assert (
            mgr.register_arena(
                FakeArena(), {"w": tensor, "empty": empty, "meta": unallocated}
            )
            == b"metadata"
        )

        mgr._agent.register_memory.assert_called_once_with(
            [(base, used, 0, "")],
            mem_type=NIXL_ACCELERATOR_MEM_TYPE,
            backends=["UCX"],
        )

    def test_draft_view_inside_registered_arena_reuses_main_registration(self):
        main = torch.zeros(8, dtype=torch.float32)
        draft = main[2:4]

        class FakeArena:
            live_allocation_count = 1

            def registered_range(self):
                return main.data_ptr(), main.numel() * main.element_size()

        mgr = self._make_manager()
        mgr.register_arena(FakeArena(), {"main": main})

        assert mgr.register_additional_tensors({"draft": draft}) == b"metadata"
        assert [item.name for item in mgr.tensor_descriptors] == ["main", "draft"]
        mgr._agent.register_memory.assert_called_once()

        mgr.deregister_tensors(["draft"])
        assert [item.name for item in mgr.tensor_descriptors] == ["main"]
        mgr._agent.deregister_memory.assert_not_called()

    def test_draft_tensor_outside_main_registration_gets_its_own_range(self):
        main = torch.zeros(8, dtype=torch.float32)
        draft = torch.ones(8, dtype=torch.float32)
        mgr = self._make_manager()
        mgr.register_tensors({"main": main})

        assert mgr.register_additional_tensors({"draft": draft}) == b"metadata"
        assert [item.name for item in mgr.tensor_descriptors] == ["main", "draft"]
        assert mgr._agent.register_memory.call_count == 2

    def test_appended_subview_reuses_registration_until_last_name_is_removed(self):
        main = torch.zeros(8, dtype=torch.float32)
        draft = torch.ones(8, dtype=torch.float32)
        draft_alias = draft[2:4]
        base_handle, draft_handle, duplicate_handle = object(), object(), object()
        mgr = self._make_manager()
        mgr._agent.register_memory.side_effect = [
            base_handle,
            draft_handle,
            duplicate_handle,
        ]

        mgr.register_tensors({"main": main})
        mgr.register_additional_tensors({"draft": draft})
        mgr.register_additional_tensors({"draft_alias": draft_alias})

        assert [item.name for item in mgr.tensor_descriptors] == [
            "main",
            "draft",
            "draft_alias",
        ]
        assert mgr._agent.register_memory.call_count == 2

        mgr.deregister_tensors(["draft"])
        assert [item.name for item in mgr.tensor_descriptors] == [
            "main",
            "draft_alias",
        ]
        mgr._agent.deregister_memory.assert_not_called()

        mgr.deregister_tensors(["draft_alias"])
        mgr._agent.deregister_memory.assert_called_once_with(draft_handle)

    def test_appended_batch_registers_parent_once_even_when_view_is_first(self):
        main = torch.zeros(8, dtype=torch.float32)
        draft = torch.ones(8, dtype=torch.float32)
        draft_alias = draft[2:4]
        mgr = self._make_manager()
        mgr._agent.register_memory.side_effect = [object(), object(), object()]

        mgr.register_tensors({"main": main})
        mgr.register_additional_tensors(
            {"draft_alias": draft_alias, "draft": draft}
        )

        assert [item.name for item in mgr.tensor_descriptors] == [
            "main",
            "draft_alias",
            "draft",
        ]
        assert mgr._agent.register_memory.call_count == 2
        registered = mgr._agent.register_memory.call_args.args[0]
        assert len(registered) == 1
        assert registered[0] is draft

    def test_appended_sibling_reuses_range_after_parent_name_is_removed(self):
        main = torch.zeros(8, dtype=torch.float32)
        draft = torch.ones(8, dtype=torch.float32)
        first_view = draft[1:3]
        second_view = draft[5:7]
        mgr = self._make_manager()
        mgr._agent.register_memory.side_effect = [object(), object(), object()]

        mgr.register_tensors({"main": main})
        mgr.register_additional_tensors({"draft": draft, "first_view": first_view})
        mgr.deregister_tensors(["draft"])
        mgr.register_additional_tensors({"second_view": second_view})

        assert [item.name for item in mgr.tensor_descriptors] == [
            "main",
            "first_view",
            "second_view",
        ]
        assert mgr._agent.register_memory.call_count == 2

    def test_arena_registration_falls_back_when_tensor_uncovered(self):
        # A tensor outside [base, base+used) must not be served by the single MR.
        class FakeArena:
            live_allocation_count = 1

            def registered_range(self):
                return 0x1000, 0x2000

        tensor = torch.zeros(4, dtype=torch.float32)
        mgr = self._make_manager()
        assert mgr.register_arena(FakeArena(), {"w": tensor}) == b"metadata"

        args, kwargs = mgr._agent.register_memory.call_args
        assert args[0][0] is tensor
        assert kwargs == {"backends": ["UCX"]}

    def test_arena_fallback_bypasses_pool_registration(self, monkeypatch):
        # Pool reg resolves the same per-handle bounds, so the fallback must
        # register per tensor even when MX_POOL_REG=1.
        monkeypatch.setenv("MX_POOL_REG", "1")

        class FakeArena:
            live_allocation_count = 1

            def registered_range(self):
                return 0x1000, 0x2000

        tensor = torch.zeros(4, dtype=torch.float32)
        mgr = self._make_manager()
        assert mgr.register_arena(FakeArena(), {"w": tensor}) == b"metadata"

        # Per-tensor, not the pool-reg alloc tuples.
        args, kwargs = mgr._agent.register_memory.call_args
        assert args[0][0] is tensor
        assert kwargs == {"backends": ["UCX"]}

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

    def test_transfer_start_callback_runs_after_descriptor_preparation(
        self, monkeypatch
    ):
        monkeypatch.setattr(torch.cuda, "set_device", lambda *args, **kwargs: None)
        local = torch.zeros(4, dtype=torch.float32)
        mgr = self._make_manager()
        mgr._tensors = {"w": local}
        mgr._agent.prep_xfer_dlist.side_effect = RuntimeError("descriptor failure")
        on_transfer_start = MagicMock()

        with pytest.raises(RuntimeError, match="descriptor failure"):
            mgr.receive_from_source(
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
                on_transfer_start=on_transfer_start,
            )

        on_transfer_start.assert_not_called()


class TestHostMemoryRegistration:
    @staticmethod
    def _make_manager() -> NixlTransferManager:
        mgr = NixlTransferManager(agent_name="test", device_id=3)
        mgr._agent = MagicMock()
        mgr._agent.get_agent_metadata.return_value = b"metadata"
        return mgr

    @staticmethod
    def _gpu_stand_in(numel: int = 4) -> MagicMock:
        tensor = MagicMock(wraps=torch.zeros(numel, dtype=torch.float32))
        tensor.is_cuda = True
        tensor.dtype = torch.float32
        return tensor

    @staticmethod
    def _source(name: str, tensor, mem_type: int) -> TensorDescriptor:
        host = mem_type == p2p_pb2.TENSOR_MEMORY_TYPE_DRAM
        return TensorDescriptor(
            name=name,
            addr=0x9000 if host else 0x1000,
            size=tensor.numel() * tensor.element_size(),
            device_id=0 if host else 5,
            dtype=str(torch.float32),
            mem_type=mem_type,
        )

    @staticmethod
    def _receiving(monkeypatch, mgr: NixlTransferManager) -> None:
        monkeypatch.setattr(torch.cuda, "set_device", lambda *a, **k: None)
        monkeypatch.setattr(torch.cuda, "synchronize", MagicMock())
        mgr._agent.check_xfer_state.return_value = "DONE"

    def test_receive_reads_host_source_into_gpu_tensor(self, monkeypatch):
        local = self._gpu_stand_in()
        mgr = self._make_manager()
        self._receiving(monkeypatch, mgr)
        mgr._tensors = {"w": local}
        source = self._source("w", local, p2p_pb2.TENSOR_MEMORY_TYPE_DRAM)

        mgr.receive_from_source(
            source_metadata=b"", source_tensors=[source], remote_agent_name="source"
        )

        assert mgr._agent.prep_xfer_dlist.call_args_list == [
            call(
                agent_name="source",
                xfer_list=[(0x9000, 16, 0)],
                mem_type="DRAM",
                backends=["UCX"],
            ),
            call(
                agent_name="",
                xfer_list=[(local.data_ptr(), 16, 3)],
                mem_type="VRAM",
                backends=["UCX"],
            ),
        ]

    def test_receive_issues_one_transfer_per_memory_type_pair(self, monkeypatch):
        gpu = self._gpu_stand_in()
        host = torch.zeros(2, dtype=torch.float32)
        mgr = self._make_manager()
        self._receiving(monkeypatch, mgr)
        mgr._tensors = {"gpu": gpu, "host": host}
        mgr._agent.make_prepped_xfer.side_effect = ["vram-handle", "dram-handle"]

        result = mgr.receive_from_source(
            source_metadata=b"",
            source_tensors=[
                self._source("gpu", gpu, p2p_pb2.TENSOR_MEMORY_TYPE_VRAM),
                self._source("host", host, p2p_pb2.TENSOR_MEMORY_TYPE_DRAM),
            ],
            remote_agent_name="source",
        )

        assert result[:2] == (24, 2)
        assert [c.kwargs["mem_type"] for c in mgr._agent.prep_xfer_dlist.call_args_list] == [
            "VRAM", "VRAM", "DRAM", "DRAM",
        ]
        assert mgr._agent.prep_xfer_dlist.call_args_list[3].kwargs["xfer_list"] == [
            (host.data_ptr(), 8, 0)
        ]
        assert mgr._agent.transfer.call_args_list == [
            call("vram-handle"),
            call("dram-handle"),
        ]
        assert mgr._agent.release_xfer_handle.call_args_list == [
            call("vram-handle"),
            call("dram-handle"),
        ]
        torch.cuda.synchronize.assert_called_once_with(3)

    def test_receive_rejects_unknown_source_memory_type(self, monkeypatch):
        local = self._gpu_stand_in()
        mgr = self._make_manager()
        self._receiving(monkeypatch, mgr)
        mgr._tensors = {"w": local}
        outcomes = []
        monkeypatch.setattr(
            nixl_transfer.transfer_metrics, "record_nixl_receive", outcomes.append
        )
        source = self._source("w", local, p2p_pb2.TENSOR_MEMORY_TYPE_VRAM)
        source.mem_type = 7

        with pytest.raises(ManifestMismatchError, match="unknown memory type 7"):
            mgr.receive_from_source(
                source_metadata=b"", source_tensors=[source], remote_agent_name="source"
            )

        assert outcomes == ["rejected"]
        mgr._agent.prep_xfer_dlist.assert_not_called()

    def test_receive_prepares_every_group_before_any_transfer(self, monkeypatch):
        gpu = self._gpu_stand_in()
        host = torch.zeros(2, dtype=torch.float32)
        mgr = self._make_manager()
        self._receiving(monkeypatch, mgr)
        mgr._tensors = {"gpu": gpu, "host": host}
        mgr._agent.prep_xfer_dlist.side_effect = [
            "src", "dst", RuntimeError("second group"),
        ]
        mgr._agent.make_prepped_xfer.return_value = "vram-handle"
        on_transfer_start = MagicMock()

        with pytest.raises(RuntimeError, match="second group"):
            mgr.receive_from_source(
                source_metadata=b"",
                source_tensors=[
                    self._source("gpu", gpu, p2p_pb2.TENSOR_MEMORY_TYPE_VRAM),
                    self._source("host", host, p2p_pb2.TENSOR_MEMORY_TYPE_DRAM),
                ],
                remote_agent_name="source",
                on_transfer_start=on_transfer_start,
            )

        on_transfer_start.assert_not_called()
        mgr._agent.transfer.assert_not_called()
        mgr._agent.release_xfer_handle.assert_called_once_with("vram-handle")

    def test_receive_release_failure_keeps_releasing_and_original_error(
        self, monkeypatch
    ):
        gpu = self._gpu_stand_in()
        host = torch.zeros(2, dtype=torch.float32)
        mgr = self._make_manager()
        self._receiving(monkeypatch, mgr)
        mgr._tensors = {"gpu": gpu, "host": host}
        mgr._agent.make_prepped_xfer.side_effect = ["vram-handle", "dram-handle"]
        mgr._agent.check_xfer_state.return_value = "ERR"
        mgr._agent.release_xfer_handle.side_effect = [RuntimeError("busy"), None]

        with pytest.raises(RuntimeError, match="Transfer failed with status ERR"):
            mgr.receive_from_source(
                source_metadata=b"",
                source_tensors=[
                    self._source("gpu", gpu, p2p_pb2.TENSOR_MEMORY_TYPE_VRAM),
                    self._source("host", host, p2p_pb2.TENSOR_MEMORY_TYPE_DRAM),
                ],
                remote_agent_name="source",
            )

        assert mgr._agent.release_xfer_handle.call_args_list == [
            call("vram-handle"),
            call("dram-handle"),
        ]

    def test_descriptors_record_memory_type_and_host_device(self):
        mgr = self._make_manager()
        descriptors = mgr._build_tensor_descriptors(
            {"host": torch.zeros(4), "accel": self._gpu_stand_in()}
        )
        assert [(d.mem_type, d.device_id) for d in descriptors] == [
            (p2p_pb2.TENSOR_MEMORY_TYPE_DRAM, 0),
            (p2p_pb2.TENSOR_MEMORY_TYPE_VRAM, 3),
        ]

    def test_dram_registration_rejects_accelerator_descriptors(self):
        mgr = self._make_manager()
        (descriptor,) = mgr._build_tensor_descriptors({"accel": self._gpu_stand_in()})

        with pytest.raises(ValueError, match="accel"):
            mgr._register_dram_descriptors([descriptor])
        mgr._agent.register_memory.assert_not_called()

    def test_host_tensors_register_beside_gpu_pool(self, monkeypatch):
        monkeypatch.setenv("MX_POOL_REG", "1")
        gpu = self._gpu_stand_in()
        host = torch.zeros(2, dtype=torch.float32)
        mgr = self._make_manager()
        mgr._find_cuda_allocations = MagicMock(return_value=[(gpu.data_ptr(), 16)])

        assert mgr.register_tensors({"gpu": gpu, "host": host}) == b"metadata"

        assert mgr._agent.register_memory.call_args_list == [
            call([(gpu.data_ptr(), 16, 3, "")], mem_type="VRAM", backends=["UCX"]),
            call([(host.data_ptr(), 8, 0, "")], mem_type="DRAM", backends=["UCX"]),
        ]
        assert (host.data_ptr(), host.data_ptr() + 8) in mgr._base_registration_ranges

    def test_arena_keeps_single_mr_beside_host_tensors(self):
        gpu = self._gpu_stand_in()
        host = torch.zeros(2, dtype=torch.float32)

        class FakeArena:
            live_allocation_count = 1

            def registered_range(self):
                return gpu.data_ptr(), 16

        mgr = self._make_manager()
        assert mgr.register_arena(FakeArena(), {"gpu": gpu, "host": host}) == b"metadata"

        assert mgr._agent.register_memory.call_args_list == [
            call([(gpu.data_ptr(), 16, 3, "")], mem_type="VRAM", backends=["UCX"]),
            call([(host.data_ptr(), 8, 0, "")], mem_type="DRAM", backends=["UCX"]),
        ]

    def test_appended_host_tensor_registers_and_deregisters_as_dram(self, monkeypatch):
        monkeypatch.setenv("MX_POOL_REG", "0")
        gpu = self._gpu_stand_in()
        host = torch.zeros(2, dtype=torch.float32)
        mgr = self._make_manager()
        mgr.register_tensors({"gpu": gpu})
        mgr._agent.register_memory.reset_mock()
        handle = MagicMock()
        mgr._agent.register_memory.return_value = handle

        mgr.register_additional_tensors({"draft.host": host})

        mgr._agent.register_memory.assert_called_once_with(
            [(host.data_ptr(), 8, 0, "")], mem_type="DRAM", backends=["UCX"]
        )
        mgr.deregister_tensors(["draft.host"])
        mgr._agent.deregister_memory.assert_called_once_with(handle)
        assert handle not in mgr._registered_memory

    def test_appended_rollback_releases_every_new_registration(self, monkeypatch):
        monkeypatch.setenv("MX_POOL_REG", "0")
        mgr = self._make_manager()
        mgr.register_tensors({"gpu": self._gpu_stand_in()})
        vram_handle, dram_handle = MagicMock(), MagicMock()
        mgr._agent.register_memory.side_effect = [vram_handle, dram_handle]
        mgr._agent.get_agent_metadata.side_effect = RuntimeError("metadata")

        with pytest.raises(RuntimeError, match="metadata"):
            mgr.register_additional_tensors(
                {"draft.gpu": self._gpu_stand_in(8), "draft.host": torch.zeros(2)}
            )

        assert mgr._agent.deregister_memory.call_args_list == [
            call(vram_handle),
            call(dram_handle),
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
            name="w",
            addr=0x1000,
            size=80,
            device_id=0,
            dtype=str(local.dtype),
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
            name="w",
            addr=0x1000,
            size=40,
            device_id=0,
            dtype="torch.bfloat16",
        )
        with pytest.raises(ManifestMismatchError, match="dtype mismatch"):
            mgr.receive_from_source(
                source_metadata=b"",
                source_tensors=[bogus],
                remote_agent_name="dummy",
            )

    def test_empty_tensor_is_validated_but_not_transferred(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
        weight = torch.ones(4, dtype=torch.float32)
        indices = torch.empty(0, dtype=torch.int32)
        mgr = self._make_manager(
            monkeypatch,
            {"weight": weight, "indices": indices},
        )
        mgr._agent.prep_xfer_dlist.side_effect = ["src", "dst"]
        mgr._agent.make_prepped_xfer.return_value = "handle"
        mgr._agent.check_xfer_state.return_value = "DONE"

        result = mgr.receive_from_source(
            source_metadata=b"",
            source_tensors=[
                TensorDescriptor(
                    "weight",
                    0x1000,
                    weight.numel() * weight.element_size(),
                    0,
                    str(weight.dtype),
                ),
                TensorDescriptor("indices", 0, 0, 0, str(indices.dtype)),
            ],
            remote_agent_name="source",
            require_exact_match=True,
        )

        assert result[:2] == (weight.numel() * weight.element_size(), 2)
        assert [
            entry.kwargs["xfer_list"]
            for entry in mgr._agent.prep_xfer_dlist.call_args_list
        ] == [
            [(0x1000, weight.numel() * weight.element_size(), 0)],
            [(weight.data_ptr(), weight.numel() * weight.element_size(), 0)],
        ]

    def test_null_address_tensor_is_validated_but_not_transferred(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
        weight = torch.ones(4, dtype=torch.float32)
        unallocated = torch.empty(1, dtype=torch.float32, device="meta")
        mgr = self._make_manager(
            monkeypatch,
            {"weight": weight, "unallocated": unallocated},
        )
        mgr._agent.prep_xfer_dlist.side_effect = ["src", "dst"]
        mgr._agent.make_prepped_xfer.return_value = "handle"
        mgr._agent.check_xfer_state.return_value = "DONE"

        result = mgr.receive_from_source(
            source_metadata=b"",
            source_tensors=[
                TensorDescriptor("weight", 0x1000, 16, 0, str(weight.dtype)),
                TensorDescriptor("unallocated", 0, 4, 0, str(unallocated.dtype)),
            ],
            remote_agent_name="source",
        )

        assert result[:2] == (16, 2)
        assert [
            entry.kwargs["xfer_list"]
            for entry in mgr._agent.prep_xfer_dlist.call_args_list
        ] == [
            [(0x1000, 16, 0)],
            [(weight.data_ptr(), 16, 0)],
        ]

    def test_null_address_nonzero_size_skip_warns(self, monkeypatch, caplog):
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
        unallocated = torch.empty(1, dtype=torch.float32, device="meta")
        empty = torch.empty(0, dtype=torch.int32)
        mgr = self._make_manager(
            monkeypatch, {"unallocated": unallocated, "empty": empty}
        )

        with caplog.at_level(logging.WARNING, logger="modelexpress.nixl_transfer"):
            mgr.receive_from_source(
                source_metadata=b"",
                source_tensors=[
                    TensorDescriptor("unallocated", 0, 4, 0, str(unallocated.dtype)),
                    TensorDescriptor("empty", 0, 0, 0, str(empty.dtype)),
                ],
                remote_agent_name="source",
            )

        messages = [rec.getMessage() for rec in caplog.records]
        assert any("tensor 'unallocated'" in m for m in messages)
        assert not any("tensor 'empty'" in m for m in messages)

    def test_all_empty_manifest_needs_no_nixl_transfer(self, monkeypatch):
        indices = torch.empty(0, dtype=torch.int32)
        mgr = self._make_manager(monkeypatch, {"indices": indices})

        result = mgr.receive_from_source(
            source_metadata=b"",
            source_tensors=[TensorDescriptor("indices", 0, 0, 0, str(indices.dtype))],
            remote_agent_name="source",
            require_exact_match=True,
        )

        assert result == (0, 1, 0.0)
        mgr._agent.prep_xfer_dlist.assert_not_called()

    def test_empty_tensor_still_rejects_source_size_mismatch(self, monkeypatch):
        indices = torch.empty(0, dtype=torch.int32)
        mgr = self._make_manager(monkeypatch, {"indices": indices})

        with pytest.raises(ManifestMismatchError, match="size mismatch"):
            mgr.receive_from_source(
                source_metadata=b"",
                source_tensors=[
                    TensorDescriptor("indices", 0x1000, 4, 0, str(indices.dtype))
                ],
                remote_agent_name="source",
                require_exact_match=True,
            )

    def test_size_mismatch_raises_for_heterogeneous_source_device(self, monkeypatch):
        # Heterogeneous transfer signature: the source tensor lives on a
        # different device ordinal (e.g. source xpu:2 -> target cuda:0). The
        # size/dtype validation must fire regardless of the device_id, so
        # relaxing the accelerator policy can never bypass these checks.
        local = torch.zeros(10, dtype=torch.float32)
        mgr = self._make_manager(monkeypatch, {"w": local})
        hetero_bogus = TensorDescriptor(
            name="w",
            addr=0x1000,
            size=80,
            device_id=2,
            dtype=str(local.dtype),
        )
        with pytest.raises(ManifestMismatchError, match="size mismatch"):
            mgr.receive_from_source(
                source_metadata=b"",
                source_tensors=[hetero_bogus],
                remote_agent_name="dummy",
            )

    @pytest.mark.parametrize("strict", [False, True])
    def test_local_only_tensor_raises_in_every_mode(self, monkeypatch, strict):
        # "lm_head" would keep its dummy values while RDMA reports success, so a
        # same-family receive fails closed too.
        embed = torch.full((4,), 7.0)
        lm_head = torch.full((4,), -1.0)
        mgr = self._make_manager(
            monkeypatch, {"embed": embed, "lm_head": lm_head}
        )
        with pytest.raises(
            ManifestMismatchError,
            match=r"1 local-only \(first: \['lm_head'\]\), 0 source-only",
        ):
            mgr.receive_from_source(
                source_metadata=b"",
                source_tensors=[
                    TensorDescriptor("embed", 0x1000, 16, 0, str(embed.dtype))
                ],
                remote_agent_name="dummy",
                require_exact_match=strict,
            )
        mgr._agent.prep_xfer_dlist.assert_not_called()
        assert torch.equal(lm_head, torch.full((4,), -1.0))

    def test_source_only_tensor_warns_without_raising(self, monkeypatch, caplog):
        # A source-only tensor leaves nothing on the target uninitialized.
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
        local = torch.zeros(1, dtype=torch.float32)
        mgr = self._make_manager(monkeypatch, {"w": local})
        mgr._agent.prep_xfer_dlist.side_effect = ["src", "dst"]
        mgr._agent.make_prepped_xfer.return_value = "handle"
        mgr._agent.check_xfer_state.return_value = "DONE"
        with caplog.at_level(logging.WARNING, logger="modelexpress.nixl_transfer"):
            result = mgr.receive_from_source(
                source_metadata=b"",
                source_tensors=[
                    TensorDescriptor("w", 0x1000, 4, 0, "torch.float32"),
                    TensorDescriptor("extra", 0x2000, 4, 0, "torch.float32"),
                ],
                remote_agent_name="dummy",
            )
        assert result[:2] == (4, 1)
        assert any(
            "1 tensors this target does not register (first: ['extra'])"
            in rec.getMessage()
            for rec in caplog.records
        )

    def test_hetero_name_mismatch_raises(self, monkeypatch):
        # Cross-family transfer: local has a tensor the source manifest omits
        # (e.g. a vendor-specific derived tensor). require_exact_match must fail
        # closed rather than transfer a subset and leave "x" at dummy values.
        mgr = self._make_manager(
            monkeypatch,
            {
                "w": torch.zeros(1, dtype=torch.float32),
                "x": torch.zeros(1, dtype=torch.float32),
            },
        )
        src = TensorDescriptor(
            name="w",
            addr=0x1000,
            size=4,
            device_id=0,
            dtype="torch.float32",
        )
        with pytest.raises(ManifestMismatchError, match="Tensor name mismatch"):
            mgr.receive_from_source(
                source_metadata=b"",
                source_tensors=[src],
                remote_agent_name="dummy",
                require_exact_match=True,
            )

    @pytest.mark.parametrize("missing_on", ["receiver", "source"])
    def test_sglang_rejects_missing_derived_tensor(self, monkeypatch, missing_on):
        from types import SimpleNamespace
        from modelexpress.engines.sglang.adapter import SglangAdapter

        adapter = SglangAdapter(
            SimpleNamespace(), SimpleNamespace(),
            SimpleNamespace(device="cuda:0", gpu_id=0),
        )
        local = {"proj.weight": torch.zeros(1)}
        derived = "proj.weight_scale_inv_swizzled"
        source = [TensorDescriptor(
            name="proj.weight", addr=0x1000, size=4,
            device_id=0, dtype="torch.float32",
        )]
        if missing_on == "receiver":
            source.append(TensorDescriptor(
                name=derived, addr=0x2000, size=4,
                device_id=0, dtype="torch.float32",
            ))
        else:
            local[derived] = torch.zeros(1)
        mgr = self._make_manager(monkeypatch, local)
        mgr._agent.check_xfer_state.return_value = "DONE"
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
        with pytest.raises(ManifestMismatchError, match="Tensor name mismatch"):
            mgr.receive_from_source(
                source_metadata=b"", source_tensors=source,
                remote_agent_name="dummy",
                require_exact_match=adapter.requires_exact_tensor_catalog(),
            )
        mgr._agent.prep_xfer_dlist.assert_not_called()
        mgr._agent.make_prepped_xfer.assert_not_called()

    def test_hetero_source_only_name_mismatch_raises(self, monkeypatch):
        # Source names a tensor the target never registered.
        mgr = self._make_manager(
            monkeypatch, {"w": torch.zeros(1, dtype=torch.float32)}
        )
        src = [
            TensorDescriptor(
                name="w",
                addr=0x1000,
                size=4,
                device_id=0,
                dtype="torch.float32",
            ),
            TensorDescriptor(
                name="extra",
                addr=0x2000,
                size=4,
                device_id=0,
                dtype="torch.float32",
            ),
        ]
        with pytest.raises(ManifestMismatchError, match="Tensor name mismatch"):
            mgr.receive_from_source(
                source_metadata=b"",
                source_tensors=src,
                remote_agent_name="dummy",
                require_exact_match=True,
            )

    def test_hetero_zero_match_raises(self, monkeypatch):
        # No overlapping names at all: a zero-match cross-family transfer would
        # report RDMA success while writing nothing. Fail closed.
        mgr = self._make_manager(
            monkeypatch, {"x": torch.zeros(1, dtype=torch.float32)}
        )
        src = TensorDescriptor(
            name="w",
            addr=0x1000,
            size=4,
            device_id=0,
            dtype="torch.float32",
        )
        with pytest.raises(ManifestMismatchError, match="Tensor name mismatch"):
            mgr.receive_from_source(
                source_metadata=b"",
                source_tensors=[src],
                remote_agent_name="dummy",
                require_exact_match=True,
            )

    def test_hetero_matched_name_passes_guard(self, monkeypatch):
        # Identical name sets must clear the name-mismatch guard: with only "w"
        # on both sides and require_exact_match=True, execution proceeds past the
        # guard to the size check, which here fails on a deliberate size skew.
        # A "Tensor name mismatch" error would mean the name guard fired
        # spuriously; a "size mismatch" proves it did not.
        local = torch.zeros(10, dtype=torch.float32)
        mgr = self._make_manager(monkeypatch, {"w": local})
        src = TensorDescriptor(
            name="w",
            addr=0x1000,
            size=80,
            device_id=0,
            dtype=str(local.dtype),
        )
        with pytest.raises(ManifestMismatchError, match="size mismatch"):
            mgr.receive_from_source(
                source_metadata=b"",
                source_tensors=[src],
                remote_agent_name="dummy",
                require_exact_match=True,
            )

    def test_explicit_destination_catalog_overrides_latest_registration(
        self, monkeypatch
    ):
        canonical = torch.zeros(10, dtype=torch.float32)
        auxiliary = torch.zeros(1, dtype=torch.float32)
        mgr = self._make_manager(monkeypatch, {"__convert__w": auxiliary})
        src = TensorDescriptor(
            name="w",
            addr=0x1000,
            size=80,
            device_id=0,
            dtype=str(canonical.dtype),
        )
        with pytest.raises(ManifestMismatchError, match="size mismatch"):
            mgr.receive_from_source(
                source_metadata=b"",
                source_tensors=[src],
                remote_agent_name="dummy",
                require_exact_match=True,
                destination_tensors={"w": canonical},
            )


class TestWaitForXfer:
    @staticmethod
    def _manager(statuses):
        manager = object.__new__(NixlTransferManager)
        manager._agent = MagicMock()
        manager._agent.check_xfer_state.side_effect = statuses
        return manager

    def test_returns_on_success(self):
        manager = self._manager(["PENDING", "SUCCESS"])
        manager._wait_for_xfer(object(), None, "test transfer")
        assert manager._agent.check_xfer_state.call_count == 2

    def test_raises_labeled_error(self):
        manager = self._manager(["ERROR"])
        with pytest.raises(RuntimeError, match="test transfer failed"):
            manager._wait_for_xfer(object(), None, "test transfer")


class TestArenaSingleMrForced:
    """MX_ARENA_SINGLE_MR keeps the single-MR path on a multi-allocation arena."""

    def test_default_is_off(self, monkeypatch):
        monkeypatch.delenv("MX_ARENA_SINGLE_MR", raising=False)
        assert _arena_single_mr_forced() is False

    def test_one_is_on(self, monkeypatch):
        monkeypatch.setenv("MX_ARENA_SINGLE_MR", "1")
        assert _arena_single_mr_forced() is True

    def test_read_at_call_time(self, monkeypatch):
        monkeypatch.setenv("MX_ARENA_SINGLE_MR", "1")
        assert _arena_single_mr_forced() is True
        monkeypatch.setenv("MX_ARENA_SINGLE_MR", "0")
        assert _arena_single_mr_forced() is False


class TestMultiAllocationArena:
    """A single MR cannot describe an arena spanning several cuMemCreate handles.

    UCX cuda_ipc resolves a region with cuMemGetAddressRange, which reports only
    the allocation holding the base pointer, so the published rkey would cover
    just the first chunk and the peer would read past what it mapped.
    """

    @staticmethod
    def _make_manager() -> NixlTransferManager:
        mgr = NixlTransferManager(
            agent_name="test", device_id=0, accelerator_backend=_StandInCudaBackend()
        )
        mgr._agent = MagicMock()
        mgr._agent.get_agent_metadata.return_value = b"metadata"
        return mgr

    @staticmethod
    def _covering_arena(tensor, live_allocs):
        base = tensor.data_ptr()
        used = tensor.numel() * tensor.element_size()

        class FakeArena:
            live_allocation_count = live_allocs

            def registered_range(self):
                return base, used

        return FakeArena(), base, used

    def test_multi_allocation_arena_registers_per_tensor(self, monkeypatch):
        monkeypatch.delenv("MX_ARENA_SINGLE_MR", raising=False)
        tensor = torch.zeros(4, dtype=torch.float32)
        arena, _, _ = self._covering_arena(tensor, live_allocs=1019)

        mgr = self._make_manager()
        assert mgr.register_arena(arena, {"w": tensor}) == b"metadata"

        # Per-tensor, not one MR over the arena range.
        args, kwargs = mgr._agent.register_memory.call_args
        assert args[0][0] is tensor
        assert kwargs == {"backends": ["UCX"]}

    def test_single_allocation_arena_keeps_one_mr(self, monkeypatch):
        monkeypatch.delenv("MX_ARENA_SINGLE_MR", raising=False)
        tensor = torch.zeros(4, dtype=torch.float32)
        arena, base, used = self._covering_arena(tensor, live_allocs=1)

        mgr = self._make_manager()
        assert mgr.register_arena(arena, {"w": tensor}) == b"metadata"

        mgr._agent.register_memory.assert_called_once_with(
            [(base, used, 0, "")],
            mem_type=NIXL_ACCELERATOR_MEM_TYPE,
            backends=["UCX"],
        )

    def test_env_override_keeps_one_mr_on_multi_allocation(self, monkeypatch):
        # dmabuf/IB can span several handles in one registration, so deployments
        # that validated the single-MR path there can keep it.
        monkeypatch.setenv("MX_ARENA_SINGLE_MR", "1")
        tensor = torch.zeros(4, dtype=torch.float32)
        arena, base, used = self._covering_arena(tensor, live_allocs=1019)

        mgr = self._make_manager()
        assert mgr.register_arena(arena, {"w": tensor}) == b"metadata"

        mgr._agent.register_memory.assert_called_once_with(
            [(base, used, 0, "")],
            mem_type=NIXL_ACCELERATOR_MEM_TYPE,
            backends=["UCX"],
        )

    def test_multi_allocation_fallback_bypasses_pool_registration(self, monkeypatch):
        # Pool reg resolves the same per-handle bounds that were insufficient.
        monkeypatch.delenv("MX_ARENA_SINGLE_MR", raising=False)
        monkeypatch.setenv("MX_POOL_REG", "1")
        tensor = torch.zeros(4, dtype=torch.float32)
        arena, _, _ = self._covering_arena(tensor, live_allocs=1019)

        mgr = self._make_manager()
        assert mgr.register_arena(arena, {"w": tensor}) == b"metadata"

        args, kwargs = mgr._agent.register_memory.call_args
        assert args[0][0] is tensor
        assert kwargs == {"backends": ["UCX"]}

    def test_warning_names_the_allocation_count(self, monkeypatch, caplog):
        monkeypatch.delenv("MX_ARENA_SINGLE_MR", raising=False)
        tensor = torch.zeros(4, dtype=torch.float32)
        arena, _, _ = self._covering_arena(tensor, live_allocs=1019)

        mgr = self._make_manager()
        with caplog.at_level(logging.WARNING):
            mgr.register_arena(arena, {"w": tensor})

        assert "1019 physical allocations" in caplog.text
        assert "MX_ARENA_SINGLE_MR=1" in caplog.text


class _MetricsSpy:
    """Captures the label each ``record_nixl_*`` call site passes.

    Stands in for the module-level ``transfer_metrics`` singleton. Any other
    metric the transfer path records is irrelevant here and is swallowed.
    """

    def __init__(self) -> None:
        self.receives: list[str] = []
        self.errors: list[str] = []

    def record_nixl_receive(self, result: str) -> None:
        self.receives.append(result)

    def record_nixl_error(self, kind: str) -> None:
        self.errors.append(kind)

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


class TestAddresslessTensorTransfers:
    def _manager(self, monkeypatch, tensors):
        monkeypatch.setenv("MX_POOL_REG", "0")
        monkeypatch.setattr(torch.cuda, "set_device", lambda *a, **k: None)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
        manager = NixlTransferManager(
            agent_name="test", device_id=0, accelerator_backend=_StandInCudaBackend()
        )
        manager._agent = MagicMock()
        manager._agent.get_agent_metadata.return_value = b"metadata"
        manager.register_tensors(tensors)
        manager._agent.prep_xfer_dlist.side_effect = ["src", "dst"]
        manager._agent.make_prepped_xfer.return_value = "handle"
        manager._agent.check_xfer_state.return_value = "DONE"
        return manager

    def _outcomes(self, monkeypatch):
        outcomes = []
        monkeypatch.setattr(
            nixl_transfer.transfer_metrics, "record_nixl_receive", outcomes.append
        )
        return outcomes

    @pytest.mark.parametrize("strict", [False, True])
    @pytest.mark.parametrize("mixed", [False, True])
    def test_unallocated_source_is_rejected_for_allocated_target(
        self, monkeypatch, strict, mixed
    ):
        source_tensors = {"w": torch.empty(1, device="meta")}
        local_tensors = {"w": torch.full((1,), 7.0)}
        if mixed:
            source_tensors["live"] = torch.ones(2)
            local_tensors["live"] = torch.zeros(2)
        source = self._manager(monkeypatch, source_tensors)
        target = self._manager(monkeypatch, local_tensors)
        outcomes = self._outcomes(monkeypatch)

        with pytest.raises(ManifestMismatchError, match="w.*source.*address"):
            target.receive_from_source(
                source_metadata=b"metadata",
                source_tensors=source.tensor_descriptors,
                remote_agent_name="source",
                require_exact_match=strict,
            )

        target._agent.prep_xfer_dlist.assert_not_called()
        assert outcomes == ["rejected"]
        assert local_tensors["w"].item() == 7.0

    @pytest.mark.parametrize("strict", [False, True])
    def test_unallocated_destination_is_rejected(self, monkeypatch, strict):
        source = self._manager(monkeypatch, {"w": torch.ones(1)})
        target = self._manager(monkeypatch, {"w": torch.empty(1, device="meta")})
        outcomes = self._outcomes(monkeypatch)

        with pytest.raises(ManifestMismatchError, match="w.*local.*address"):
            target.receive_from_source(
                source_metadata=b"metadata",
                source_tensors=source.tensor_descriptors,
                remote_agent_name="source",
                require_exact_match=strict,
            )

        target._agent.prep_xfer_dlist.assert_not_called()
        assert outcomes == ["rejected"]

    def test_registered_empty_tensors_still_complete_in_strict_mode(self, monkeypatch):
        tensors = {"a": torch.empty(0), "b": torch.empty(0, dtype=torch.int32)}
        source = self._manager(monkeypatch, tensors)
        target = self._manager(monkeypatch, tensors)
        outcomes = self._outcomes(monkeypatch)

        assert target.receive_from_source(
            source_metadata=b"metadata",
            source_tensors=source.tensor_descriptors,
            remote_agent_name="source",
            require_exact_match=True,
        ) == (0, 2, 0.0)

        target._agent.prep_xfer_dlist.assert_not_called()
        assert outcomes == ["complete"]

    @pytest.mark.parametrize("kind", ["empty", "meta", "noncontiguous_meta"])
    def test_appended_no_payload_tensor_needs_no_registration(self, monkeypatch, kind):
        tensors = {
            "empty": torch.empty(0),
            "meta": torch.empty(1, device="meta"),
            "noncontiguous_meta": torch.empty((2, 3), device="meta").T,
        }
        tensor = tensors[kind]
        target = self._manager(monkeypatch, {"main": torch.ones(1)})
        target._agent.register_memory.reset_mock()

        assert target.register_additional_tensors({"draft": tensor}) == b"metadata"

        target._agent.register_memory.assert_not_called()
        assert target._tensors["draft"] is tensor
        assert target.tensor_descriptors[-1] == TensorDescriptor(
            name="draft",
            addr=tensor.data_ptr(),
            size=tensor.numel() * tensor.element_size(),
            device_id=0,
            dtype=str(tensor.dtype),
        )
        assert target._appended_registrations["draft"] is None
        target.deregister_tensors(["draft"])
        target._agent.deregister_memory.assert_not_called()
        assert list(target._tensors) == ["main"]

    def test_appending_mixed_tensors_registers_only_allocated_storage(self, monkeypatch):
        target = self._manager(monkeypatch, {"main": torch.ones(1)})
        target._agent.register_memory.reset_mock()
        live = torch.ones(2)

        target.register_additional_tensors(
            {"unallocated": torch.empty(1, device="meta"), "live": live}
        )

        target._agent.register_memory.assert_called_once_with([live], backends=["UCX"])
        assert target._appended_registrations["unallocated"] is None
        assert list(target._tensors) == ["main", "unallocated", "live"]


class TestReceiveOutcomeLabelsAtTheCallSite:
    """Pins which label each ``record_nixl_*`` call site in the receive path picks.

    ``test_metrics.py`` proves the recorder counts once and clamps unknown values
    *given* a label, but nothing there constrains the callers. Without these
    tests every call site in ``nixl_transfer`` could pass the wrong constant --
    an ``empty`` receive booked as ``complete``, a wedged QP booked as
    ``status_error`` -- and the suite would stay green while the dashboards lied.
    Each assertion is on the whole recorded list, not on membership, so a double
    count fails too: ``partial`` in particular is set at one site and recorded at
    another, and the gap between them is exactly where a second count would hide.
    """

    def _manager(self, monkeypatch, local_tensors):
        # CPU host: the accelerator backend delegates both of these to torch.cuda.
        monkeypatch.setattr(torch.cuda, "set_device", lambda *a, **k: None)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
        mgr = NixlTransferManager(agent_name="test", device_id=0)
        mgr._agent = MagicMock()
        mgr._tensors = local_tensors
        return mgr

    def _spy(self, monkeypatch) -> _MetricsSpy:
        spy = _MetricsSpy()
        monkeypatch.setattr(nixl_transfer, "transfer_metrics", spy)
        return spy

    def _arm(self, mgr, state: str) -> None:
        """Drive the RDMA calls to a chosen terminal ``check_xfer_state``."""
        mgr._agent.prep_xfer_dlist.side_effect = ["src", "dst"]
        mgr._agent.make_prepped_xfer.return_value = "handle"
        mgr._agent.check_xfer_state.return_value = state

    @staticmethod
    def _matching(name: str, tensor: torch.Tensor) -> TensorDescriptor:
        return TensorDescriptor(
            name=name,
            addr=0x1000,
            size=tensor.numel() * tensor.element_size(),
            device_id=0,
            dtype=str(tensor.dtype),
        )

    def test_a_clean_transfer_counts_one_complete(self, monkeypatch):
        local = torch.zeros(4, dtype=torch.float32)
        spy = self._spy(monkeypatch)
        mgr = self._manager(monkeypatch, {"w": local})
        self._arm(mgr, "DONE")

        mgr.receive_from_source(
            source_metadata=b"",
            source_tensors=[self._matching("w", local)],
            remote_agent_name="source",
        )

        assert spy.receives == ["complete"]
        assert spy.errors == []

    def test_a_source_only_diff_counts_one_partial_and_not_a_complete(
        self, monkeypatch
    ):
        local = torch.zeros(4, dtype=torch.float32)
        spy = self._spy(monkeypatch)
        mgr = self._manager(monkeypatch, {"w": local})
        self._arm(mgr, "DONE")

        mgr.receive_from_source(
            source_metadata=b"",
            source_tensors=[
                self._matching("w", local),
                self._matching("unmatched", local),
            ],
            remote_agent_name="source",
        )

        assert spy.receives == ["partial"]
        assert spy.errors == []

    def test_a_zero_match_transfer_counts_one_empty(self, monkeypatch):
        # Returns (0, 0, 0.0) -- success to the caller, nothing moved on the wire.
        local = torch.zeros(4, dtype=torch.float32)
        spy = self._spy(monkeypatch)
        mgr = self._manager(monkeypatch, {})

        assert mgr.receive_from_source(
            source_metadata=b"",
            source_tensors=[self._matching("w", local)],
            remote_agent_name="source",
        ) == (0, 0, 0.0)

        assert spy.receives == ["empty"]
        assert spy.errors == []

    def test_every_refusal_counts_one_rejected(self, monkeypatch):
        """All four ``ManifestMismatchError`` sites book the same refusal label.

        A refusal raises instead of returning, so without these the family would
        only partition the receives that succeeded.
        """
        local = torch.zeros(10, dtype=torch.float32)
        size = local.numel() * local.element_size()
        refusals = {
            "size mismatch": (
                {"w": local},
                [TensorDescriptor("w", 0x1000, size * 2, 0, str(local.dtype))],
                False,
            ),
            "dtype mismatch": (
                {"w": local},
                [TensorDescriptor("w", 0x1000, size, 0, "torch.bfloat16")],
                False,
            ),
            "Tensor name mismatch": (
                {"w": local, "local_only": local},
                [TensorDescriptor("w", 0x1000, size, 0, str(local.dtype))],
                True,
            ),
            # Both sides empty: the only way past the name-diff check above,
            # which otherwise fires first on any zero-match manifest.
            "No matching tensors": ({}, [], True),
        }

        for match, (tensors, source, strict) in refusals.items():
            spy = self._spy(monkeypatch)
            mgr = self._manager(monkeypatch, tensors)
            with pytest.raises(ManifestMismatchError, match=match):
                mgr.receive_from_source(
                    source_metadata=b"",
                    source_tensors=source,
                    remote_agent_name="source",
                    require_exact_match=strict,
                )
            assert spy.receives == ["rejected"], f"{match} was not booked rejected"
            assert spy.errors == []

    def test_a_wedged_transfer_counts_a_timeout_and_no_receive(self, monkeypatch):
        # A wedged QP yields neither a completion nor an ERR status, so the
        # timeout is the only evidence. It must not be folded into status_error.
        local = torch.zeros(4, dtype=torch.float32)
        spy = self._spy(monkeypatch)
        mgr = self._manager(monkeypatch, {"w": local})
        self._arm(mgr, "PENDING")

        with pytest.raises(TimeoutError):
            mgr.receive_from_source(
                source_metadata=b"",
                source_tensors=[self._matching("w", local)],
                remote_agent_name="source",
                timeout_seconds=0.0,
            )

        assert spy.errors == ["timeout"]
        assert spy.receives == []

    def test_an_err_status_counts_a_status_error_and_no_receive(self, monkeypatch):
        local = torch.zeros(4, dtype=torch.float32)
        spy = self._spy(monkeypatch)
        mgr = self._manager(monkeypatch, {"w": local})
        self._arm(mgr, "ERR")

        with pytest.raises(RuntimeError, match="failed with status ERR"):
            mgr.receive_from_source(
                source_metadata=b"",
                source_tensors=[self._matching("w", local)],
                remote_agent_name="source",
            )

        assert spy.errors == ["status_error"]
        assert spy.receives == []

    # The batched reshard path (`await_read_batches` -> `_wait_for_xfers`) has its
    # own copy of the two failure classifications. Driving the wait directly keeps
    # this to the classification under test rather than a fabricated batch handle.

    def test_a_wedged_batch_counts_a_timeout(self, monkeypatch):
        spy = self._spy(monkeypatch)
        mgr = self._manager(monkeypatch, {})
        mgr._agent.check_xfer_state.return_value = "PENDING"

        with pytest.raises(TimeoutError):
            mgr._wait_for_xfers(["h1", "h2"], 0.0, "NIXL reshard READ batch")

        assert spy.errors == ["timeout"]
        assert spy.receives == []

    def test_an_err_status_in_a_batch_counts_a_status_error(self, monkeypatch):
        spy = self._spy(monkeypatch)
        mgr = self._manager(monkeypatch, {})
        mgr._agent.check_xfer_state.return_value = "ERR"

        with pytest.raises(RuntimeError, match="failed with status ERR"):
            mgr._wait_for_xfers(["h1", "h2"], None, "NIXL reshard READ batch")

        assert spy.errors == ["status_error"]
        assert spy.receives == []
