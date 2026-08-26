# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for allocation discovery (cuMemGetAddressRange), the MX_POOL_REG toggle,
and receive_from_source manifest validation."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, call

import pytest
import torch

from modelexpress import nixl_transfer
from modelexpress.accelerators import NIXL_ACCELERATOR_MEM_TYPE
from modelexpress.nixl_transfer import (
    _arena_single_mr_forced,
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

    def test_size_mismatch_raises_for_heterogeneous_source_device(self, monkeypatch):
        # Heterogeneous transfer signature: the source tensor lives on a
        # different device ordinal (e.g. source xpu:2 -> target cuda:0). The
        # size/dtype validation must fire regardless of the device_id, so
        # relaxing the accelerator policy can never bypass these checks.
        local = torch.zeros(10, dtype=torch.float32)
        mgr = self._make_manager(monkeypatch, {"w": local})
        hetero_bogus = TensorDescriptor(
            name="w", addr=0x1000, size=80, device_id=2, dtype=str(local.dtype),
        )
        with pytest.raises(ManifestMismatchError, match="size mismatch"):
            mgr.receive_from_source(
                source_metadata=b"",
                source_tensors=[hetero_bogus],
                remote_agent_name="dummy",
            )

    def test_unmatched_name_skips_silently(self, monkeypatch):
        # No matching local tensor for the source's "w". Loop should `continue`
        # without raising; the caller decides whether the empty match list is
        # an error. We just verify the validation doesn't fire spuriously.
        mgr = self._make_manager(
            monkeypatch, {"x": torch.zeros(1, dtype=torch.float32)}
        )
        wrong_name = TensorDescriptor(
            name="w",
            addr=0x1000,
            size=4,
            device_id=0,
            dtype="torch.float32",
        )
        # Empty match -> early-return (0, 0, 0.0); no exception.
        result = mgr.receive_from_source(
            source_metadata=b"",
            source_tensors=[wrong_name],
            remote_agent_name="dummy",
        )
        assert result == (0, 0, 0.0)

    def test_unmatched_name_warns(self, monkeypatch, caplog):
        # Unmatched names stay non-fatal but are surfaced: an unmatched local
        # tensor keeps its init values.
        mgr = self._make_manager(
            monkeypatch, {"x": torch.zeros(1, dtype=torch.float32)}
        )
        wrong_name = TensorDescriptor(
            name="w",
            addr=0x1000,
            size=4,
            device_id=0,
            dtype="torch.float32",
        )
        with caplog.at_level(logging.WARNING, logger="modelexpress.nixl_transfer"):
            mgr.receive_from_source(
                source_metadata=b"",
                source_tensors=[wrong_name],
                remote_agent_name="dummy",
            )
        assert any(
            "1 local-only, 1 source-only" in rec.getMessage() for rec in caplog.records
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
            name="w", addr=0x1000, size=4, device_id=0, dtype="torch.float32",
        )
        with pytest.raises(ManifestMismatchError, match="heterogeneous transfer"):
            mgr.receive_from_source(
                source_metadata=b"",
                source_tensors=[src],
                remote_agent_name="dummy",
                require_exact_match=True,
            )

    def test_hetero_source_only_name_mismatch_raises(self, monkeypatch):
        # Source names a tensor the target never registered.
        mgr = self._make_manager(
            monkeypatch, {"w": torch.zeros(1, dtype=torch.float32)}
        )
        src = [
            TensorDescriptor(
                name="w", addr=0x1000, size=4, device_id=0, dtype="torch.float32",
            ),
            TensorDescriptor(
                name="extra", addr=0x2000, size=4, device_id=0, dtype="torch.float32",
            ),
        ]
        with pytest.raises(ManifestMismatchError, match="heterogeneous transfer"):
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
            name="w", addr=0x1000, size=4, device_id=0, dtype="torch.float32",
        )
        with pytest.raises(ManifestMismatchError, match="heterogeneous transfer"):
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
        # A ManifestMismatchError about "heterogeneous transfer" would mean the
        # name guard fired spuriously; a "size mismatch" proves it did not.
        local = torch.zeros(10, dtype=torch.float32)
        mgr = self._make_manager(monkeypatch, {"w": local})
        src = TensorDescriptor(
            name="w", addr=0x1000, size=80, device_id=0, dtype=str(local.dtype),
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
            name="w", addr=0x1000, size=80, device_id=0, dtype=str(canonical.dtype),
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
        mgr = NixlTransferManager(agent_name="test", device_id=0)
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

    def test_a_name_diff_counts_one_partial_and_not_a_complete(self, monkeypatch):
        # A local-only tensor keeps its dummy values while the transfer still
        # returns success, which is the whole reason `partial` exists.
        local = torch.zeros(4, dtype=torch.float32)
        spy = self._spy(monkeypatch)
        mgr = self._manager(monkeypatch, {"w": local, "unmatched": local})
        self._arm(mgr, "DONE")

        mgr.receive_from_source(
            source_metadata=b"",
            source_tensors=[self._matching("w", local)],
            remote_agent_name="source",
        )

        assert spy.receives == ["partial"]
        assert spy.errors == []

    def test_a_zero_match_transfer_counts_one_empty(self, monkeypatch):
        # Returns (0, 0, 0.0) -- success to the caller, nothing moved on the wire.
        local = torch.zeros(4, dtype=torch.float32)
        spy = self._spy(monkeypatch)
        mgr = self._manager(monkeypatch, {"x": local})

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
