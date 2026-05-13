# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the VMM-range registration path in NixlTransferManager.

The compaction primitive itself (vmm_compact.compact_tensors) requires a
real CUDA device and is exercised by the bench harness on the cluster.
Here we cover the seam where compact_tensors's output meets NIXL
registration: when a vmm_range is provided, register_tensors must
register exactly one region and must NOT call _find_cuda_allocations.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from modelexpress.nixl_transfer import NixlTransferManager


def _mgr(monkeypatch) -> NixlTransferManager:
    # Bypass torch.cuda.set_device so this can run on a CPU host.
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args, **kwargs: None)
    m = NixlTransferManager(agent_name="test", device_id=0)
    m._agent = MagicMock()
    m._agent.get_agent_metadata.return_value = b"meta"
    return m


def _cpu_tensors() -> dict[str, torch.Tensor]:
    # data_ptrs are irrelevant when vmm_range short-circuits allocation
    # discovery, but the tensors must be contiguous (the early sanity loop
    # rejects non-contiguous tensors regardless of mode).
    return {
        "w0": torch.zeros(4, dtype=torch.float32),
        "w1": torch.zeros(8, dtype=torch.float32),
    }


class TestVmmRangeRegistration:
    def test_vmm_range_registers_single_region(self, monkeypatch):
        m = _mgr(monkeypatch)
        m.register_tensors(_cpu_tensors(), vmm_range=(0x10000, 0x40000))

        # register_memory must have been called once with the single
        # (base, size, device_id, "") tuple, mem_type="cuda".
        m._agent.register_memory.assert_called_once()
        args, kwargs = m._agent.register_memory.call_args
        alloc_list = args[0]
        assert alloc_list == [(0x10000, 0x40000, 0, "")]
        assert kwargs.get("mem_type") == "cuda"

    def test_vmm_range_bypasses_alloc_discovery(self, monkeypatch):
        # MX_POOL_REG=1 would normally trigger _find_cuda_allocations; with
        # vmm_range set, that path must be skipped (single VMM region wins).
        monkeypatch.setenv("MX_POOL_REG", "1")
        m = _mgr(monkeypatch)

        spy = MagicMock(side_effect=AssertionError(
            "_find_cuda_allocations must not be called when vmm_range is set"
        ))
        monkeypatch.setattr(NixlTransferManager, "_find_cuda_allocations", spy)

        m.register_tensors(_cpu_tensors(), vmm_range=(0x10000, 0x40000))
        spy.assert_not_called()

    def test_vmm_range_preserves_per_tensor_descriptors(self, monkeypatch):
        # Application-level name-matching during transfer relies on
        # per-tensor descriptors regardless of registration mode.
        m = _mgr(monkeypatch)
        tensors = _cpu_tensors()
        m.register_tensors(tensors, vmm_range=(0x10000, 0x40000))

        names = {d.name for d in m.tensor_descriptors}
        assert names == set(tensors.keys())
        assert len(m.tensor_descriptors) == len(tensors)

    def test_non_contiguous_tensor_rejected(self, monkeypatch):
        # Stride-view tensors cannot be RDMA targets; the contiguity gate
        # fires before mode selection.
        m = _mgr(monkeypatch)
        base = torch.zeros(8, dtype=torch.float32)
        non_contig = base[::2]  # strided view
        with pytest.raises(RuntimeError, match="not contiguous"):
            m.register_tensors(
                {"bad": non_contig}, vmm_range=(0x10000, 0x40000)
            )

    def test_returns_metadata_blob(self, monkeypatch):
        m = _mgr(monkeypatch)
        m._agent.get_agent_metadata.return_value = b"some-nixl-meta"
        result = m.register_tensors(_cpu_tensors(), vmm_range=(0x1000, 0x2000))
        assert result == b"some-nixl-meta"

    def test_default_vmm_range_none_preserves_backward_compat(self, monkeypatch):
        # Without vmm_range and without MX_POOL_REG, must fall through to
        # per-tensor registration (register the tensor list directly).
        monkeypatch.delenv("MX_POOL_REG", raising=False)
        m = _mgr(monkeypatch)
        tensors = _cpu_tensors()
        m.register_tensors(tensors)

        m._agent.register_memory.assert_called_once()
        args, kwargs = m._agent.register_memory.call_args
        # First positional arg is the tensor list, not a (base, size, ...) tuple list.
        assert args[0] == list(tensors.values())
        # mem_type should not be set in per-tensor mode (NIXL infers from tensors).
        assert "mem_type" not in kwargs
