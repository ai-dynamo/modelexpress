# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NIXL local-memtype auto-detect and rebind_tensors.

Phase 0.5 pinned-CPU staging (Istvan 2.5): the NIXL manager now tracks
the memtype of the currently-active local tensor set so ``prep_xfer_dlist``
picks the right memtype on the receiver side. Uniformity is enforced at
:func:`register_tensors` and :meth:`rebind_tensors` time so mixed-device
sets fail fast rather than mid-transfer.
"""

from __future__ import annotations

import pytest
import torch

from modelexpress.nixl_transfer import (
    NixlTransferManager,
    _MEM_TYPE_CUDA,
    _MEM_TYPE_DRAM,
    _resolve_local_mem_type,
)


def test_memtype_constants_match_nixl_python_api():
    """Guard against regressions on the NIXL memtype key names.

    NIXL's Python API (``nixl_cu12._api``) accepts a fixed set of
    memtype strings — case-sensitive — in its ``nixl_mems`` dict. Using
    the wrong key (e.g. lowercase "dram") passes ``register_memory``
    (which auto-detects from torch tensors) but fails at
    ``prep_xfer_dlist`` with KeyError. Lock the constants to
    NIXL-accepted values so future edits don't reintroduce this
    subtle bug.
    """
    # NIXL Python API accepts: "DRAM", "VRAM", "cpu", "cuda" (case-sensitive)
    # for the CUDA and CPU-DRAM memtypes we care about.
    assert _MEM_TYPE_CUDA in ("cuda", "VRAM"), (
        f"CUDA memtype {_MEM_TYPE_CUDA!r} not a valid NIXL key"
    )
    assert _MEM_TYPE_DRAM in ("cpu", "DRAM"), (
        f"DRAM memtype {_MEM_TYPE_DRAM!r} not a valid NIXL key"
    )


class TestResolveLocalMemType:
    """The uniformity resolver runs before NIXL is touched, so it works
    on any host — no CUDA required for the CPU-only cases.
    """

    def test_all_cpu_returns_dram(self):
        tensors = {
            "a": torch.empty(4, dtype=torch.bfloat16),
            "b": torch.empty(8, dtype=torch.bfloat16),
        }
        assert _resolve_local_mem_type(tensors) == _MEM_TYPE_DRAM

    def test_all_cpu_pinned_returns_dram(self):
        tensors = {
            "a": torch.empty(4, dtype=torch.bfloat16, pin_memory=torch.cuda.is_available()),
        }
        assert _resolve_local_mem_type(tensors) == _MEM_TYPE_DRAM

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_all_cuda_returns_cuda(self):
        tensors = {
            "a": torch.empty(4, dtype=torch.bfloat16, device="cuda"),
            "b": torch.empty(8, dtype=torch.bfloat16, device="cuda"),
        }
        assert _resolve_local_mem_type(tensors) == _MEM_TYPE_CUDA

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty tensor set"):
            _resolve_local_mem_type({})

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_mixed_devices_raises(self):
        tensors = {
            "cpu": torch.empty(4, dtype=torch.bfloat16),
            "gpu": torch.empty(4, dtype=torch.bfloat16, device="cuda"),
        }
        with pytest.raises(ValueError, match="mixed or unsupported"):
            _resolve_local_mem_type(tensors)


class TestManagerLocalMemTypeDefault:
    """Constructing the manager without touching NIXL is CPU-safe; we
    just want to verify the class attribute defaults are sane so a
    caller who hits an early error path (before register_tensors) sees
    a consistent state.
    """

    def test_default_is_cuda(self):
        mgr = NixlTransferManager(agent_name="test", device_id=0)
        assert mgr._local_mem_type == _MEM_TYPE_CUDA


class _FakeAgent:
    """Minimal ``nixl_agent`` stub for exercising register_tensors +
    rebind_tensors control flow without a real NIXL runtime.

    We only need to observe (a) which mem_type was passed to
    ``register_memory`` and (b) that ``rebind_tensors`` does not call it
    at all. The rest of the API (get_agent_metadata, prep_xfer_dlist)
    is stubbed with zero-behavior methods.
    """

    def __init__(self) -> None:
        self.register_calls: list[tuple[str, int]] = []
        self.metadata_returns = b"stub-metadata"

    def register_memory(self, tensors_or_tuples, *, mem_type=None, backends=None):
        self.register_calls.append((mem_type, len(list(tensors_or_tuples))))

    def get_agent_metadata(self):
        return self.metadata_returns


class TestRegisterTensorsMemType:
    """``register_tensors`` must set ``_local_mem_type`` from the tensors'
    device and pass the matching ``mem_type`` to
    ``NixlAgent.register_memory``.
    """

    def _mk_manager(self):
        mgr = NixlTransferManager(agent_name="test", device_id=0)
        mgr._agent = _FakeAgent()
        return mgr

    def test_cpu_tensors_register_as_dram(self, monkeypatch):
        # MX_POOL_REG only affects CUDA discovery; explicitly off for
        # this test so we hit the per-tensor register path.
        monkeypatch.delenv("MX_POOL_REG", raising=False)
        mgr = self._mk_manager()
        tensors = {
            "w0": torch.empty(4, dtype=torch.bfloat16),
            "w1": torch.empty(8, dtype=torch.bfloat16),
        }
        mgr.register_tensors(tensors)
        assert mgr._local_mem_type == _MEM_TYPE_DRAM
        assert mgr._agent.register_calls == [(_MEM_TYPE_DRAM, 2)]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_tensors_register_as_cuda(self, monkeypatch):
        monkeypatch.delenv("MX_POOL_REG", raising=False)
        mgr = self._mk_manager()
        tensors = {
            "w0": torch.empty(4, dtype=torch.bfloat16, device="cuda"),
        }
        mgr.register_tensors(tensors)
        assert mgr._local_mem_type == _MEM_TYPE_CUDA
        assert mgr._agent.register_calls == [(_MEM_TYPE_CUDA, 1)]


class TestRebindTensors:
    """``rebind_tensors`` must swap ``_tensors`` + ``_local_mem_type``
    without touching NIXL registrations. This is the safety net that
    keeps the buffer-caching path correct after mid-cycle scratch
    registrations.
    """

    def _mk_manager(self):
        mgr = NixlTransferManager(agent_name="test", device_id=0)
        mgr._agent = _FakeAgent()
        return mgr

    def test_rebind_does_not_call_register_memory(self, monkeypatch):
        monkeypatch.delenv("MX_POOL_REG", raising=False)
        mgr = self._mk_manager()

        cpu_tensors = {"w0": torch.empty(4, dtype=torch.bfloat16)}
        mgr.register_tensors(cpu_tensors)
        assert len(mgr._agent.register_calls) == 1  # one call from register_tensors

        # Rebind to a different (still-CPU) set and confirm no
        # additional register_memory call fired.
        other = {"w1": torch.empty(8, dtype=torch.bfloat16)}
        mgr.rebind_tensors(other)
        assert len(mgr._agent.register_calls) == 1
        assert mgr._local_mem_type == _MEM_TYPE_DRAM
        assert set(mgr._tensors.keys()) == {"w1"}

    def test_rebind_updates_memtype_on_device_switch(self, monkeypatch):
        """Registering CPU then rebinding to CUDA-devices flips the
        active memtype so subsequent transfers pick ``mem_type="cuda"``
        for the local side. Only exercised when CUDA is available.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        monkeypatch.delenv("MX_POOL_REG", raising=False)
        mgr = self._mk_manager()

        cpu_tensors = {"w0": torch.empty(4, dtype=torch.bfloat16)}
        mgr.register_tensors(cpu_tensors)
        assert mgr._local_mem_type == _MEM_TYPE_DRAM

        cuda_tensors = {"g0": torch.empty(4, dtype=torch.bfloat16, device="cuda")}
        mgr.rebind_tensors(cuda_tensors)
        assert mgr._local_mem_type == _MEM_TYPE_CUDA

    def test_rebind_without_agent_raises(self):
        mgr = NixlTransferManager(agent_name="test", device_id=0)
        # No _agent set — should error before touching tensors.
        with pytest.raises(RuntimeError, match="not initialized"):
            mgr.rebind_tensors({"w0": torch.empty(4, dtype=torch.bfloat16)})

    def test_rebind_empty_set_raises(self):
        mgr = NixlTransferManager(agent_name="test", device_id=0)
        mgr._agent = _FakeAgent()
        with pytest.raises(ValueError, match="empty tensor set"):
            mgr.rebind_tensors({})
