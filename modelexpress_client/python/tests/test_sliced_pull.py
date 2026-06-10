# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the v1 sliced-pull NIXL primitive.

The primitive is :class:`NixlTransferManager.receive_sliced_from_source`,
which issues a single NIXL transfer with N (source-slice, dest-view)
descriptor pairs — the bandwidth-optimal mixed-TP data plane.

These tests stub the NixlAgent so they run without GPUs or RDMA fabric:
the stub records the `(addr, size, device_id)` tuples passed to
`prep_xfer_dlist` and simulates a successful transfer by host-side
copying source bytes into dest bytes. That's enough to validate:

  * source_offset_bytes + slice_bytes math
  * dest view byte-equivalence (contiguous narrow IS the right pointer)
  * batched descriptors (N slices → 1 transfer, N descriptor pairs)
  * end-to-end byte-identity for both target-narrower and target-wider
    direction shapes (column-parallel axis-0 narrows)
"""

from __future__ import annotations

import ctypes
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch


_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent / "modelexpress"


def _load(modname: str, path: Path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_nixl_module():
    """Build a fake `nixl._api` so importing nixl_transfer succeeds.

    The real module imports `nixl_agent` + `nixl_agent_config` from `nixl._api`.
    We replace those with stubs that record calls; the test then replaces
    `_agent` on the constructed manager with a custom mock that
    simulates the transfer.
    """
    if "nixl" in sys.modules:
        return
    nixl_pkg = types.ModuleType("nixl")
    nixl_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules["nixl"] = nixl_pkg
    api_mod = types.ModuleType("nixl._api")
    class _FakeAgent:
        def __init__(self, *a, **k): pass
    api_mod.nixl_agent = _FakeAgent
    api_mod.nixl_agent_config = lambda *a, **k: None
    sys.modules["nixl._api"] = api_mod
    nixl_pkg._api = api_mod  # type: ignore[attr-defined]


@pytest.fixture(scope="module")
def env():
    """Load nixl_transfer with the real torch + a fake nixl stub.

    These tests need a clean modelexpress package namespace — the other
    test files in this directory stub `modelexpress.*` for their own
    purposes, so we forcibly reset to our minimal stub set here.
    """
    _fake_nixl_module()

    # Force-reset our needed modules (other test files may have stubbed
    # them already with module-scoped fixtures).
    for k in [
        "modelexpress", "modelexpress.types", "modelexpress.ucx_utils",
        "modelexpress.nixl_transfer",
    ]:
        sys.modules.pop(k, None)

    # ucx_utils stub
    ucx = types.ModuleType("modelexpress.ucx_utils")
    ucx.get_ucx_overrides = lambda: {}
    sys.modules["modelexpress.ucx_utils"] = ucx

    # modelexpress.types stub for TensorDescriptor + ManifestMismatchError
    types_mod = types.ModuleType("modelexpress.types")
    @dataclass
    class _TD:
        name: str
        addr: int
        size: int
        device_id: int
        dtype: str
    class _MismatchError(Exception):
        pass
    types_mod.TensorDescriptor = _TD
    types_mod.ManifestMismatchError = _MismatchError
    sys.modules["modelexpress.types"] = types_mod

    # Avoid the real package __init__
    pkg = types.ModuleType("modelexpress")
    pkg.__path__ = [str(_PKG_ROOT)]
    sys.modules["modelexpress"] = pkg

    nxl = _load("modelexpress.nixl_transfer", _PKG_ROOT / "nixl_transfer.py")
    return types.SimpleNamespace(nxl=nxl, TensorDescriptor=_TD)


def _make_simulating_agent(env, source_buffers: dict[str, torch.Tensor],
                            source_base_addrs: dict[str, int]):
    """Build a fake NixlAgent that simulates an RDMA pull via host memcpy.

    When ``make_prepped_xfer`` is called, the captured remote_descs +
    local_descs are paired by index and the source bytes are copied
    into the dest memory using ``ctypes.memmove``. Tests inspect
    ``recorded_remote_descs`` / ``recorded_local_descs`` to assert the
    descriptor math is right.

    ``source_base_addrs[name]`` is the synthetic "remote addr" each
    source tensor will be addressed by; the simulator uses it to find
    the right backing buffer.
    """
    agent = MagicMock()
    recorded = types.SimpleNamespace(
        remote_descs=[], local_descs=[], xfer_calls=0,
    )

    def _prep_remote(*, agent_name, xfer_list, mem_type, backends):
        # Capture and pass through; return the list as the "handle".
        recorded.remote_descs.extend(xfer_list)
        return list(xfer_list)

    def _prep_local(*, agent_name, xfer_list, mem_type, backends):
        recorded.local_descs.extend(xfer_list)
        return list(xfer_list)

    def _make_xfer(*, operation, local_xfer_side, local_indices,
                   remote_xfer_side, remote_indices, backends):
        recorded.xfer_calls += 1
        # Simulate: for each pair, copy bytes from source backing buffer
        # into dest memory.
        for i, j in zip(local_indices, remote_indices):
            local_addr, local_size, _ = local_xfer_side[i]
            remote_addr, remote_size, _ = remote_xfer_side[j]
            assert local_size == remote_size
            # Resolve remote_addr → source_buffer + offset
            src_name = None
            offset = 0
            for n, base in source_base_addrs.items():
                if base <= remote_addr < base + source_buffers[n].numel() * source_buffers[n].element_size():
                    src_name = n
                    offset = remote_addr - base
                    break
            if src_name is None:
                raise AssertionError(
                    f"simulator: remote_addr={remote_addr} not in any source"
                )
            src = source_buffers[src_name]
            src_ptr = src.data_ptr() + offset
            ctypes.memmove(local_addr, src_ptr, local_size)
        return "h"

    def _check(handle):
        return "DONE"

    def _release(handle):
        return None

    def _transfer(handle):
        return None

    def _add_remote_agent(metadata):
        return "remote-agent"

    agent.prep_xfer_dlist = MagicMock(side_effect=[_prep_remote.__wrapped__ if hasattr(_prep_remote, '__wrapped__') else None])
    # MagicMock side_effect needs to handle both calls; use a counter to alternate.
    call_count = {"n": 0}
    def _prep(*, agent_name, xfer_list, mem_type, backends):
        call_count["n"] += 1
        if call_count["n"] % 2 == 1:
            return _prep_remote(agent_name=agent_name, xfer_list=xfer_list,
                                mem_type=mem_type, backends=backends)
        return _prep_local(agent_name=agent_name, xfer_list=xfer_list,
                           mem_type=mem_type, backends=backends)
    agent.prep_xfer_dlist = _prep
    agent.make_prepped_xfer = _make_xfer
    agent.check_xfer_state = _check
    agent.release_xfer_handle = _release
    agent.transfer = _transfer
    agent.add_remote_agent = _add_remote_agent
    return agent, recorded


def _make_manager(env, agent):
    """Build a NixlTransferManager wired to the simulating agent."""
    # Force CPU device_id so torch.cuda.set_device + synchronize are no-ops in tests.
    mgr = env.nxl.NixlTransferManager.__new__(env.nxl.NixlTransferManager)
    mgr._agent = agent
    mgr._device_id = 0
    mgr._backends = ("UCX",)
    mgr._agent_name = "test-agent"
    mgr._tensors = {}
    mgr._tensor_descriptors = []
    mgr._metadata = b""
    return mgr


def _patch_cuda_in_module(mod):
    """Stub torch.cuda.set_device/synchronize so the test runs on CPU."""
    real_set = torch.cuda.set_device
    real_sync = torch.cuda.synchronize
    torch.cuda.set_device = lambda *a, **k: None
    torch.cuda.synchronize = lambda *a, **k: None
    return real_set, real_sync


def _restore_cuda(saved):
    torch.cuda.set_device, torch.cuda.synchronize = saved


def test_full_pull_matches_receive_from_source(env):
    """Single request, no offset, full source size — equivalent to the
    legacy bulk transfer. Validates the simple case."""
    torch.manual_seed(0)
    src = torch.randn(64, 32, dtype=torch.float32)
    source_buffers = {"x": src}
    source_base_addrs = {"x": 0xCAFE_0000_0000}
    saved = _patch_cuda_in_module(env.nxl)
    try:
        agent, recorded = _make_simulating_agent(env, source_buffers, source_base_addrs)
        mgr = _make_manager(env, agent)
        TD = env.TensorDescriptor
        source_tensors = [TD(
            name="x", addr=source_base_addrs["x"], size=src.numel() * src.element_size(),
            device_id=0, dtype="float32",
        )]
        dest = torch.zeros_like(src)
        req = env.nxl.SlicedTransferRequest(
            name="x",
            source_offset_bytes=0,
            slice_bytes=src.numel() * src.element_size(),
            dest_view=dest,
        )
        total, n, _elapsed = mgr.receive_sliced_from_source(
            source_metadata=b"meta",
            source_tensors=source_tensors,
            slice_requests=[req],
        )
        assert n == 1
        assert total == src.numel() * src.element_size()
        assert torch.equal(dest, src)
        assert recorded.xfer_calls == 1
        assert len(recorded.remote_descs) == 1
        assert recorded.remote_descs[0][1] == src.numel() * src.element_size()
    finally:
        _restore_cuda(saved)


def test_partial_pull_target_wider_axis_0(env):
    """Target-wider column-parallel slice: pull rows [N/4, N/2) of a
    [N, hidden] source into a contiguous dest view."""
    torch.manual_seed(7)
    N, hidden = 64, 32
    src = torch.randn(N, hidden, dtype=torch.float32)
    source_buffers = {"col": src}
    source_base_addrs = {"col": 0xBEEF_0000_0000}
    saved = _patch_cuda_in_module(env.nxl)
    try:
        agent, recorded = _make_simulating_agent(env, source_buffers, source_base_addrs)
        mgr = _make_manager(env, agent)
        TD = env.TensorDescriptor
        source_tensors = [TD(
            name="col", addr=source_base_addrs["col"],
            size=src.numel() * src.element_size(), device_id=0, dtype="float32",
        )]
        # vLLM TP=4, target rank 1 wants rows [N/4, 2N/4).
        row_lo, row_hi = N // 4, N // 2
        dest = torch.zeros(row_hi - row_lo, hidden, dtype=torch.float32)
        slice_bytes = (row_hi - row_lo) * hidden * src.element_size()
        offset_bytes = row_lo * hidden * src.element_size()
        req = env.nxl.SlicedTransferRequest(
            name="col", source_offset_bytes=offset_bytes,
            slice_bytes=slice_bytes, dest_view=dest,
        )
        total, n, _ = mgr.receive_sliced_from_source(
            source_metadata=b"m", source_tensors=source_tensors, slice_requests=[req],
        )
        assert n == 1
        assert total == slice_bytes
        # Byte-identical to the source slice.
        assert torch.equal(dest, src[row_lo:row_hi])
        # Remote descriptor uses base+offset, not base.
        assert recorded.remote_descs[0][0] == source_base_addrs["col"] + offset_bytes
        assert recorded.remote_descs[0][1] == slice_bytes
        # Local descriptor points at dest.data_ptr().
        assert recorded.local_descs[0][0] == dest.data_ptr()
    finally:
        _restore_cuda(saved)


def test_batched_pulls_one_combined_transfer(env):
    """N slice requests → 1 NIXL transfer with N descriptor pairs.
    Validates the per-source batching invariant."""
    torch.manual_seed(13)
    a = torch.randn(32, 16, dtype=torch.float32)
    b = torch.randn(48, 16, dtype=torch.float32)
    c = torch.randn(8, 16, dtype=torch.float32)
    source_buffers = {"a": a, "b": b, "c": c}
    source_base_addrs = {"a": 0x1_0000_0000, "b": 0x2_0000_0000, "c": 0x3_0000_0000}
    saved = _patch_cuda_in_module(env.nxl)
    try:
        agent, recorded = _make_simulating_agent(env, source_buffers, source_base_addrs)
        mgr = _make_manager(env, agent)
        TD = env.TensorDescriptor
        source_tensors = [
            TD(name=n, addr=source_base_addrs[n],
               size=t.numel() * t.element_size(), device_id=0, dtype="float32")
            for n, t in source_buffers.items()
        ]
        # Various slices: half of a, all of b, last 4 rows of c.
        dest_a = torch.zeros(16, 16, dtype=torch.float32)
        dest_b = torch.zeros_like(b)
        dest_c = torch.zeros(4, 16, dtype=torch.float32)
        reqs = [
            env.nxl.SlicedTransferRequest(
                "a", source_offset_bytes=0,
                slice_bytes=16 * 16 * 4, dest_view=dest_a,
            ),
            env.nxl.SlicedTransferRequest(
                "b", source_offset_bytes=0,
                slice_bytes=b.numel() * b.element_size(), dest_view=dest_b,
            ),
            env.nxl.SlicedTransferRequest(
                "c", source_offset_bytes=4 * 16 * 4,
                slice_bytes=4 * 16 * 4, dest_view=dest_c,
            ),
        ]
        total, n, _ = mgr.receive_sliced_from_source(
            source_metadata=b"m", source_tensors=source_tensors, slice_requests=reqs,
        )
        assert n == 3
        assert recorded.xfer_calls == 1  # one combined transfer
        assert len(recorded.remote_descs) == 3
        assert len(recorded.local_descs) == 3
        # Correctness
        assert torch.equal(dest_a, a[:16])
        assert torch.equal(dest_b, b)
        assert torch.equal(dest_c, c[4:])
    finally:
        _restore_cuda(saved)


def test_rejects_noncontiguous_dest_view(env):
    """Row-parallel axis-1 narrow gives a non-contiguous view. The
    primitive must refuse — the caller should fall back to scratch+copy."""
    torch.manual_seed(0)
    src = torch.randn(64, 32, dtype=torch.float32)
    source_buffers = {"r": src}
    source_base_addrs = {"r": 0x4_0000_0000}
    saved = _patch_cuda_in_module(env.nxl)
    try:
        agent, _ = _make_simulating_agent(env, source_buffers, source_base_addrs)
        mgr = _make_manager(env, agent)
        TD = env.TensorDescriptor
        source_tensors = [TD(name="r", addr=source_base_addrs["r"],
                              size=src.numel() * src.element_size(),
                              device_id=0, dtype="float32")]
        dest = torch.zeros(64, 32, dtype=torch.float32)
        # axis-1 narrow → non-contiguous
        nc_view = dest.narrow(1, 0, 16)
        assert not nc_view.is_contiguous()
        req = env.nxl.SlicedTransferRequest(
            "r", source_offset_bytes=0, slice_bytes=64 * 16 * 4, dest_view=nc_view,
        )
        with pytest.raises(RuntimeError, match="non-contiguous"):
            mgr.receive_sliced_from_source(
                source_metadata=b"m", source_tensors=source_tensors,
                slice_requests=[req],
            )
    finally:
        _restore_cuda(saved)


def test_rejects_out_of_range_slice(env):
    """Slice that runs past the source's end → RuntimeError."""
    saved = _patch_cuda_in_module(env.nxl)
    try:
        agent, _ = _make_simulating_agent(
            env, {"x": torch.zeros(4, 4, dtype=torch.float32)},
            {"x": 0x100},
        )
        mgr = _make_manager(env, agent)
        TD = env.TensorDescriptor
        source_tensors = [TD(name="x", addr=0x100, size=64, device_id=0, dtype="float32")]
        dest = torch.zeros(20, dtype=torch.float32)
        req = env.nxl.SlicedTransferRequest(
            "x", source_offset_bytes=32, slice_bytes=80,  # 32 + 80 > 64
            dest_view=dest,
        )
        with pytest.raises(RuntimeError, match="past end"):
            mgr.receive_sliced_from_source(
                source_metadata=b"m", source_tensors=source_tensors,
                slice_requests=[req],
            )
    finally:
        _restore_cuda(saved)


def test_rejects_dest_size_mismatch(env):
    """dest_view size must equal slice_bytes."""
    saved = _patch_cuda_in_module(env.nxl)
    try:
        agent, _ = _make_simulating_agent(env, {"x": torch.zeros(16, dtype=torch.float32)}, {"x": 0x200})
        mgr = _make_manager(env, agent)
        TD = env.TensorDescriptor
        source_tensors = [TD(name="x", addr=0x200, size=64, device_id=0, dtype="float32")]
        dest = torch.zeros(8, dtype=torch.float32)  # 32 bytes
        req = env.nxl.SlicedTransferRequest(
            "x", source_offset_bytes=0, slice_bytes=64,  # but dest is only 32 bytes
            dest_view=dest,
        )
        with pytest.raises(RuntimeError, match="dest view size"):
            mgr.receive_sliced_from_source(
                source_metadata=b"m", source_tensors=source_tensors,
                slice_requests=[req],
            )
    finally:
        _restore_cuda(saved)


def test_unknown_tensor_name_raises(env):
    """Request for a tensor not in the source manifest → RuntimeError."""
    saved = _patch_cuda_in_module(env.nxl)
    try:
        agent, _ = _make_simulating_agent(env, {"a": torch.zeros(16, dtype=torch.float32)}, {"a": 0x300})
        mgr = _make_manager(env, agent)
        TD = env.TensorDescriptor
        source_tensors = [TD(name="a", addr=0x300, size=64, device_id=0, dtype="float32")]
        req = env.nxl.SlicedTransferRequest(
            "b", source_offset_bytes=0, slice_bytes=16, dest_view=torch.zeros(4, dtype=torch.float32),
        )
        with pytest.raises(RuntimeError, match="not in source manifest"):
            mgr.receive_sliced_from_source(
                source_metadata=b"m", source_tensors=source_tensors,
                slice_requests=[req],
            )
    finally:
        _restore_cuda(saved)


def test_empty_requests_returns_zero(env):
    """No-op: empty request list returns (0, 0, 0)."""
    saved = _patch_cuda_in_module(env.nxl)
    try:
        agent, recorded = _make_simulating_agent(env, {}, {})
        mgr = _make_manager(env, agent)
        total, n, elapsed = mgr.receive_sliced_from_source(
            source_metadata=b"m", source_tensors=[], slice_requests=[],
        )
        assert (total, n) == (0, 0)
        assert recorded.xfer_calls == 0  # no transfer issued
    finally:
        _restore_cuda(saved)
