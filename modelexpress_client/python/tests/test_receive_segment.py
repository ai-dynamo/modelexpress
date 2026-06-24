# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :meth:`MxRefitReceiver.receive_segment` and
:meth:`MxRefitReceiver.prefetch_source` (the Gen 3 rank-to-rank fast path).

NIXL itself isn't available in the unit-test environment, so these tests
mock the agent + client surfaces and verify the receiver wires the right
parameters into NIXL primitives. Cluster-level validation lives in
``benchmarks/bench_verl_rank_to_rank.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# pytest will skip this whole module if torch isn't available (it is in
# the cluster image; on the dev box torch is present too).
torch = pytest.importorskip("torch")

# The MxRefitReceiver import triggers an import of nixl_transfer which in
# turn imports modelexpress.p2p_pb2 (grpc generated). We mock the whole
# nixl_transfer module so the test runs even where nixl/grpc aren't
# installed cleanly.
import sys
from types import ModuleType


def _install_nixl_stub() -> None:
    if "modelexpress.nixl_transfer" in sys.modules:
        return
    stub = ModuleType("modelexpress.nixl_transfer")

    class _NixlTransferManager:
        def __init__(self, *a, **kw): self._agent = MagicMock()
        def initialize(self): pass
        def register_tensors(self, *a, **kw): return b""
        def shutdown(self): pass

    def _is_nixl_available(): return True

    stub.NixlTransferManager = _NixlTransferManager
    stub.is_nixl_available = _is_nixl_available
    sys.modules["modelexpress.nixl_transfer"] = stub


_install_nixl_stub()

from modelexpress.refit_receiver import MxRefitReceiver  # noqa: E402


@pytest.fixture
def receiver():
    """A receiver with a fully-mocked NIXL agent + MxClient."""
    rx = MxRefitReceiver(agent_name="test-rx", device_id=0)
    # Skip the real initialize() path; install mocks directly.
    rx._nixl = MagicMock()
    rx._nixl._agent = MagicMock()
    rx._nixl._agent.prep_xfer_dlist.side_effect = lambda **kw: f"prepped:{kw['agent_name']}"
    rx._nixl._agent.make_prepped_xfer.return_value = "handle-xfer-1"
    rx._nixl._agent.check_xfer_state.return_value = "DONE"
    rx._client = MagicMock()
    rx._initialized = True
    # Avoid touching real CUDA in unit tests.
    return rx


# ---------------------------------------------------------------------------
# prefetch_source
# ---------------------------------------------------------------------------


def test_prefetch_source_caches_remote_agent(receiver):
    """A second call for the same (source_id, worker_id) should reuse cache."""
    meta = MagicMock()
    meta.found = True
    meta.worker.nixl_metadata = b"BLOBBLOB"
    receiver._client.get_metadata.return_value = meta
    receiver._nixl._agent.add_remote_agent.return_value = "remote-agent-abc"

    with patch("torch.cuda.set_device"):
        a1 = receiver.prefetch_source("src-1", "worker-1")
        a2 = receiver.prefetch_source("src-1", "worker-1")

    assert a1 == "remote-agent-abc" == a2
    receiver._client.get_metadata.assert_called_once()
    receiver._nixl._agent.add_remote_agent.assert_called_once_with(b"BLOBBLOB")


def test_prefetch_source_raises_when_source_not_found(receiver):
    meta = MagicMock()
    meta.found = False
    receiver._client.get_metadata.return_value = meta
    with pytest.raises(RuntimeError, match="not on MX server"):
        receiver.prefetch_source("src-missing", "worker-x")


def test_prefetch_source_separate_sources_separate_cache(receiver):
    """Different (source_id, worker_id) keys must trigger separate metadata loads."""
    meta1 = MagicMock(); meta1.found = True; meta1.worker.nixl_metadata = b"A"
    meta2 = MagicMock(); meta2.found = True; meta2.worker.nixl_metadata = b"B"
    receiver._client.get_metadata.side_effect = [meta1, meta2]
    receiver._nixl._agent.add_remote_agent.side_effect = ["agent-A", "agent-B"]

    with patch("torch.cuda.set_device"):
        a1 = receiver.prefetch_source("src-1", "worker-1")
        a2 = receiver.prefetch_source("src-2", "worker-2")

    assert a1 == "agent-A" and a2 == "agent-B"
    assert receiver._client.get_metadata.call_count == 2


# ---------------------------------------------------------------------------
# receive_segment
# ---------------------------------------------------------------------------


def test_receive_segment_calls_nixl_with_byte_descriptors(receiver):
    """Verify the (addr, byte_count, device_id) tuples we hand to NIXL."""
    with patch("torch.cuda.set_device"), patch("torch.cuda.synchronize"):
        elapsed = receiver.receive_segment(
            remote_agent_name="agent-X",
            source_addr=0x10000,
            byte_count=4096,
            target_addr=0x90000,
            source_device_id=3,
        )
    assert elapsed >= 0.0

    # Source side
    src_call = receiver._nixl._agent.prep_xfer_dlist.call_args_list[0]
    assert src_call.kwargs["agent_name"] == "agent-X"
    assert src_call.kwargs["xfer_list"] == [(0x10000, 4096, 3)]
    assert src_call.kwargs["mem_type"] == "cuda"
    assert src_call.kwargs["backends"] == ["UCX"]

    # Local side
    dst_call = receiver._nixl._agent.prep_xfer_dlist.call_args_list[1]
    assert dst_call.kwargs["agent_name"] == ""
    assert dst_call.kwargs["xfer_list"] == [(0x90000, 4096, 0)]  # device_id=0


def test_receive_segment_issues_single_read(receiver):
    with patch("torch.cuda.set_device"), patch("torch.cuda.synchronize"):
        receiver.receive_segment(
            remote_agent_name="agent-X",
            source_addr=0x1000, byte_count=128, target_addr=0x9000,
        )
    receiver._nixl._agent.make_prepped_xfer.assert_called_once()
    call = receiver._nixl._agent.make_prepped_xfer.call_args
    assert call.kwargs["operation"] == "READ"
    assert call.kwargs["local_indices"] == [0]
    assert call.kwargs["remote_indices"] == [0]
    assert call.kwargs["backends"] == ["UCX"]


def test_receive_segment_releases_handle_on_success(receiver):
    with patch("torch.cuda.set_device"), patch("torch.cuda.synchronize"):
        receiver.receive_segment(
            remote_agent_name="agent-X",
            source_addr=0x1000, byte_count=128, target_addr=0x9000,
        )
    receiver._nixl._agent.release_xfer_handle.assert_called_once_with("handle-xfer-1")


def test_receive_segment_raises_on_nixl_error(receiver):
    receiver._nixl._agent.check_xfer_state.return_value = "ERR"
    with patch("torch.cuda.set_device"), patch("torch.cuda.synchronize"):
        with pytest.raises(RuntimeError, match="transfer failed status=ERR"):
            receiver.receive_segment(
                remote_agent_name="agent-X",
                source_addr=0x1000, byte_count=128, target_addr=0x9000,
            )
    receiver._nixl._agent.release_xfer_handle.assert_called_once()


def test_receive_segment_timeout(receiver):
    """If NIXL never reports DONE, we should raise TimeoutError."""
    receiver._nixl._agent.check_xfer_state.return_value = "IN_PROGRESS"
    with patch("torch.cuda.set_device"), patch("torch.cuda.synchronize"):
        with pytest.raises(TimeoutError, match="receive_segment"):
            receiver.receive_segment(
                remote_agent_name="agent-X",
                source_addr=0x1000, byte_count=128, target_addr=0x9000,
                timeout_seconds=0.05,
            )
    receiver._nixl._agent.release_xfer_handle.assert_called_once()


def test_receive_segment_requires_initialize():
    rx = MxRefitReceiver(agent_name="test-rx-2", device_id=0)
    with pytest.raises(RuntimeError, match="Call initialize"):
        rx.receive_segment(
            remote_agent_name="x", source_addr=0, byte_count=1, target_addr=0,
        )
