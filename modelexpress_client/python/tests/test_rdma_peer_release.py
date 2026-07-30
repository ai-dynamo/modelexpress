# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A reader must release the source as part of the load, not at process exit.

Regression cover for nvbug 6519532, which reproduced 4/4 on 0.5.0-rc.3 *after*
the peer-disconnect machinery had already shipped. The machinery was reachable
only from an ``atexit`` hook, and the NIXL agent lives in vLLM's EngineCore
subprocess, which dies without running interpreter exit handlers - the retest
captured ``EngineDeadError`` and zero occurrences of the shutdown log line. So
the disconnect existed and never ran, the source kept a half-open QP, and every
later reader stalled for the full transfer budget and fell back to disk.

Releasing on the load path is what makes the cleanup reachable: it happens while
the process is still healthy, so it does not depend on how the pod is torn down.
It also covers SIGKILL and OOM, which no signal handler could.

Only the reader can do this. In P2P the reader drives the metadata exchange, so
only the reader loads a remote agent; the source has no peer record to
invalidate and cannot detect a graceful departure.

Run: pytest tests/test_rdma_peer_release.py
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from modelexpress.load_strategy.base import SourceTransferError
from modelexpress.load_strategy.rdma_strategy import RdmaStrategy

P2P_AGENT = "mx-auto-worker0-0d6a01a9"
BLOB_AGENT = "agent-from-blob"


def _descriptor(name="w0"):
    return SimpleNamespace(name=name, addr=4096, size=128, device_id=0, dtype="torch.bfloat16")


def _manager():
    mgr = MagicMock()
    mgr.receive_from_source.return_value = (128, 1, 0.01)
    mgr.add_remote_agent.return_value = BLOB_AGENT
    mgr.remove_remote_agent.return_value = True
    return mgr


def _ctx(manager):
    return SimpleNamespace(
        global_rank=0,
        worker_rank=0,
        worker_id="target-0",
        nixl_manager=manager,
        # Matching the source's accelerator keeps require_exact_match off, so
        # these tests exercise peer lifecycle rather than manifest strictness.
        accelerator_backend=SimpleNamespace(name="cuda", synchronize=MagicMock()),
        adapter=MagicMock(),
        identity=SimpleNamespace(model_name="m"),
    )


def _source_worker(p2p: bool):
    """A source the strategy will read from, in P2P or centralized mode."""
    return SimpleNamespace(
        agent_name=P2P_AGENT,
        # Presence of this endpoint is what selects the P2P branch.
        worker_grpc_endpoint="10.0.18.37:5556" if p2p else "",
        metadata_endpoint="10.0.18.37:5555",
        nixl_metadata=b"blob",
        accelerator="cuda",
    )


def _receive(manager, p2p=True):
    strategy = RdmaStrategy()
    ctx = _ctx(manager)
    with (
        patch("modelexpress.load_strategy.rdma_strategy.register_tensors"),
        patch(
            "modelexpress.load_strategy.rdma_strategy.worker_tensor_descriptors",
            return_value=[_descriptor()],
        ),
    ):
        strategy._receive_from_peer(MagicMock(), ctx, _source_worker(p2p), "src-1")


class TestReleaseOnTheLoadPath:
    def test_p2p_source_is_released_after_a_successful_load(self):
        """The case rc.3 disproved: cleanup must not wait for interpreter exit."""
        mgr = _manager()
        _receive(mgr, p2p=True)
        mgr.remove_remote_agent.assert_called_once_with(P2P_AGENT)

    def test_the_released_peer_is_the_one_that_was_fetched(self):
        """Releasing a different name would leave the real QP half-open."""
        mgr = _manager()
        _receive(mgr, p2p=True)
        fetched = mgr.fetch_remote_and_wait.call_args.kwargs["remote_agent_name"]
        assert mgr.remove_remote_agent.call_args.args[0] == fetched

    def test_release_happens_after_the_transfer(self):
        """Disconnecting first would tear down the QP the READ still needs."""
        mgr = _manager()
        order = []
        mgr.receive_from_source.side_effect = lambda **_: (
            order.append("transfer") or (128, 1, 0.01)
        )
        mgr.remove_remote_agent.side_effect = lambda *_: order.append("release") or True

        _receive(mgr, p2p=True)

        assert order == ["transfer", "release"]


class TestReleaseOnFailure:
    def test_a_failed_transfer_still_releases_the_source(self):
        """The wedged-source case. A reader that times out and moves to the next
        candidate must not leave connection state behind on the way out."""
        mgr = _manager()
        mgr.receive_from_source.side_effect = TimeoutError("Transfer timed out")

        with pytest.raises(SourceTransferError):
            _receive(mgr, p2p=True)

        mgr.remove_remote_agent.assert_called_once_with(P2P_AGENT)

    def test_the_transfer_error_is_not_masked_by_cleanup(self):
        """Retry and fallback both depend on the original failure propagating."""
        mgr = _manager()
        mgr.receive_from_source.side_effect = TimeoutError("Transfer timed out")
        mgr.remove_remote_agent.side_effect = RuntimeError("invalidate failed")

        with pytest.raises(RuntimeError, match="invalidate failed"):
            _receive(mgr, p2p=True)


class TestCentralizedMode:
    def test_centralized_load_releases_the_peer_it_loaded(self):
        """Centralized readers load the source from a blob, so they own the same
        teardown duty - the asymmetry that wedges the source is identical."""
        mgr = _manager()
        _receive(mgr, p2p=False)
        mgr.add_remote_agent.assert_called_once_with(b"blob")
        mgr.remove_remote_agent.assert_called_once_with(BLOB_AGENT)

    def test_centralized_load_does_not_double_load_the_peer(self):
        """The strategy loads the peer so it holds the name; passing it through
        keeps receive_from_source from loading a second, untracked copy."""
        mgr = _manager()
        _receive(mgr, p2p=False)
        assert mgr.receive_from_source.call_args.kwargs["remote_agent_name"] == BLOB_AGENT
