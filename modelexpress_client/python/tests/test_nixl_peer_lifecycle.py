# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Peer lifecycle: a reader must disconnect from its source before it exits.

Regression cover for a source wedged by a departing reader. A target pulled weights
from a source over P2P, then was scaled to zero. Every later target stalled for the
full transfer timeout and fell back to disk, and the source stayed wedged until it
restarted while still heartbeating READY.

The asymmetry that makes this the *reader's* job: in P2P the target sends
``NIXLCOMM:SEND`` and the source answers with its own metadata, so only the target
loads a remote agent. NIXL's ``invalidateRemoteMD`` disconnects by looking the peer
up in ``remoteBackends_``, which on the source has no entry for a reader it never
loaded. The source therefore cannot invalidate its readers, and a reader that exits
without disconnecting leaves it holding a QP nothing can clean up.

Run: pytest tests/test_nixl_peer_lifecycle.py
"""

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from modelexpress.nixl_transfer import NixlTransferManager


class FakeAgent:
    """Minimal stand-in recording the lifecycle calls we care about."""

    def __init__(self, fail_remove: bool = False):
        self.removed: list[str] = []
        self.deregistered: list[object] = []
        self.fail_remove = fail_remove
        self._fetched: set[str] = set()

    def fetch_remote_metadata(self, name, ip, port):
        self._fetched.add(name)

    def check_remote_metadata(self, name):
        return name in self._fetched

    def add_remote_agent(self, metadata: bytes) -> str:
        return "src-from-blob"

    def remove_remote_agent(self, name: str):
        if self.fail_remove:
            raise RuntimeError(f"remote metadata for agent '{name}' not found")
        self.removed.append(name)

    def deregister_memory(self, registered):
        self.deregistered.append(registered)


def _manager(agent=None, metadata=b"md", accelerator=None):
    mgr = NixlTransferManager(
        agent_name="tgt", device_id=0, accelerator_backend=accelerator
    )
    mgr._agent = agent if agent is not None else FakeAgent()
    mgr._metadata = metadata
    return mgr


class TestTracksLoadedPeers:
    def test_a_fetched_source_is_tracked_with_its_endpoint(self):
        """The endpoint is retained because it identifies the peer we connected to."""
        mgr = _manager()
        mgr.fetch_remote_and_wait("src-a", "10.0.18.37", 5555)
        assert mgr._remote_agents == {"src-a": ("10.0.18.37", 5555)}

    def test_a_blob_loaded_source_is_tracked_without_an_endpoint(self):
        """Centralized mode has no socket endpoint, but still needs disconnecting."""
        mgr = _manager()
        name = mgr.add_remote_agent(b"some-metadata")
        assert mgr._remote_agents == {name: None}

    def test_centralized_receive_tracks_peer_for_shutdown(
        self, mock_accelerator_backend_cls
    ):
        """The receive wrapper must not bypass the manager-owned inventory.

        Takes the mock accelerator because ``receive_from_source`` selects a
        device before transferring, which the real CUDA backend cannot do on a
        CPU-only runner.
        """
        agent = FakeAgent()
        mgr = _manager(agent=agent, accelerator=mock_accelerator_backend_cls())

        mgr.receive_from_source(b"some-metadata", [])
        mgr.shutdown()

        assert agent.removed == ["src-from-blob"]

    def test_a_failed_fetch_is_not_tracked(self):
        """Tracking a peer we never loaded would make shutdown remove a stranger."""

        class NeverReady(FakeAgent):
            def check_remote_metadata(self, name):
                return False

        mgr = _manager(agent=NeverReady())
        with pytest.raises(TimeoutError):
            mgr.fetch_remote_and_wait("src-a", "10.0.18.37", 5555, timeout_seconds=0.05)
        assert mgr._remote_agents == {}


class TestDisconnect:
    def test_manager_registers_process_exit_teardown_once(self):
        mgr = _manager()
        with (
            patch("modelexpress.nixl_transfer.atexit.register") as register,
            patch("modelexpress.nixl_transfer.atexit.unregister") as unregister,
        ):
            mgr._register_atexit()
            mgr._register_atexit()
            register.assert_called_once_with(mgr.shutdown)

            mgr.shutdown()
            unregister.assert_called_once_with(mgr.shutdown)

    def test_shutdown_disconnects_the_source_it_pulled_from(self):
        """The bug: this is what a departing target never did."""
        agent = FakeAgent()
        mgr = _manager(agent=agent)
        mgr.fetch_remote_and_wait("src-a", "10.0.18.37", 5555)

        mgr.shutdown()

        assert agent.removed == ["src-a"], "reader exited without disconnecting"
        assert mgr._remote_agents == {}

    def test_shutdown_disconnects_every_peer(self):
        agent = FakeAgent()
        mgr = _manager(agent=agent)
        mgr.fetch_remote_and_wait("src-a", "10.0.18.37", 5555)
        mgr.fetch_remote_and_wait("src-b", "10.0.18.38", 5555)

        mgr.shutdown()

        assert sorted(agent.removed) == ["src-a", "src-b"]

    def test_disconnect_happens_before_the_agent_is_dropped(self):
        """Ordering is the whole point: once ``_agent`` is None there is no handle
        left to disconnect through, and the peer is stranded."""
        agent = FakeAgent()
        mgr = _manager(agent=agent)
        mgr.fetch_remote_and_wait("src-a", "10.0.18.37", 5555)

        seen = {}
        real_remove = agent.remove_remote_agent

        def spy(name):
            seen["agent_alive"] = mgr._agent is not None
            return real_remove(name)

        agent.remove_remote_agent = spy
        mgr.shutdown()

        assert seen["agent_alive"] is True
        assert mgr._agent is None

    def test_registered_memory_is_released_before_agent_is_dropped(self):
        agent = FakeAgent()
        mgr = _manager(agent=agent)
        first, second = object(), object()
        mgr._registered_memory = [first, second]
        seen = []
        real_deregister = agent.deregister_memory

        def spy(registered):
            seen.append((registered, mgr._agent is agent))
            real_deregister(registered)

        agent.deregister_memory = spy
        mgr.shutdown()

        assert seen == [(second, True), (first, True)]
        assert agent.deregistered == [second, first]
        assert mgr._registered_memory == []
        assert mgr._agent is None

    def test_removing_a_peer_twice_is_harmless(self):
        """The peer may already be gone, e.g. it sent us NIXLCOMM:INVL on exit."""
        agent = FakeAgent()
        mgr = _manager(agent=agent)
        mgr.fetch_remote_and_wait("src-a", "10.0.18.37", 5555)

        assert mgr.remove_remote_agent("src-a") is True
        assert mgr.remove_remote_agent("src-a") is True  # NIXL no-ops on NOT_FOUND
        assert agent.removed == ["src-a", "src-a"]

    def test_a_failing_disconnect_does_not_break_shutdown(self):
        """Teardown runs on paths where something has usually already failed;
        raising here would turn a clean exit into a crash."""
        agent = FakeAgent(fail_remove=True)
        mgr = _manager(agent=agent)
        mgr.fetch_remote_and_wait("src-a", "10.0.18.37", 5555)

        assert mgr.remove_remote_agent("src-a") is False
        mgr.shutdown()  # must not raise
        assert mgr._agent is None
        assert mgr._remote_agents == {}

    def test_disconnect_without_an_agent_is_a_no_op(self):
        mgr = NixlTransferManager(agent_name="tgt", device_id=0)
        assert mgr.disconnect_remote_agents() == 0
        assert mgr.remove_remote_agent("anything") is False

    def test_disconnect_reports_how_many_peers_it_closed(self):
        agent = FakeAgent()
        mgr = _manager(agent=agent)
        mgr.fetch_remote_and_wait("src-a", "10.0.18.37", 5555)
        mgr.fetch_remote_and_wait("src-b", "10.0.18.38", 5555)
        assert mgr.disconnect_remote_agents() == 2
        assert mgr.disconnect_remote_agents() == 0


class TestHealthReflectsDataPlane:
    """A structural health check stays true through a dead data plane, which is how
    a broken worker keeps heartbeating READY and keeps being selected."""

    def test_a_fresh_manager_with_metadata_is_healthy(self):
        assert _manager().is_healthy() is True

    def test_no_agent_is_unhealthy(self):
        mgr = _manager()
        mgr._agent = None
        assert mgr.is_healthy() is False

    def test_no_metadata_is_unhealthy(self):
        assert _manager(metadata=b"").is_healthy() is False

    def test_a_transfer_timeout_makes_the_agent_unhealthy(self):
        """A wedged QP yields neither a completion nor an ERR status, so the
        timeout is the only evidence that the data plane is broken."""

        class Stuck:
            def check_xfer_state(self, handle):
                return "PROC"

        mgr = _manager(agent=Stuck())
        with pytest.raises(TimeoutError):
            mgr._wait_for_xfer(object(), 0.02, "Transfer")

        assert mgr.is_healthy() is False
        assert "timed out" in (mgr.data_plane_error or "")

    def test_an_error_status_makes_the_agent_unhealthy(self):
        class Failing:
            def check_xfer_state(self, handle):
                return "ERR"

        mgr = _manager(agent=Failing())
        with pytest.raises(RuntimeError):
            mgr._wait_for_xfer(object(), 5.0, "Transfer")

        assert mgr.is_healthy() is False
        assert "ERR" in (mgr.data_plane_error or "")

    def test_a_completed_transfer_leaves_health_untouched(self):
        class Done:
            def check_xfer_state(self, handle):
                return "DONE"

        mgr = _manager(agent=Done())
        mgr._wait_for_xfer(object(), 5.0, "Transfer")
        assert mgr.is_healthy() is True
        assert mgr.data_plane_error is None

    def test_a_later_success_clears_an_earlier_failure(self):
        """Health must not latch. A completed transfer proves the data plane works,
        so a worker demoted for one transient timeout has to be able to return to
        READY - otherwise the publisher's recovery path can never fire in
        production and one blip sidelines a worker for the life of the process."""

        class Flaky:
            def __init__(self):
                self.state = "PROC"

            def check_xfer_state(self, handle):
                return self.state

        agent = Flaky()
        mgr = _manager(agent=agent)
        with pytest.raises(TimeoutError):
            mgr._wait_for_xfer(object(), 0.02, "Transfer")
        assert mgr.is_healthy() is False

        agent.state = "DONE"
        mgr._wait_for_xfer(object(), 5.0, "Transfer")

        assert mgr.is_healthy() is True
        assert mgr.data_plane_error is None

    def test_health_does_not_claim_to_catch_a_passive_source(self):
        """Documents the limit deliberately. A pure P2P source issues no transfers -
        the reader's one-sided READ is invisible to it - so a source wedged by a
        departed peer has nothing to record and reports healthy. Catching that needs
        an active probe or reader-side reporting, and this test exists so nobody
        assumes otherwise from the fix."""
        source = _manager()
        assert source.data_plane_error is None
        assert source.is_healthy() is True


class TestHealthReflectsTheBatchDataPlane:
    """The same contract on the batched wait.

    ``_wait_for_xfer`` records data-plane failures; ``_wait_for_xfers`` did not, and
    it is the only wait a reshard refit performs. So the health signal that demotes a
    worker existed on the classic load path and was absent on the refit path, which
    is the path that pulls whole checkpoints across the fabric.
    """

    def test_a_batch_timeout_makes_the_agent_unhealthy(self):
        class Stuck:
            def check_xfer_state(self, handle):
                return "PROC"

        mgr = _manager(agent=Stuck())
        with pytest.raises(TimeoutError):
            mgr._wait_for_xfers([object(), object()], 0.02, "READ batch")

        assert mgr.is_healthy() is False
        assert "timed out" in (mgr.data_plane_error or "")

    def test_a_batch_error_status_makes_the_agent_unhealthy(self):
        class Failing:
            def check_xfer_state(self, handle):
                return "ERR"

        mgr = _manager(agent=Failing())
        with pytest.raises(RuntimeError):
            mgr._wait_for_xfers([object()], 5.0, "READ batch")

        assert mgr.is_healthy() is False
        assert "ERR" in (mgr.data_plane_error or "")

    def test_a_completed_batch_leaves_health_untouched(self):
        class Done:
            def check_xfer_state(self, handle):
                return "DONE"

        mgr = _manager(agent=Done())
        mgr._wait_for_xfers([object(), object()], 5.0, "READ batch")

        assert mgr.is_healthy() is True
        assert mgr.data_plane_error is None

    def test_a_later_successful_batch_clears_an_earlier_failure(self):
        """Health must not latch on the batch path either."""

        class Flaky:
            def __init__(self):
                self.state = "PROC"

            def check_xfer_state(self, handle):
                return self.state

        agent = Flaky()
        mgr = _manager(agent=agent)
        with pytest.raises(TimeoutError):
            mgr._wait_for_xfers([object()], 0.02, "READ batch")
        assert mgr.is_healthy() is False

        agent.state = "DONE"
        mgr._wait_for_xfers([object()], 5.0, "READ batch")

        assert mgr.is_healthy() is True
        assert mgr.data_plane_error is None

    def test_one_failed_handle_in_a_batch_is_not_cleared_by_its_siblings(self):
        """The reason health is cleared once for the set rather than per completed
        handle. A batch where most handles succeed and one fails is a failed batch."""

        class OneBad:
            def __init__(self):
                self.seen = 0

            def check_xfer_state(self, handle):
                self.seen += 1
                return "ERR" if self.seen == 3 else "DONE"

        mgr = _manager(agent=OneBad())
        with pytest.raises(RuntimeError):
            mgr._wait_for_xfers([object(), object(), object()], 5.0, "READ batch")

        assert mgr.is_healthy() is False

    def test_waiting_on_nothing_does_not_clear_an_earlier_failure(self):
        """An empty set proves nothing about the fabric, so it must not restore
        health. ``await_read_batches`` returns early today, so this pins the helper
        rather than that caller."""
        mgr = _manager()
        mgr._data_plane_error = "an earlier timeout"

        mgr._wait_for_xfers([], 5.0, "READ batch")

        assert mgr.is_healthy() is False
        assert mgr.data_plane_error == "an earlier timeout"

    def test_the_refit_wait_records_a_failure_through_the_public_entry_point(self):
        """The gap reached health through `await_read_batches`, so cover it there and
        not only on the private helper."""

        class Stuck:
            def check_xfer_state(self, handle):
                return "PROC"

            def release_xfer_handle(self, handle):
                pass

        mgr = _manager(agent=Stuck())
        posted = SimpleNamespace(
            handle=object(), total_bytes=1, num_ranges=1, posted_at=time.perf_counter()
        )

        with pytest.raises(TimeoutError):
            mgr.await_read_batches([posted], timeout_seconds=0.02)

        assert mgr.is_healthy() is False
        assert "timed out" in (mgr.data_plane_error or "")
