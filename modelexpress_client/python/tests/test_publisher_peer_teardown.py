# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publisher demotion and manager-owned teardown for a departing reader.

Two related lifecycle behaviours are covered:

1. A publisher marks only its own source STALE. The shared NIXL manager owns
   process-wide remote-peer disconnection, so routine publisher replacement does
   not invalidate peers loaded by unrelated receive or artifact paths.

2. A worker whose data plane is broken must be demoted, not merely skipped.
   Skipping the heartbeat only stops refreshing ``updated_at``, so the entry keeps
   its last status and stays selectable until the server's reaper times it out.

Run: pytest tests/test_publisher_peer_teardown.py
"""

import time
from unittest.mock import MagicMock, call, patch

import pytest

from modelexpress.metadata.publisher import PublisherThread

READY = 2
STALE = 3


@pytest.fixture
def mx_client():
    client = MagicMock()
    client.update_status.return_value = True
    return client


@pytest.fixture
def nixl_manager():
    manager = MagicMock()
    manager.is_healthy.return_value = True
    manager.disconnect_remote_agents.return_value = 1
    manager.data_plane_error = None
    return manager


@pytest.fixture
def publisher(mx_client, nixl_manager):
    with patch.dict("os.environ", {"MX_HEARTBEAT_INTERVAL_SECS": "1"}):
        pub = PublisherThread(
            mx_client=mx_client,
            mx_source_id="abc123",
            worker_id="w1",
            worker_rank=0,
            nixl_manager=nixl_manager,
        )
    yield pub
    pub.stop()


def _status_calls(mx_client, status):
    return [
        c
        for c in mx_client.update_status.call_args_list
        if c
        == call(mx_source_id="abc123", worker_id="w1", worker_rank=0, status=status)
    ]


class TestPublisherDoesNotOwnSharedPeers:
    def test_stop_leaves_manager_peers_connected(self, publisher, nixl_manager):
        """Routine publisher replacement must not tear down shared receive paths."""
        publisher.start()
        time.sleep(1.5)
        publisher.stop()
        nixl_manager.disconnect_remote_agents.assert_not_called()

    def test_atexit_leaves_manager_teardown_to_its_owner(
        self, publisher, nixl_manager
    ):
        """The manager's later atexit hook owns process-wide peer teardown."""
        publisher.start()
        time.sleep(1.5)
        publisher._on_exit()
        nixl_manager.disconnect_remote_agents.assert_not_called()

    def test_atexit_joins_before_marking_stale(self, publisher, mx_client):
        """An in-flight tick must finish before shutdown publishes STALE."""
        order = []
        publisher._thread = MagicMock()
        publisher._thread.join.side_effect = lambda **_: order.append("join")
        publisher._started = True
        mx_client.update_status.side_effect = lambda **kw: order.append(
            f"status:{kw['status']}"
        ) or True

        publisher._on_exit()

        assert order == ["join", f"status:{STALE}"]
        publisher._thread.join.assert_called_once_with(timeout=publisher._interval + 5)

    def test_a_publisher_without_a_nixl_manager_is_fine(self, mx_client):
        """Not every publisher owns a NIXL agent."""
        with patch.dict("os.environ", {"MX_HEARTBEAT_INTERVAL_SECS": "1"}):
            pub = PublisherThread(
                mx_client=mx_client,
                mx_source_id="abc123",
                worker_id="w1",
                worker_rank=0,
                nixl_manager=None,
            )
        pub.start()
        time.sleep(1.2)
        pub.stop()  # must not raise


class TestDemotesUnhealthyWorker:
    def test_an_unhealthy_worker_is_marked_stale(
        self, publisher, mx_client, nixl_manager
    ):
        """Previously the tick just returned, leaving the worker READY in the
        registry and selectable for another 90s of reaper timeout."""
        publisher.start()
        time.sleep(1.5)  # go READY first
        nixl_manager.is_healthy.return_value = False
        nixl_manager.data_plane_error = "Transfer timed out after 300.0s"
        time.sleep(1.5)

        assert _status_calls(mx_client, STALE), "unhealthy worker was not demoted"

    def test_demotion_happens_once_not_every_interval(
        self, publisher, mx_client, nixl_manager
    ):
        """A persistently broken agent should not log and demote on every tick."""
        publisher.start()
        time.sleep(1.5)
        nixl_manager.is_healthy.return_value = False
        time.sleep(2.5)
        publisher.stop()

        # One demotion from the unhealthy ticks; stop() adds its own shutdown STALE.
        assert len(_status_calls(mx_client, STALE)) <= 2

    def test_failed_demotion_is_retried(
        self, publisher, mx_client, nixl_manager
    ):
        """A transient RPC failure must not latch the local demotion state."""
        publisher._started = True
        mx_client.update_status.side_effect = [RuntimeError("temporary"), True]

        publisher._mark_unhealthy()
        assert publisher._unhealthy is False

        publisher._mark_unhealthy()
        assert publisher._unhealthy is True
        assert mx_client.update_status.call_count == 2

    def test_recovery_restores_ready(self, publisher, mx_client, nixl_manager):
        """Demotion is advisory, not terminal: a recovered agent must come back."""
        publisher.start()
        time.sleep(1.5)
        nixl_manager.is_healthy.return_value = False
        time.sleep(1.5)
        ready_before = len(_status_calls(mx_client, READY))
        nixl_manager.is_healthy.return_value = True
        time.sleep(1.5)

        assert len(_status_calls(mx_client, READY)) > ready_before

    def test_a_worker_that_never_went_ready_is_not_demoted(
        self, mx_client, nixl_manager
    ):
        """Nothing can have selected it, so STALE would be noise - and it may still
        be INITIALIZING, which STALE would misrepresent."""
        nixl_manager.is_healthy.return_value = False
        with patch.dict("os.environ", {"MX_HEARTBEAT_INTERVAL_SECS": "1"}):
            pub = PublisherThread(
                mx_client=mx_client,
                mx_source_id="abc123",
                worker_id="w1",
                worker_rank=0,
                nixl_manager=nixl_manager,
            )
        pub.start()
        time.sleep(1.5)
        pub.stop()

        assert _status_calls(mx_client, STALE) == []
