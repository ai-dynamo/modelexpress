# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for source publication and heartbeat signaling."""

import threading
import time
from unittest.mock import MagicMock, call, patch

import pytest

from modelexpress.metadata.publisher import PublisherThread


@pytest.fixture
def mx_client():
    client = MagicMock()
    client.update_status.return_value = True
    return client


@pytest.fixture
def nixl_manager():
    manager = MagicMock()
    manager.is_healthy.return_value = True
    return manager


@pytest.fixture
def heartbeat(mx_client, nixl_manager):
    with patch.dict("os.environ", {"MX_HEARTBEAT_INTERVAL_SECS": "1"}):
        hb = PublisherThread(
            mx_client=mx_client,
            mx_source_id="abc123",
            worker_id="w1",
            worker_rank=0,
            nixl_manager=nixl_manager,
        )
    yield hb
    hb.stop()


class TestHeartbeatSendsReady:
    def test_sends_ready_when_healthy(self, heartbeat, mx_client, nixl_manager):
        heartbeat.start()
        time.sleep(1.5)
        heartbeat.stop()

        calls = mx_client.update_status.call_args_list
        assert len(calls) >= 1
        assert calls[0] == call(
            mx_source_id="abc123",
            worker_id="w1",
            worker_rank=0,
            status=2,  # SOURCE_STATUS_READY
        )

    def test_skips_when_unhealthy(self, heartbeat, mx_client, nixl_manager):
        nixl_manager.is_healthy.return_value = False
        heartbeat.start()
        time.sleep(1.5)
        heartbeat.stop()

        # Only the _mark_stale call from stop(), no READY calls
        ready_calls = [
            c for c in mx_client.update_status.call_args_list
            if c == call(
                mx_source_id="abc123",
                worker_id="w1",
                worker_rank=0,
                status=2,
            )
        ]
        assert len(ready_calls) == 0

    def test_multiple_ticks_refresh_updated_at(self, heartbeat, mx_client):
        heartbeat.start()
        time.sleep(2.5)
        heartbeat.stop()

        # At 1s interval, 2.5s sleep should give at least 2 READY calls
        ready_calls = [
            c for c in mx_client.update_status.call_args_list
            if c == call(
                mx_source_id="abc123",
                worker_id="w1",
                worker_rank=0,
                status=2,
            )
        ]
        assert len(ready_calls) >= 2


class TestPublisherPublishAndReady:
    def test_first_tick_publishes_then_sends_ready(self, mx_client, nixl_manager):
        publisher = PublisherThread(
            mx_client=mx_client,
            worker_id="w1",
            worker_rank=0,
            nixl_manager=nixl_manager,
            publish_fn=lambda: "abc123",
            interval_secs=1,
        )

        publisher._tick()

        assert publisher.mx_source_id == "abc123"
        assert mx_client.update_status.call_args_list == [
            call(
                mx_source_id="abc123",
                worker_id="w1",
                worker_rank=0,
                status=2,
            )
        ]

    def test_ready_gate_blocks_publish(self, mx_client, nixl_manager):
        publish = MagicMock(return_value="abc123")
        publisher = PublisherThread(
            mx_client=mx_client,
            worker_id="w1",
            worker_rank=0,
            nixl_manager=nixl_manager,
            publish_fn=publish,
            ready_fn=lambda: False,
            interval_secs=1,
        )

        publisher._tick()

        publish.assert_not_called()
        mx_client.update_status.assert_not_called()

    def test_ready_gate_is_polled_before_heartbeat_interval(self, mx_client, nixl_manager):
        publisher = PublisherThread(
            mx_client=mx_client,
            worker_id="w1",
            worker_rank=0,
            nixl_manager=nixl_manager,
            publish_fn=MagicMock(return_value="abc123"),
            ready_fn=lambda: False,
            interval_secs=30,
        )
        waits = []

        def stop_after_wait(timeout):
            waits.append(timeout)
            publisher._stop_event.set()

        with patch.object(publisher._stop_event, "wait", side_effect=stop_after_wait):
            publisher._run()

        assert waits == [5]

    def test_can_stop_after_publish_without_heartbeat(self, mx_client, nixl_manager):
        ready = MagicMock(return_value=True)
        publish = MagicMock(return_value="abc123")
        publisher = PublisherThread(
            mx_client=mx_client,
            worker_id="w1",
            worker_rank=0,
            nixl_manager=nixl_manager,
            publish_fn=publish,
            ready_fn=ready,
            heartbeat_after_publish=False,
            interval_secs=1,
        )

        publisher._tick()
        publisher._tick()

        assert publisher.mx_source_id == "abc123"
        mx_client.update_status.assert_not_called()
        assert publisher._stop_event.is_set()
        ready.assert_called_once_with()
        publish.assert_called_once_with()


class TestHeartbeatStop:
    def test_stop_marks_stale(self, heartbeat, mx_client):
        heartbeat.start()
        time.sleep(1.5)  # Let at least one READY go through
        heartbeat.stop()

        stale_calls = [
            c for c in mx_client.update_status.call_args_list
            if c == call(
                mx_source_id="abc123",
                worker_id="w1",
                worker_rank=0,
                status=3,  # SOURCE_STATUS_STALE
            )
        ]
        assert len(stale_calls) == 1

    def test_stop_without_ready_skips_stale(self, heartbeat, mx_client, nixl_manager):
        nixl_manager.is_healthy.return_value = False
        heartbeat.start()
        time.sleep(1.5)
        heartbeat.stop()

        # Never became READY, so _mark_stale should not send STALE
        stale_calls = [
            c for c in mx_client.update_status.call_args_list
            if c == call(
                mx_source_id="abc123",
                worker_id="w1",
                worker_rank=0,
                status=3,
            )
        ]
        assert len(stale_calls) == 0

    def test_stop_is_idempotent(self, heartbeat, mx_client):
        heartbeat.start()
        time.sleep(1.5)
        heartbeat.stop()
        heartbeat.stop()  # Second stop should not send another STALE

        stale_calls = [
            c for c in mx_client.update_status.call_args_list
            if c == call(
                mx_source_id="abc123",
                worker_id="w1",
                worker_rank=0,
                status=3,
            )
        ]
        assert len(stale_calls) == 1


class TestHeartbeatOnExit:
    def test_on_exit_marks_stale(self, heartbeat, mx_client):
        heartbeat.start()
        time.sleep(1.5)
        heartbeat._on_exit()

        stale_calls = [
            c for c in mx_client.update_status.call_args_list
            if c == call(
                mx_source_id="abc123",
                worker_id="w1",
                worker_rank=0,
                status=3,
            )
        ]
        assert len(stale_calls) == 1

    def test_on_exit_swallows_errors(self, heartbeat, mx_client):
        heartbeat.start()
        time.sleep(1.5)
        mx_client.update_status.side_effect = RuntimeError("connection lost")
        heartbeat._on_exit()  # Should not raise

    def test_on_exit_orders_stale_after_inflight_ready(self, heartbeat, mx_client):
        ready_started = threading.Event()
        release_ready = threading.Event()

        def update_status(**kwargs):
            if kwargs["status"] == 2:
                ready_started.set()
                release_ready.wait(timeout=5)
            return True

        mx_client.update_status.side_effect = update_status
        heartbeat.start()
        assert ready_started.wait(timeout=5)

        exit_thread = threading.Thread(target=heartbeat._on_exit)
        exit_thread.start()
        assert heartbeat._stop_event.wait(timeout=5)
        release_ready.set()
        exit_thread.join(timeout=5)

        assert not exit_thread.is_alive()
        statuses = [call.kwargs["status"] for call in mx_client.update_status.call_args_list]
        assert statuses == [2, 3]


class TestHeartbeatReRegistration:
    """Recovery when the server rejects heartbeats (e.g. its store was reset)."""

    def _publisher(self, mx_client, nixl_manager, publish):
        return PublisherThread(
            mx_client=mx_client,
            worker_id="w1",
            worker_rank=0,
            nixl_manager=nixl_manager,
            publish_fn=publish,
            interval_secs=1,
        )

    def test_consecutive_rejections_trigger_republish(self, mx_client, nixl_manager):
        publish = MagicMock(side_effect=["id1", "id2"])
        publisher = self._publisher(mx_client, nixl_manager, publish)
        # Publish succeeds, then the server loses the registration: two
        # rejected heartbeats, then the re-published source heartbeats fine.
        mx_client.update_status.side_effect = [True, False, False, True]

        publisher._tick()  # publish id1 + READY accepted
        publisher._tick()  # rejected (1/2)
        assert publisher.mx_source_id == "id1"
        publisher._tick()  # rejected (2/2) -> re-registration scheduled
        assert publisher.mx_source_id is None
        publisher._tick()  # re-publish id2 + READY accepted

        assert publish.call_count == 2
        assert publisher.mx_source_id == "id2"
        ready_ids = [
            c.kwargs["mx_source_id"]
            for c in mx_client.update_status.call_args_list
            if c.kwargs["status"] == 2
        ]
        assert ready_ids == ["id1", "id1", "id1", "id2"]

    def test_single_rejection_does_not_republish(self, mx_client, nixl_manager):
        publish = MagicMock(return_value="id1")
        publisher = self._publisher(mx_client, nixl_manager, publish)
        # Rejections never happen twice in a row.
        mx_client.update_status.side_effect = [True, False, True, False, True]

        for _ in range(5):
            publisher._tick()

        publish.assert_called_once_with()
        assert publisher.mx_source_id == "id1"

    def test_accepted_heartbeat_resets_rejection_count(self, mx_client, nixl_manager):
        publish = MagicMock(return_value="id1")
        publisher = self._publisher(mx_client, nixl_manager, publish)
        mx_client.update_status.side_effect = [True, False, True, False, False]

        for _ in range(4):
            publisher._tick()
        assert publisher.mx_source_id == "id1"  # count was reset by tick 3
        publisher._tick()  # second consecutive rejection -> re-register

        assert publisher.mx_source_id is None

    def test_republish_resets_publish_timeout_window(self, mx_client, nixl_manager):
        publish = MagicMock(side_effect=["id1", "id2"])
        publisher = self._publisher(mx_client, nixl_manager, publish)
        mx_client.update_status.side_effect = [True, False, False, True]
        # Pretend the initial publish already used up its timeout.
        publisher._tick()
        publisher._publish_started_at = time.monotonic() - 10_000
        publisher._publish_given_up = True

        publisher._tick()
        publisher._tick()  # triggers re-registration
        assert publisher._publish_started_at is None
        assert publisher._publish_given_up is False
        publisher._tick()

        assert publisher.mx_source_id == "id2"
        assert not publisher._stop_event.is_set()

    def test_rejections_without_publish_fn_warn_once_and_keep_heartbeating(
        self, mx_client, nixl_manager, caplog
    ):
        publisher = PublisherThread(
            mx_client=mx_client,
            mx_source_id="abc123",
            worker_id="w1",
            worker_rank=0,
            nixl_manager=nixl_manager,
            interval_secs=1,
        )
        mx_client.update_status.return_value = False

        with caplog.at_level("WARNING", logger="modelexpress.metadata.publisher"):
            for _ in range(6):
                publisher._tick()

        # Cannot recover, but keeps trying and stays on the cached id.
        assert publisher.mx_source_id == "abc123"
        assert mx_client.update_status.call_count == 6
        warnings = [
            r for r in caplog.records if "no publish_fn" in r.getMessage()
        ]
        assert len(warnings) == 1

    def test_transport_error_does_not_trigger_republish(self, mx_client, nixl_manager):
        publish = MagicMock(return_value="id1")
        publisher = self._publisher(mx_client, nixl_manager, publish)
        mx_client.update_status.side_effect = [
            True,
            RuntimeError("unreachable"),
            RuntimeError("unreachable"),
            True,
        ]

        publisher._tick()
        for _ in range(2):
            with pytest.raises(RuntimeError):
                publisher._tick()
        publisher._tick()

        # Server was unreachable, not rejecting: keep the cached id.
        publish.assert_called_once_with()
        assert publisher.mx_source_id == "id1"

    def test_stop_while_reregistration_pending_does_not_raise(
        self, mx_client, nixl_manager
    ):
        publish = MagicMock(return_value="id1")
        publisher = self._publisher(mx_client, nixl_manager, publish)
        mx_client.update_status.side_effect = [True, False, False]

        for _ in range(3):
            publisher._tick()
        assert publisher.mx_source_id is None

        mx_client.update_status.side_effect = None
        mx_client.update_status.reset_mock()
        publisher.stop()

        # No mx_source_id left, so no STALE update is sent.
        stale_calls = [
            c for c in mx_client.update_status.call_args_list
            if c.kwargs.get("status") == 3
        ]
        assert stale_calls == []


class TestHeartbeatDaemon:
    def test_thread_is_daemon(self, heartbeat):
        heartbeat.start()
        assert heartbeat._thread.daemon is True

    def test_update_status_error_does_not_crash_thread(self, heartbeat, mx_client):
        mx_client.update_status.side_effect = [
            RuntimeError("transient"),  # First tick fails
            True,                        # Second tick succeeds
            True,
        ]
        heartbeat.start()
        time.sleep(2.5)

        assert heartbeat._thread.is_alive()
