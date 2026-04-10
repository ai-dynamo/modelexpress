# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for HeartbeatThread."""

import time
from unittest.mock import MagicMock, call, patch

import pytest

from modelexpress.heartbeat import HeartbeatThread


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
        hb = HeartbeatThread(
            mx_client=mx_client,
            worker_id="w1",
            worker_rank=0,
            nixl_manager=nixl_manager,
            publish_fn=lambda: "abc123",
        )
    yield hb
    hb.stop()


class TestHeartbeatPublishAndReady:
    def test_first_tick_publishes_then_ready(self, heartbeat, mx_client):
        heartbeat.start()
        time.sleep(2.5)
        heartbeat.stop()

        assert heartbeat.mx_source_id == "abc123"
        ready_calls = [
            c for c in mx_client.update_status.call_args_list
            if c == call(
                mx_source_id="abc123",
                worker_id="w1",
                worker_rank=0,
                status=2,  # SOURCE_STATUS_READY
            )
        ]
        assert len(ready_calls) >= 1

    def test_publish_failure_retries(self, mx_client, nixl_manager):
        attempt = {"count": 0}

        def failing_then_ok():
            attempt["count"] += 1
            if attempt["count"] < 3:
                raise RuntimeError("server down")
            return "abc123"

        with patch.dict("os.environ", {"MX_HEARTBEAT_INTERVAL_SECS": "1"}):
            hb = HeartbeatThread(
                mx_client=mx_client,
                worker_id="w1",
                worker_rank=0,
                nixl_manager=nixl_manager,
                publish_fn=failing_then_ok,
            )
        hb.start()
        time.sleep(3.5)
        hb.stop()

        assert hb.mx_source_id == "abc123"
        assert attempt["count"] == 3

    def test_skips_ready_when_unhealthy(self, heartbeat, mx_client, nixl_manager):
        nixl_manager.is_healthy.return_value = False
        heartbeat.start()
        time.sleep(2.5)
        heartbeat.stop()

        # Publish should succeed on first tick, but READY should not be sent
        assert heartbeat.mx_source_id == "abc123"
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
        time.sleep(3.5)
        heartbeat.stop()

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


class TestHeartbeatStop:
    def test_stop_marks_stale(self, heartbeat, mx_client):
        heartbeat.start()
        time.sleep(2.5)  # Let publish + READY go through
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

    def test_stop_without_ready_skips_stale(self, mx_client, nixl_manager):
        nixl_manager.is_healthy.return_value = False

        def never_publish():
            raise RuntimeError("server down")

        with patch.dict("os.environ", {"MX_HEARTBEAT_INTERVAL_SECS": "1"}):
            hb = HeartbeatThread(
                mx_client=mx_client,
                worker_id="w1",
                worker_rank=0,
                nixl_manager=nixl_manager,
                publish_fn=never_publish,
            )
        hb.start()
        time.sleep(1.5)
        hb.stop()

        # Never published, so _mark_stale should not send STALE
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
        time.sleep(2.5)
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
        time.sleep(2.5)
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
        time.sleep(2.5)
        mx_client.update_status.side_effect = RuntimeError("connection lost")
        heartbeat._on_exit()  # Should not raise


class TestHeartbeatDaemon:
    def test_thread_is_daemon(self, heartbeat):
        heartbeat.start()
        assert heartbeat._thread.daemon is True

    def test_update_status_error_does_not_crash_thread(self, heartbeat, mx_client):
        mx_client.update_status.side_effect = [
            RuntimeError("transient"),  # First tick fails (during READY)
            True,                        # Second tick succeeds
            True,
        ]
        heartbeat.start()
        time.sleep(3.5)

        assert heartbeat._thread.is_alive()
