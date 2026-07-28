# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""The rendezvous heartbeat must advertise the current blob, not the first one.

A trainer publishes a blob holding its agent metadata and shard table, and the shard
table carries per-shard digests. The heartbeat re-publishes so the server's reaper does
not mark the source STALE while a slow receiver is still initialising.

The first implementation captured the ``worker`` argument in the thread's closure, so
it re-sent the blob from the *first* publish forever. Every later publish wrote a fresh
blob and the heartbeat then reverted it, a race decided by whichever wrote last. The
consequence was a receiver comparing correctly delivered bytes against a step-0 digest
and reporting corruption that did not exist - seen on hardware as one rank verifying
clean while another flagged `model.layers.18.self_attn.o_proj.weight`, repeatably.

These tests pin the current-blob behaviour and the surrounding lifecycle.

Run: pytest tests/test_reshard_refit_heartbeat.py
"""

import threading
import time

import pytest

from modelexpress.refit.reshard.rendezvous import MxReshardRendezvous


PERIOD_S = 0.02
SETTLE_S = 0.25


class _RecordingClient:
    """Captures every publish so the test can see what the heartbeat re-sent."""

    def __init__(self, fail=False):
        self.published = []
        self.fail = fail
        self._lock = threading.Lock()

    def publish_metadata(self, identity, worker, worker_id):
        with self._lock:
            self.published.append(worker.nixl_metadata)
        if self.fail:
            raise RuntimeError("server unreachable")
        return "src-1"

    def snapshot(self):
        with self._lock:
            return list(self.published)


@pytest.fixture
def fast_heartbeat(monkeypatch):
    monkeypatch.setenv("MX_RESHARD_HEARTBEAT_S", str(PERIOD_S))


def _rendezvous(client):
    return MxReshardRendezvous(
        client=client, role="trainer", rank=0, model_name="m", worker_id="w0"
    )


def _stop(rz):
    stop = getattr(rz, "_hb_stop", None)
    if stop is not None:
        stop.set()


def test_the_heartbeat_resends_the_latest_blob_not_the_first(fast_heartbeat):
    """The regression. After a second publish, every subsequent beat must carry the
    second blob; a single reappearance of the first is the bug."""
    client = _RecordingClient()
    rz = _rendezvous(client)
    try:
        rz.publish(b"blob-step-1")
        time.sleep(SETTLE_S)
        rz.publish(b"blob-step-2")
        time.sleep(SETTLE_S)

        sent = client.snapshot()
        after_second = sent[sent.index(b"blob-step-2"):]
        assert b"blob-step-1" not in after_second, (
            "the heartbeat reverted to the first blob, which is how a stale digest "
            "reaches a receiver that then reports phantom corruption"
        )
        assert after_second.count(b"blob-step-2") > 1, "heartbeat did not beat"
    finally:
        _stop(rz)


def test_a_second_publish_does_not_start_a_second_heartbeat(fast_heartbeat):
    """Two threads would double the publish rate and race each other."""
    client = _RecordingClient()
    rz = _rendezvous(client)
    try:
        rz.publish(b"a")
        first = rz._hb_thread
        rz.publish(b"b")
        assert rz._hb_thread is first
        beating = [t for t in threading.enumerate() if t.name.startswith("mx-reshard-hb-")]
        assert len(beating) == 1, f"expected one heartbeat thread, found {len(beating)}"
    finally:
        _stop(rz)


def test_the_current_blob_is_set_before_the_publish_call(fast_heartbeat):
    """A beat firing between publish_metadata and the attribute assignment would send
    the previous blob, so the assignment has to come first."""
    seen = []

    class _CheckingClient(_RecordingClient):
        def publish_metadata(self, identity, worker, worker_id):
            seen.append(getattr(rz, "_hb_worker", None) is not None)
            return super().publish_metadata(identity, worker, worker_id)

    client = _CheckingClient()
    rz = _rendezvous(client)
    try:
        rz.publish(b"a")
        assert seen[0] is True
    finally:
        _stop(rz)


def test_publishing_stops_when_the_period_is_not_positive(monkeypatch):
    """An operator disabling the heartbeat must get no thread at all."""
    monkeypatch.setenv("MX_RESHARD_HEARTBEAT_S", "0")
    client = _RecordingClient()
    rz = _rendezvous(client)
    rz.publish(b"a")
    assert getattr(rz, "_hb_thread", None) is None
    time.sleep(SETTLE_S)
    assert client.snapshot() == [b"a"], "no beats expected when disabled"


def test_a_failing_server_does_not_kill_the_heartbeat(fast_heartbeat):
    """The reaper is unforgiving, so a transient publish failure must not end the
    thread and leave the source to go STALE silently. The server going away for a few
    seconds is ordinary; the source never coming back is not."""
    client = _RecordingClient()
    rz = _rendezvous(client)
    try:
        rz.publish(b"a")
        client.fail = True
        time.sleep(SETTLE_S)
        during_outage = len(client.snapshot())
        assert during_outage > 2, "heartbeat died on the first failure"

        client.fail = False
        time.sleep(SETTLE_S)
        assert len(client.snapshot()) > during_outage, "heartbeat did not recover"
    finally:
        _stop(rz)


def test_a_failed_first_publish_starts_no_heartbeat(fast_heartbeat):
    """Nothing was advertised, so there is nothing to keep alive, and the caller has
    to see the failure rather than have a thread paper over it."""
    client = _RecordingClient(fail=True)
    rz = _rendezvous(client)
    try:
        with pytest.raises(RuntimeError):
            rz.publish(b"a")
        assert getattr(rz, "_hb_thread", None) is None
    finally:
        _stop(rz)


def test_stopping_the_event_ends_the_thread(fast_heartbeat):
    """Teardown has to be possible, or tests and workers leak threads."""
    client = _RecordingClient()
    rz = _rendezvous(client)
    rz.publish(b"a")
    thread = rz._hb_thread
    _stop(rz)
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_the_thread_is_a_daemon_named_for_its_rank(fast_heartbeat):
    """Daemon so a crashed driver can still exit; named so it is identifiable in a
    stack dump taken during a hung refit."""
    client = _RecordingClient()
    rz = _rendezvous(client)
    try:
        rz.publish(b"a")
        assert rz._hb_thread.daemon is True
        assert "mx-reshard-hb-0" == rz._hb_thread.name
    finally:
        _stop(rz)
