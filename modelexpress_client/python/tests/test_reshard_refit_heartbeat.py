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
    # Counted as a delta, not an absolute. Other tests in the suite leave heartbeat
    # threads behind, so an absolute count made this fail whenever it ran after one
    # of them and pass in isolation - which reads as a flaky test and got this one
    # dismissed as noise for a whole session.
    before = {
        t for t in threading.enumerate() if t.name.startswith("mx-reshard-hb-")
    }
    client = _RecordingClient()
    rz = _rendezvous(client)
    try:
        rz.publish(b"a")
        first = rz._hb_thread
        rz.publish(b"b")
        assert rz._hb_thread is first
        started = [
            t
            for t in threading.enumerate()
            if t.name.startswith("mx-reshard-hb-") and t not in before
        ]
        assert len(started) == 1, f"expected one new heartbeat thread, got {started}"
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


# --- the same defect by another route: one rendezvous per publish ---------------
# The tests above publish twice through one object, and the guard in
# _start_rendezvous_heartbeat is per object, so they pass while production reverts.
# publish_registered_shard_table is called once per refit and used to construct a
# fresh rendezvous each time, so after k refits k threads were re-asserting k
# different snapshots under one key. These drive the real entry point.


class _Manager:
    agent_name = "agent-0"
    nixl_metadata = b"nixl-meta"


def _publish_through_the_real_entry_point(client, blob_tag, *, worker_id="w0"):
    """Call publish_registered_shard_table the way a refit does."""
    from modelexpress.refit.reshard.megatron_publisher import (
        publish_registered_shard_table,
    )
    from modelexpress.refit.reshard.rendezvous import PublishedShard, PublishedTensor

    published = [
        PublishedTensor(
            name="w",
            dtype="torch.bfloat16",
            elsize=2,
            full_shape=(4,),
            shards=[
                PublishedShard(
                    agent_name="agent-0",
                    device_id=0,
                    addr=4096,
                    shard_offset=(0,),
                    shape=(4,),
                    digest=blob_tag,
                )
            ],
        )
    ]
    return publish_registered_shard_table(
        manager=_Manager(),
        client=client,
        model_name="m",
        worker_rank=0,
        worker_id=worker_id,
        published=published,
        metadata_endpoint="10.0.0.1:1234",
    )


@pytest.fixture
def clean_rendezvous_cache():
    from modelexpress.refit.reshard import megatron_publisher

    megatron_publisher._RENDEZVOUS.clear()
    yield
    for rz in megatron_publisher._RENDEZVOUS.values():
        rz.stop_heartbeat()
    megatron_publisher._RENDEZVOUS.clear()


def _live_heartbeats():
    return [
        t
        for t in threading.enumerate()
        if t.name.startswith("mx-reshard-hb-") and t.is_alive()
    ]


def test_a_refit_per_step_does_not_accumulate_heartbeat_threads(
    fast_heartbeat, clean_rendezvous_cache
):
    """The bug, stated as a count. Three refits used to mean three threads."""
    before = len(_live_heartbeats())
    client = _RecordingClient()

    for step in range(1, 4):
        _publish_through_the_real_entry_point(client, f"digest-step-{step}")

    assert len(_live_heartbeats()) - before == 1


def test_repeated_refits_never_revert_the_published_digest(
    fast_heartbeat, clean_rendezvous_cache
):
    """The consequence, and the reason this mattered: a reverted table makes a
    receiver check correct bytes against an earlier step's digest."""
    client = _RecordingClient()

    _publish_through_the_real_entry_point(client, "digest-step-1")
    time.sleep(SETTLE_S)
    _publish_through_the_real_entry_point(client, "digest-step-2")
    time.sleep(SETTLE_S)

    # Everything sent after the second publish must carry step 2. A single
    # reappearance of step 1 is the defect, because the server keeps the last write.
    after_second = client.snapshot()
    tail = after_second[after_second.index(
        next(b for b in after_second if b"digest-step-2" in b)
    ):]
    assert tail, "the second publish should have been sent"
    assert not [b for b in tail if b"digest-step-1" in b], (
        "an earlier step's shard table was re-advertised after a later publish"
    )


def test_the_same_identity_reuses_one_rendezvous(
    fast_heartbeat, clean_rendezvous_cache
):
    from modelexpress.refit.reshard import megatron_publisher

    client = _RecordingClient()
    _publish_through_the_real_entry_point(client, "d1")
    _publish_through_the_real_entry_point(client, "d2")

    assert len(megatron_publisher._RENDEZVOUS) == 1


def test_distinct_workers_keep_distinct_rendezvous(
    fast_heartbeat, clean_rendezvous_cache
):
    """Reuse must be keyed tightly enough that two ranks in one process do not
    share a publisher identity."""
    from modelexpress.refit.reshard import megatron_publisher

    client = _RecordingClient()
    _publish_through_the_real_entry_point(client, "d1", worker_id="w0")
    _publish_through_the_real_entry_point(client, "d2", worker_id="w1")

    assert len(megatron_publisher._RENDEZVOUS) == 2


def test_a_replaced_client_retires_the_heartbeat_it_orphans(
    fast_heartbeat, clean_rendezvous_cache
):
    """A rendezvous whose client is gone cannot refresh its blob, so leaving its
    heartbeat running would re-assert a snapshot nothing can update."""
    before = len(_live_heartbeats())
    _publish_through_the_real_entry_point(_RecordingClient(), "d1")
    _publish_through_the_real_entry_point(_RecordingClient(), "d2")
    time.sleep(SETTLE_S)

    assert len(_live_heartbeats()) - before == 1


def test_stop_heartbeat_is_safe_before_any_publish():
    rz = _rendezvous(_RecordingClient())
    rz.stop_heartbeat()  # must not raise
