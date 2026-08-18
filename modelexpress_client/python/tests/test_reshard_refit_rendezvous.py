# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import metadata
from threading import Event
from types import SimpleNamespace

import pytest

from modelexpress import p2p_pb2
from modelexpress.refit.reshard.rendezvous import (
    MxReshardRendezvous,
    PublishedShard,
    PublishedTensor,
    _mx_version,
    wrap_rendezvous_blob,
)


def _one_tensor(agent_name="trainer-agent"):
    """The smallest publishable shard table: one tensor, one shard."""
    return [
        PublishedTensor(
            name="weight",
            dtype="torch.bfloat16",
            elsize=2,
            full_shape=(4, 4),
            shards=[
                PublishedShard(
                    agent_name=agent_name,
                    device_id=0,
                    addr=4096,
                    shard_offset=(0, 0),
                    shape=(4, 4),
                )
            ],
        )
    ]


def test_mx_version_falls_back_only_when_package_is_missing(monkeypatch):
    def missing(_name):
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(metadata, "version", missing)
    assert _mx_version() == "0.0.0"

    def broken(_name):
        raise RuntimeError("metadata backend failure")

    monkeypatch.setattr(metadata, "version", broken)
    with pytest.raises(RuntimeError, match="metadata backend failure"):
        _mx_version()


def test_discovery_filters_for_ready_trainers():
    class Client:
        def __init__(self):
            self.status_filter = None

        def list_sources(self, _identity, status_filter=None):
            self.status_filter = status_filter
            return SimpleNamespace(instances=[])

    client = Client()
    rendezvous = MxReshardRendezvous(
        client,
        role="inference",
        rank=0,
        model_name="model",
    )

    with pytest.raises(TimeoutError):
        rendezvous.discover_trainers(expected_trainers=1, timeout=0)
    assert client.status_filter == p2p_pb2.SOURCE_STATUS_READY


def test_published_rendezvous_stays_ready_and_closes_stale(monkeypatch):
    class Client:
        def __init__(self):
            self.worker = None
            self.worker_id = None
            self.status_filter = None
            self.publish_count = 0
            self.heartbeat_seen = Event()
            self.status_updates = []

        def publish_metadata(self, _identity, worker, worker_id):
            self.worker = worker
            self.worker_id = worker_id
            self.publish_count += 1
            return "source-id"

        def list_sources(self, _identity, status_filter=None):
            self.status_filter = status_filter
            instances = []
            if self.worker is not None and self.worker.status == status_filter:
                instances.append(
                    SimpleNamespace(
                        mx_source_id="source-id",
                        worker_id=self.worker_id,
                    )
                )
            return SimpleNamespace(instances=instances)

        def get_metadata(self, _source_id, _worker_id):
            return SimpleNamespace(found=True, worker=self.worker)

        def update_status(self, **kwargs):
            self.status_updates.append(kwargs)
            if kwargs["status"] == p2p_pb2.SOURCE_STATUS_READY:
                self.heartbeat_seen.set()
            return True

    monkeypatch.setenv("MX_HEARTBEAT_INTERVAL_SECS", "1")
    client = Client()
    rendezvous = MxReshardRendezvous(
        client,
        role="trainer",
        rank=2,
        model_name="model",
        worker_id="trainer-2",
    )
    blob = wrap_rendezvous_blob(
        agent_metadata=b"nixl",
        agent_name="trainer-agent",
        metadata_endpoint="trainer:1234",
        tensors=_one_tensor(),
    )

    try:
        assert rendezvous.publish(blob) == "source-id"
        assert client.worker.status == p2p_pb2.SOURCE_STATUS_READY
        assert client.heartbeat_seen.wait(timeout=1.0)
        assert client.publish_count == 1
        discovered = rendezvous.discover_trainers(expected_trainers=1)
        assert [
            (p.agent_metadata, p.agent_name, p.metadata_endpoint) for p in discovered
        ] == [(b"nixl", "trainer-agent", "trainer:1234")]
        assert [t.name for t in discovered[0].tensors] == ["weight"]
        assert client.status_filter == p2p_pb2.SOURCE_STATUS_READY
    finally:
        rendezvous.close()

    assert client.status_updates[-1] == {
        "mx_source_id": "source-id",
        "worker_id": "trainer-2",
        "worker_rank": 2,
        "status": p2p_pb2.SOURCE_STATUS_STALE,
    }


class _DiscoveryClient:
    """Serves a fixed set of READY sources, each with its own shard table."""

    def __init__(self, blobs):
        self._blobs = list(blobs)

    def list_sources(self, _identity, status_filter=None):
        return SimpleNamespace(
            instances=[
                SimpleNamespace(mx_source_id=f"src-{i}", worker_id=f"w-{i}")
                for i in range(len(self._blobs))
            ]
        )

    def get_metadata(self, source_id, _worker_id):
        index = int(source_id.rsplit("-", 1)[1])
        return SimpleNamespace(
            found=True,
            worker=SimpleNamespace(nixl_metadata=self._blobs[index]),
        )


def _blob(agent_name, tensors):
    return wrap_rendezvous_blob(
        agent_metadata=b"nixl",
        agent_name=agent_name,
        metadata_endpoint=f"{agent_name}:1234",
        tensors=tensors,
    )


def _rendezvous(client):
    return MxReshardRendezvous(client, role="inference", rank=0, model_name="model")


def test_a_publisher_with_no_tensors_does_not_count_toward_the_quorum():
    """It has registered no memory, so counting it makes the receiver stop
    waiting for the ranks that do have bytes and then stall in the handshake."""
    client = _DiscoveryClient(
        [_blob("empty-rank", []), _blob("real-rank", _one_tensor())]
    )

    with pytest.raises(TimeoutError) as excinfo:
        _rendezvous(client).discover_trainers(expected_trainers=2, timeout=0)

    message = str(excinfo.value)
    assert "2 READY source(s)" in message
    assert "1 with a non-empty shard table" in message
    assert "1 empty" in message


def test_quorum_is_met_once_every_rank_publishes_tensors():
    client = _DiscoveryClient(
        [_blob("rank-0", _one_tensor()), _blob("rank-1", _one_tensor())]
    )

    discovered = _rendezvous(client).discover_trainers(expected_trainers=2, timeout=0)

    assert [p.agent_name for p in discovered] == ["rank-0", "rank-1"]


def test_empty_publishers_are_excluded_from_the_returned_payloads():
    """An extra healthy rank means the quorum is reachable without the empty one,
    which must still not appear in the plan's sources."""
    client = _DiscoveryClient(
        [
            _blob("rank-0", _one_tensor()),
            _blob("empty-rank", []),
            _blob("rank-1", _one_tensor()),
        ]
    )

    discovered = _rendezvous(client).discover_trainers(expected_trainers=2, timeout=0)

    assert "empty-rank" not in [p.agent_name for p in discovered]
    assert len(discovered) == 2


def test_only_the_requested_number_of_ranks_is_returned():
    """A stale source from an earlier run stays READY. Returning it adds a peer to
    handshake and a second set of shards describing the same tensor names."""
    client = _DiscoveryClient(
        [
            _blob("rank-0", _one_tensor()),
            _blob("rank-1", _one_tensor()),
            _blob("stale-rank", _one_tensor()),
        ]
    )

    discovered = _rendezvous(client).discover_trainers(expected_trainers=2, timeout=0)

    assert [p.agent_name for p in discovered] == ["rank-0", "rank-1"]


def test_a_discovered_payload_reads_by_field_and_by_position():
    """Named access is the point of the change. Positions are still stable, which is
    what lets an optional field like the publisher step be appended without moving
    anything a caller already reads."""
    client = _DiscoveryClient([_blob("rank-0", _one_tensor())])

    (payload,) = _rendezvous(client).discover_trainers(expected_trainers=1, timeout=0)

    assert payload[:4] == (
        payload.agent_metadata,
        payload.agent_name,
        payload.metadata_endpoint,
        payload.tensors,
    )
    assert payload[3] is payload.tensors


def test_a_partial_ready_set_still_reports_its_shard_tables():
    """Below the quorum the shard-table state is exactly what the timeout has to
    report: one READY publisher with nothing to serve is a different failure from
    one rank that has not come up yet."""
    client = _DiscoveryClient([_blob("empty-rank", [])])

    with pytest.raises(TimeoutError) as excinfo:
        _rendezvous(client).discover_trainers(expected_trainers=2, timeout=0)

    message = str(excinfo.value)
    assert "1 READY source(s)" in message
    assert "0 with a non-empty shard table" in message
    assert "1 empty" in message


def test_invalid_heartbeat_period_fails_before_publish(monkeypatch):
    class Client:
        def publish_metadata(self, *_args, **_kwargs):
            raise AssertionError("publish must not run")

    monkeypatch.setenv("MX_HEARTBEAT_INTERVAL_SECS", "0")
    rendezvous = MxReshardRendezvous(
        Client(),
        role="trainer",
        rank=0,
        model_name="model",
    )

    with pytest.raises(ValueError, match="must be positive"):
        rendezvous.publish(b"registered")


def test_quorum_can_skip_shard_tables_without_losing_emptiness():
    """The per-step quorum check needs each rank's version stamp, not its shard
    table, and rebuilding the table dominated the call. Skipping it must not make
    a rank that published nothing look like a valid member of the quorum, which is
    the one thing the emptiness rule exists to prevent."""
    client = _DiscoveryClient(
        [_blob("empty-rank", []), _blob("real-rank", _one_tensor())]
    )

    with pytest.raises(TimeoutError) as excinfo:
        _rendezvous(client).discover_trainers(
            expected_trainers=2, timeout=0, with_tensors=False
        )

    message = str(excinfo.value)
    assert "1 with a non-empty shard table" in message
    assert "1 empty" in message


def test_skipping_shard_tables_still_reports_the_entry_count(caplog):
    """``entry_count`` is what keeps emptiness decidable, and it is also the figure
    that showed this cost scales with source count rather than bytes moved."""
    import json
    import logging

    client = _DiscoveryClient([_blob("rank-0", _one_tensor() * 3)])

    with caplog.at_level(logging.WARNING):
        (payload,) = _rendezvous(client).discover_trainers(
            expected_trainers=1, timeout=0, with_tensors=False
        )

    assert payload.tensors == []
    assert payload.entry_count() == 3
    assert payload.agent_name == "rank-0"
    record = next(
        json.loads(message.split("MX_DISCOVER_COST ", 1)[1])
        for message in caplog.messages
        if "MX_DISCOVER_COST " in message
    )
    assert record["rank"] == 0
    assert record["tensors"] == 3
    assert record["tables_built"] is False


def test_entry_count_falls_back_to_the_table_when_not_recorded():
    """Payloads built directly, as tests and older callers do, record no count."""
    from modelexpress.refit.reshard.rendezvous import RendezvousPayload

    payload = RendezvousPayload(b"", "a", "", _one_tensor())
    assert payload.tensor_count is None
    assert payload.entry_count() == 1


def test_decoding_parsed_entries_matches_decoding_the_blob():
    """The shard table used to be re-serialized only to be parsed again. Removing
    that round-trip must not change a single decoded field."""
    import json

    from modelexpress.refit.reshard.rendezvous import (
        _SCHEMA,
        decode_shard_entries,
        decode_shard_table,
        encode_shard_table,
    )

    blob = encode_shard_table(_one_tensor())
    entries = json.loads(blob.decode("utf-8"))["tensors"]

    assert decode_shard_entries(entries) == decode_shard_table(blob)
    assert decode_shard_table(blob) == _one_tensor()
    assert json.loads(blob.decode("utf-8"))["schema"] == _SCHEMA


def test_metadata_fetches_stay_serial():
    """Concurrency here was measured to be slower, not faster: from a thread pool
    the fetch went 4.02 s -> ~6.9 s median on 16 sources, because every receiver
    rank runs this loop and the single metadata server, not this process, is the
    contended resource. Pinned with a barrier that a concurrent implementation
    would satisfy and a serial one cannot, so the decision cannot be quietly
    reversed without this failing."""
    from threading import Barrier, BrokenBarrierError

    ranks = 4
    barrier = Barrier(ranks, timeout=0.5)
    overlapped = []

    class Barriered(_DiscoveryClient):
        def get_metadata(self, source_id, worker_id):
            try:
                barrier.wait()
                overlapped.append(source_id)
            except BrokenBarrierError:
                pass
            return super().get_metadata(source_id, worker_id)

    client = Barriered([_blob(f"rank-{i}", _one_tensor()) for i in range(ranks)])

    discovered = _rendezvous(client).discover_trainers(
        expected_trainers=ranks, timeout=0
    )

    assert len(discovered) == ranks
    assert overlapped == [], "fetches overlapped; this path is deliberately serial"


def test_quorum_membership_does_not_depend_on_completion_order():
    """Concurrency must not make which ranks satisfy the quorum depend on who
    answers first. With more READY sources than needed, the same prefix has to win
    every time, or two receivers can disagree about the source set they read."""
    import time as _time

    class Reordering(_DiscoveryClient):
        def get_metadata(self, source_id, worker_id):
            # Earlier ranks answer last, inverting completion order.
            index = int(source_id.rsplit("-", 1)[1])
            _time.sleep(0.02 * (4 - index))
            return super().get_metadata(source_id, worker_id)

    client = Reordering([_blob(f"rank-{i}", _one_tensor()) for i in range(4)])

    discovered = _rendezvous(client).discover_trainers(expected_trainers=2, timeout=0)

    assert [p.agent_name for p in discovered] == ["rank-0", "rank-1"]


def test_one_unreadable_rank_does_not_abort_the_sweep():
    """The poll loop's value is reporting how many ranks are readable. A single
    rank's transport error must therefore be counted, not raised: otherwise a
    transient failure on one rank surfaces as a discovery crash with no count."""

    class OneBadRank(_DiscoveryClient):
        def get_metadata(self, source_id, worker_id):
            if source_id.endswith("-1"):
                raise RuntimeError("transport blip on rank 1")
            return super().get_metadata(source_id, worker_id)

    client = OneBadRank([_blob(f"rank-{i}", _one_tensor()) for i in range(3)])

    with pytest.raises(TimeoutError) as excinfo:
        _rendezvous(client).discover_trainers(expected_trainers=3, timeout=0)
    assert "3 READY source(s)" in str(excinfo.value)
    assert "2 with a non-empty shard table" in str(excinfo.value)

    # The readable ranks are still returned when the quorum only needs them.
    discovered = _rendezvous(client).discover_trainers(expected_trainers=2, timeout=0)
    assert [p.agent_name for p in discovered] == ["rank-0", "rank-2"]
