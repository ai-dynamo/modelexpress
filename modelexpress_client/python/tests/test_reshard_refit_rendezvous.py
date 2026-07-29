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
        assert [(meta, name, ep) for (meta, name, ep, _t) in discovered] == [
            (b"nixl", "trainer-agent", "trainer:1234")
        ]
        assert [t.name for t in discovered[0][3]] == ["weight"]
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

    assert [name for (_meta, name, _ep, _tensors) in discovered] == [
        "rank-0",
        "rank-1",
    ]


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

    assert "empty-rank" not in [name for (_meta, name, _ep, _t) in discovered]
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

    assert [name for (_meta, name, _ep, _t) in discovered] == ["rank-0", "rank-1"]


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
