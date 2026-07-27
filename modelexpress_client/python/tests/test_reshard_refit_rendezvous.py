# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import metadata
from types import SimpleNamespace

import pytest

from modelexpress import p2p_pb2
from modelexpress.refit.reshard.rendezvous import (
    MxReshardRendezvous,
    PublishedShard,
    PublishedTensor,
    _mx_version,
    merge_shard_tables,
    wrap_rendezvous_blob,
)


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


def _blob(agent: str, endpoint: str, tensors: list) -> bytes:
    return wrap_rendezvous_blob(b"meta", agent, endpoint, tensors)


def _one_tensor(agent: str) -> list:
    return [
        PublishedTensor(
            name="weight",
            dtype="torch.bfloat16",
            elsize=2,
            full_shape=(4, 4),
            shards=[
                PublishedShard(
                    agent_name=agent,
                    device_id=0,
                    addr=1000,
                    shard_offset=(0, 0),
                    shape=(4, 4),
                )
            ],
        )
    ]


class _DiscoveryClient:
    """Serves a fixed set of (agent, tensors) as READY sources."""

    def __init__(self, entries):
        self.entries = entries

    def list_sources(self, _identity, status_filter=None):
        return SimpleNamespace(
            instances=[
                SimpleNamespace(mx_source_id="s", worker_id=agent)
                for agent, _ in self.entries
            ]
        )

    def get_metadata(self, _source_id, worker_id):
        tensors = dict(self.entries)[worker_id]
        return SimpleNamespace(
            found=True,
            worker=SimpleNamespace(
                nixl_metadata=_blob(worker_id, f"{worker_id}:9999", tensors)
            ),
        )


def test_publishers_with_empty_shard_tables_do_not_count_toward_quorum():
    """A rank advertising no tensors has registered nothing to read. Counting it
    lets the receiver stop waiting for real ranks and stall in the handshake."""
    client = _DiscoveryClient([("r0", _one_tensor("r0")), ("r1", [])])
    rendezvous = MxReshardRendezvous(
        client, role="inference", rank=0, model_name="model"
    )

    with pytest.raises(TimeoutError, match="1 empty"):
        rendezvous.discover_trainers(expected_trainers=2, timeout=0)


def test_discovery_returns_only_non_empty_publishers():
    client = _DiscoveryClient(
        [("r0", _one_tensor("r0")), ("r1", []), ("r2", _one_tensor("r2"))]
    )
    rendezvous = MxReshardRendezvous(
        client, role="inference", rank=0, model_name="model"
    )

    payloads = rendezvous.discover_trainers(expected_trainers=2, timeout=0)

    assert [name for (_m, name, _e, _t) in payloads] == ["r0", "r2"]


def test_discovery_timeout_message_separates_ready_from_usable():
    """'saw 0' and 'saw 16 but all empty' need different fixes, so the message
    must distinguish them."""
    client = _DiscoveryClient([("r0", []), ("r1", [])])
    rendezvous = MxReshardRendezvous(
        client, role="inference", rank=0, model_name="model"
    )

    with pytest.raises(TimeoutError) as excinfo:
        rendezvous.discover_trainers(expected_trainers=2, timeout=0)
    message = str(excinfo.value)
    assert "2 READY source(s)" in message
    assert "0 with a non-empty shard table" in message
    assert "2 empty" in message


def test_publish_marks_registered_rendezvous_ready():
    class Client:
        def __init__(self):
            self.worker = None

        def publish_metadata(self, _identity, worker, _worker_id):
            self.worker = worker
            return "source-id"

    client = Client()
    rendezvous = MxReshardRendezvous(
        client,
        role="trainer",
        rank=3,
        model_name="model",
    )

    assert rendezvous.publish(b"registered") == "source-id"
    assert client.worker.status == p2p_pb2.SOURCE_STATUS_READY


def test_merge_deduplicates_dp_replica_geometry():
    def table(agent: str, address: int):
        return [
            PublishedTensor(
                name="weight",
                dtype="torch.bfloat16",
                elsize=2,
                full_shape=(8, 4),
                shards=[
                    PublishedShard(
                        agent_name=agent,
                        device_id=0,
                        addr=address,
                        shard_offset=(0, 0),
                        shape=(4, 4),
                    )
                ],
            )
        ]

    merged = merge_shard_tables([table("dp0", 100), table("dp1", 200)])

    assert len(merged) == 1
    assert len(merged[0].shards) == 1
    assert merged[0].shards[0].agent_name == "dp0"


def _dp_replica_tables(replicas: int, shards_per_rank: int = 2):
    """`replicas` publishers each advertising the same `shards_per_rank` geometries.

    This is the DP8 / EDP2 shape: byte-identical copies of a shard on distinct
    ranks, at distinct addresses.
    """
    tables = []
    for dp in range(replicas):
        tables.append(
            [
                PublishedTensor(
                    name="weight",
                    dtype="torch.bfloat16",
                    elsize=2,
                    full_shape=(8, 4),
                    shards=[
                        PublishedShard(
                            agent_name=f"dp{dp}",
                            device_id=dp,
                            addr=1000 * (dp + 1) + 10 * i,
                            shard_offset=(4 * i, 0),
                            shape=(4, 4),
                        )
                        for i in range(shards_per_rank)
                    ],
                )
            ]
        )
    return tables


def test_replica_offset_rotates_which_publisher_serves():
    tables = _dp_replica_tables(replicas=8)
    served = {}
    for offset in range(8):
        merged = merge_shard_tables(tables, replica_offset=offset)
        served[offset] = [s.agent_name for s in merged[0].shards]

    # Each offset picks a different replica, so 8 receivers spread over 8 ranks
    # instead of all landing on dp0.
    assert served[0] == ["dp0", "dp0"]
    assert served[3] == ["dp3", "dp3"]
    assert len({tuple(v) for v in served.values()}) == 8


def test_replica_offset_wraps_past_the_replica_count():
    tables = _dp_replica_tables(replicas=2)
    # Rank 5 with only 2 replicas available must still resolve, not raise.
    merged = merge_shard_tables(tables, replica_offset=5)
    assert [s.agent_name for s in merged[0].shards] == ["dp1", "dp1"]


def test_replica_offset_changes_only_the_owner_not_the_geometry():
    """The plan must be identical in shape and bytes regardless of who serves."""
    tables = _dp_replica_tables(replicas=8)
    baseline = merge_shard_tables(tables, replica_offset=0)[0]
    for offset in range(1, 8):
        other = merge_shard_tables(tables, replica_offset=offset)[0]
        assert other.full_shape == baseline.full_shape
        assert other.dtype == baseline.dtype and other.elsize == baseline.elsize
        assert [
            (tuple(s.shard_offset), tuple(s.shape)) for s in other.shards
        ] == [(tuple(s.shard_offset), tuple(s.shape)) for s in baseline.shards]
        # ... but the addresses do differ, i.e. it really is a different rank.
        assert [s.addr for s in other.shards] != [s.addr for s in baseline.shards]


def test_default_offset_preserves_first_writer_wins():
    """Default must be byte-for-byte the old behavior, so it is a safe no-op."""
    tables = _dp_replica_tables(replicas=4)
    merged = merge_shard_tables(tables)
    assert [s.agent_name for s in merged[0].shards] == ["dp0", "dp0"]


def test_merge_still_fans_in_distinct_geometry_across_ranks():
    """Non-replica shards (real cross-rank fan-in) must all be retained."""
    tables = []
    for rank in range(4):
        tables.append(
            [
                PublishedTensor(
                    name="weight",
                    dtype="torch.bfloat16",
                    elsize=2,
                    full_shape=(16, 4),
                    shards=[
                        PublishedShard(
                            agent_name=f"r{rank}",
                            device_id=rank,
                            addr=1000 * (rank + 1),
                            shard_offset=(4 * rank, 0),
                            shape=(4, 4),
                        )
                    ],
                )
            ]
        )
    for offset in (0, 2, 7):
        merged = merge_shard_tables(tables, replica_offset=offset)
        assert len(merged[0].shards) == 4, offset
        assert sorted(s.agent_name for s in merged[0].shards) == [
            "r0",
            "r1",
            "r2",
            "r3",
        ]
