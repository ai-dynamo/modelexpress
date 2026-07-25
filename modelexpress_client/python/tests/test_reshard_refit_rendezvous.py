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


def test_published_rendezvous_is_ready_and_discoverable():
    class Client:
        def __init__(self):
            self.worker = None
            self.worker_id = None
            self.status_filter = None

        def publish_metadata(self, _identity, worker, worker_id):
            self.worker = worker
            self.worker_id = worker_id
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
        tensors=[],
    )

    assert rendezvous.publish(blob) == "source-id"
    assert client.worker.status == p2p_pb2.SOURCE_STATUS_READY
    assert rendezvous.discover_trainers(expected_trainers=1) == [
        (b"nixl", "trainer-agent", "trainer:1234", [])
    ]
    assert client.status_filter == p2p_pb2.SOURCE_STATUS_READY


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


def test_merge_keeps_distinct_geometry_from_different_ranks():
    def table(agent: str, offset: tuple[int, int]):
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
                        addr=100,
                        shard_offset=offset,
                        shape=(4, 4),
                    )
                ],
            )
        ]

    merged = merge_shard_tables([table("tp0", (0, 0)), table("tp1", (4, 0))])

    assert len(merged) == 1
    assert {s.shard_offset for s in merged[0].shards} == {(0, 0), (4, 0)}
