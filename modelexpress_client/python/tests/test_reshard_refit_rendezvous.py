# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import metadata
from threading import Event
from types import SimpleNamespace

import pytest

from modelexpress import p2p_pb2
from modelexpress.refit.reshard.rendezvous import (
    MxReshardRendezvous,
    _mx_version,
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
        tensors=[],
    )

    try:
        assert rendezvous.publish(blob) == "source-id"
        assert client.worker.status == p2p_pb2.SOURCE_STATUS_READY
        assert client.heartbeat_seen.wait(timeout=1.0)
        assert client.publish_count == 1
        assert rendezvous.discover_trainers(expected_trainers=1) == [
            (b"nixl", "trainer-agent", "trainer:1234", [])
        ]
        assert client.status_filter == p2p_pb2.SOURCE_STATUS_READY
    finally:
        rendezvous.close()

    assert client.status_updates[-1] == {
        "mx_source_id": "source-id",
        "worker_id": "trainer-2",
        "worker_rank": 2,
        "status": p2p_pb2.SOURCE_STATUS_STALE,
    }


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
