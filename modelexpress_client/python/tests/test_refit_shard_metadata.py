# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import grpc
import pytest

from modelexpress_rl import WeightPayloadFormat, refit_pb2, refit_pb2_grpc
from modelexpress_rl.control import WeightVersion, WeightVersionState
from modelexpress_rl.inference.source import TrainerSourceResolver
from tests.test_refit_generator_client import _start_server


@pytest.fixture
def protocol(monkeypatch):
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1")
    server, endpoint, control = _start_server()
    control.shards = control.shards[:1]
    with grpc.insecure_channel(endpoint) as channel:
        service = refit_pb2_grpc.RefitServiceStub(channel)
        case = SimpleNamespace(
            control=control, worker=control.worker,
            version=WeightVersion(
                version_id="version-a", model_name="test/model",
                payload_format=WeightPayloadFormat.FULL_TENSOR, layout_signature="layout-a",
                state=WeightVersionState.READY, created_at_unix_ms=0,
                trainer_mesh_generation=1, trainer_mesh_id="mesh-a",
            ),
        )
        case.resolver = TrainerSourceResolver(
            service=lambda: service, rpc_timeout_seconds=2,
        )
        try:
            yield case
        finally:
            server.stop(grace=None).wait()


def publish(case, version, digest, address=4096):
    payload = json.loads(case.worker.manifest)
    payload["tensors"][0]["shards"][0].update(digest=digest, addr=address)
    case.worker.manifest = json.dumps(payload).encode()
    shard = case.control.shards[0]
    case.control.version.uid = version
    shard.version_id = version
    shard.manifest_digest = hashlib.sha256(case.worker.manifest).hexdigest()
    return replace(case.version, version_id=version)


@pytest.mark.parametrize("initial_verify,verify", [(False, False), (True, True), (False, True), (True, False)])
def test_current_manifest_checksums_and_unchanged_manifest_reuse(protocol, monkeypatch, initial_verify, verify):
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", str(int(initial_verify)))
    version = publish(protocol, "version-a", "first")
    first, = protocol.resolver.candidates(version)
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", str(int(verify)))
    again, = protocol.resolver.candidates(version)
    assert len(protocol.worker.requests) == 1
    assert again.snapshot.shards[0].manifest == protocol.worker.manifest
    assert list(again.checksums.values()) == (["first"] if verify else [])
    version = publish(protocol, "version-b", "second")
    current, = protocol.resolver.candidates(version)
    assert len(protocol.worker.requests) == 2
    assert list(first.checksums.values()) == (["first"] if initial_verify else [])
    assert list(current.checksums.values()) == (["second"] if verify else [])


@pytest.mark.parametrize("change", ["geometry", "generation", "replica"])
def test_changed_sources_resolve_current_manifest(protocol, change):
    first, = protocol.resolver.candidates(publish(protocol, "version-a", "first"))
    version = publish(protocol, "version-b", "second", address=8192)
    if change == "generation":
        protocol.control.mesh_generation_on_recheck = 2
        version = replace(version, trainer_mesh_generation=2)
    elif change == "replica":
        protocol.control.shards[0].worker_id = "replacement"
    current, = protocol.resolver.candidates(version)
    assert current.snapshot.shards[0].manifest == protocol.worker.manifest
    assert next(iter(current.checksums))[2] == 8192
    assert list(current.checksums.values()) == ["second"]
    assert len(protocol.worker.requests) == 2


@pytest.mark.parametrize("fallback", [False, True])
def test_invalid_primary_manifest_uses_only_published_fallback(protocol, fallback):
    publish(protocol, "version-a", "first")
    alternate = refit_pb2.WeightVersionShard()
    alternate.CopyFrom(protocol.control.shards[0])
    alternate.worker_id = "trainer-replica"
    protocol.control.shards[0].manifest_digest = "bad-digest"
    if fallback:
        protocol.control.shards.append(alternate)
    candidates = list(protocol.resolver.candidates(protocol.version))
    assert len(candidates) == int(fallback)
    if fallback:
        assert candidates[0].snapshot.shards[0].worker_id == "trainer-replica"
        assert list(candidates[0].checksums.values()) == ["first"]


def test_unpublished_cached_selection_is_not_reused(protocol):
    first, = protocol.resolver.candidates(publish(protocol, "version-a", "first"))
    assert list(protocol.resolver.candidates(replace(protocol.version, version_id="version-b"))) == []


def test_resolver_retains_manifests_across_abandoned_selections(protocol):
    publish(protocol, "version-a", "first")
    alternate = refit_pb2.WeightVersionShard()
    alternate.CopyFrom(protocol.control.shards[0])
    alternate.worker_id = "trainer-replica"
    protocol.control.shards.append(alternate)
    candidates = list(protocol.resolver.candidates(protocol.version))
    assert [source.snapshot.shards[0].worker_id for source in candidates] == [
        "trainer-0", "trainer-replica",
    ]
    assert len(protocol.worker.requests) == 2
    for publication in (alternate, protocol.control.shards[0]):
        protocol.control.shards = [publication]
        current, = protocol.resolver.candidates(protocol.version)
        assert current.snapshot.shards[0].worker_id == publication.worker_id
        assert list(current.checksums.values()) == ["first"]
    assert len(protocol.worker.requests) == 2
