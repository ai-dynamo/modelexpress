# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the transport abstraction and backend adapters."""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modelexpress import p2p_pb2
from modelexpress.metadata.artifact_transport import (
    ArtifactCacheMiss,
    ArtifactCacheStale,
    ArtifactInstallDisposition,
    ArtifactInstallState,
    ArtifactInstallStatus,
    ArtifactTransportContext,
    ArtifactTransportUnavailable,
)
from modelexpress.metadata.mooncake_artifact_transport import (
    MooncakeArtifactTransport,
)
from modelexpress.metadata.p2p_artifact_transport import P2PArtifactTransport


class _FakeP2PArtifact:
    name = "fake"
    mx_source_type = p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE
    roots = ()
    bundle_root = None

    def __init__(self):
        self.discovery_calls = []
        self.discovery_header = None
        self.discovery_error = None

    def prepare_source(self):
        raise NotImplementedError

    def install(self, header):
        raise NotImplementedError

    def target_file_paths(self):
        return []


@pytest.fixture(autouse=True)
def clear_mooncake_pending_publications():
    import modelexpress.metadata.mooncake_artifact_transport as module

    module._pending_publications.clear()
    yield
    module._pending_publications.clear()


def test_p2p_transport_fetch_keeps_discovery_inside_adapter(monkeypatch):
    import modelexpress.metadata.p2p_artifact_transport as module

    artifact = _FakeP2PArtifact()
    artifact.discovery_header = p2p_pb2.GetArtifactManifestHeaderResponse(
        artifact_id="p2p-artifact"
    )
    artifact.discovery_error = None
    identity = p2p_pb2.SourceIdentity()
    mx_client = object()
    nixl_manager = object()
    context = ArtifactTransportContext(
        node_rank=2,
        accelerator="cuda",
    )

    discover = MagicMock(
        return_value=SimpleNamespace(
            worker_grpc_endpoint="worker:1234",
            mx_source_id="source",
            artifact_id="p2p-artifact",
        )
    )
    transfer = MagicMock(return_value=artifact.discovery_header)
    monkeypatch.setattr(module, "discover_artifact_source", discover)
    monkeypatch.setattr(module, "transfer_artifact_from_worker", transfer)

    result = P2PArtifactTransport(
        mx_client=mx_client,
        nixl_manager=nixl_manager,
    ).fetch(artifact, identity, context)

    assert result.transport == "p2p"
    assert result.header.artifact_id == "p2p-artifact"
    discover.assert_called_once_with(
        mx_client, identity, worker_rank=None, node_rank=2, accelerator="cuda"
    )
    transfer.assert_called_once_with(
        "worker:1234",
        "source",
        "p2p-artifact",
        nixl_manager,
        target_file_paths=[],
    )


def test_p2p_transport_maps_discovery_miss(monkeypatch):
    import modelexpress.metadata.p2p_artifact_transport as module

    artifact = _FakeP2PArtifact()
    monkeypatch.setattr(
        module,
        "discover_artifact_source",
        MagicMock(side_effect=LookupError("no source")),
    )

    with pytest.raises(ArtifactCacheMiss, match="no source"):
        P2PArtifactTransport(
            mx_client=object(),
            nixl_manager=object(),
        ).fetch(
            artifact,
            p2p_pb2.SourceIdentity(),
            ArtifactTransportContext(),
        )


def test_mooncake_transport_maps_cache_miss(monkeypatch):
    import modelexpress.metadata.mooncake_artifact_transport as module

    monkeypatch.setattr(
        module,
        "install_from_mooncake",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            module.MooncakeArtifactCacheMiss("missing")
        ),
    )

    with pytest.raises(ArtifactCacheMiss, match="missing"):
        MooncakeArtifactTransport().fetch(
            _FakeP2PArtifact(),
            p2p_pb2.SourceIdentity(),
            ArtifactTransportContext(),
        )


def test_mooncake_transport_maps_stale_cache_separately(monkeypatch):
    import modelexpress.metadata.mooncake_artifact_transport as module

    monkeypatch.setattr(
        module,
        "install_from_mooncake",
        MagicMock(
            side_effect=module.MooncakeArtifactCacheStale(
                "checksum mismatch",
                cache_key="cache-key",
                fingerprint="stale-fingerprint",
            )
        ),
    )

    with pytest.raises(ArtifactCacheStale, match="checksum mismatch"):
        MooncakeArtifactTransport().fetch(
            _FakeP2PArtifact(),
            p2p_pb2.SourceIdentity(),
            ArtifactTransportContext(),
        )


def test_mooncake_transport_passes_stale_fingerprint_to_repair_publish(
    monkeypatch,
):
    import modelexpress.metadata.mooncake_artifact_transport as module

    artifact = _FakeP2PArtifact()
    identity = p2p_pb2.SourceIdentity()
    context = ArtifactTransportContext()
    stale = module.MooncakeArtifactCacheStale(
        "checksum mismatch",
        cache_key="cache-key",
        fingerprint="stale-fingerprint",
    )
    monkeypatch.setattr(module, "compute_artifact_cache_key", lambda *a, **k: "key")
    monkeypatch.setattr(
        module,
        "install_from_mooncake",
        MagicMock(side_effect=stale),
    )
    publish = MagicMock(return_value="key")
    monkeypatch.setattr(module, "publish_to_mooncake", publish)

    transport = MooncakeArtifactTransport()
    with pytest.raises(ArtifactCacheStale):
        transport.fetch(artifact, identity, context)
    handle = transport.publish(artifact, identity, MagicMock(), context)

    assert handle.identifier == "key"
    assert publish.call_args.kwargs["repair_from_fingerprint"] == (
        "stale-fingerprint"
    )
    assert not module._pending_publications


def test_mooncake_transport_plain_miss_uses_normal_publish(monkeypatch):
    import modelexpress.metadata.mooncake_artifact_transport as module

    artifact = _FakeP2PArtifact()
    identity = p2p_pb2.SourceIdentity()
    context = ArtifactTransportContext()
    monkeypatch.setattr(module, "compute_artifact_cache_key", lambda *a, **k: "key")
    monkeypatch.setattr(
        module,
        "install_from_mooncake",
        MagicMock(side_effect=module.MooncakeArtifactCacheMiss("missing")),
    )
    publish = MagicMock(return_value="key")
    monkeypatch.setattr(module, "publish_to_mooncake", publish)

    transport = MooncakeArtifactTransport()
    with pytest.raises(ArtifactCacheMiss):
        transport.fetch(artifact, identity, context)
    transport.publish(artifact, identity, MagicMock(), context)

    assert publish.call_args.kwargs["repair_from_fingerprint"] is None
    assert not module._pending_publications


def test_mooncake_transport_maps_operational_failure_to_unavailable(monkeypatch):
    import modelexpress.metadata.mooncake_artifact_transport as module

    monkeypatch.setattr(
        module,
        "install_from_mooncake",
        MagicMock(
            side_effect=module.MooncakeArtifactCacheUnavailable(
                "connecting to the Mooncake store failed"
            )
        ),
    )

    with pytest.raises(
        ArtifactTransportUnavailable,
        match="connecting to the Mooncake store failed",
    ):
        MooncakeArtifactTransport().fetch(
            _FakeP2PArtifact(),
            p2p_pb2.SourceIdentity(),
            ArtifactTransportContext(),
        )


def test_publish_retry_policy_is_transport_specific():
    assert P2PArtifactTransport.publish_requires_install_state is False
    assert MooncakeArtifactTransport.publish_requires_install_state is True
    assert P2PArtifactTransport.retry_publish_on_failure is True
    assert MooncakeArtifactTransport.retry_publish_on_failure is False


def test_mooncake_availability_check_does_not_import_native_module(monkeypatch):
    """Native configuration must be deferred until the store session."""
    import modelexpress.metadata.mooncake_artifact_transport as module

    calls = []

    def find_spec(fullname, path=None, target=None):
        calls.append((fullname, path))
        if fullname == "mooncake":
            spec = ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = ["/fake/mooncake"]
            return spec
        if fullname == "mooncake.store":
            return ModuleSpec(fullname, loader=object())
        return None

    monkeypatch.delitem(sys.modules, "mooncake", raising=False)
    monkeypatch.delitem(sys.modules, "mooncake.store", raising=False)
    monkeypatch.setattr(module.PathFinder, "find_spec", find_spec)

    assert MooncakeArtifactTransport.is_available()
    assert calls == [
        ("mooncake", None),
        ("mooncake.store", ["/fake/mooncake"]),
    ]
    assert "mooncake" not in sys.modules
    assert "mooncake.store" not in sys.modules


def test_p2p_lifecycle_policy_preserves_original_publish_behavior():
    transport = P2PArtifactTransport()
    attempted = transport.state_after_cache_miss()

    assert attempted == ArtifactInstallState(ArtifactInstallStatus.ATTEMPTED)
    assert transport.resolve_install_state(attempted) is ArtifactInstallDisposition.SKIP
    assert transport.should_publish(None)
    assert transport.should_publish(attempted)
    assert transport.should_publish(
        ArtifactInstallState(
            ArtifactInstallStatus.INSTALLED,
            artifact_id="artifact-id",
        )
    )


def test_mooncake_lifecycle_policy_shares_a_live_miss():
    transport = MooncakeArtifactTransport()
    miss = transport.state_after_cache_miss()

    assert miss.status is ArtifactInstallStatus.MISS
    assert miss.transport == "mooncake"
    assert miss.owner_pid is not None
    assert transport.resolve_install_state(miss) is (
        ArtifactInstallDisposition.CACHED_MISS
    )
    assert transport.should_publish(miss)


def test_mooncake_lifecycle_policy_retries_a_stale_miss():
    transport = MooncakeArtifactTransport()
    miss = ArtifactInstallState(
        ArtifactInstallStatus.MISS,
        transport="mooncake",
        owner_pid=999_999_999,
        owner_starttime="1",
    )

    assert transport.resolve_install_state(miss) is ArtifactInstallDisposition.FETCH
    assert not transport.should_publish(miss)


def test_mooncake_lifecycle_policy_does_not_publish_another_process_miss(
    monkeypatch,
):
    import modelexpress.metadata.mooncake_artifact_transport as module

    miss = ArtifactInstallState(
        ArtifactInstallStatus.MISS,
        transport="mooncake",
        owner_pid=999_999_999,
        owner_starttime="1",
    )
    monkeypatch.setattr(module, "_state_owner_is_alive", lambda _state: True)

    assert not MooncakeArtifactTransport().should_publish(miss)


@pytest.mark.parametrize(
    "state",
    [
        None,
        ArtifactInstallState(ArtifactInstallStatus.ATTEMPTED),
        ArtifactInstallState(
            ArtifactInstallStatus.INSTALLED,
            artifact_id="artifact-id",
        ),
    ],
)
def test_mooncake_lifecycle_policy_does_not_publish_without_a_miss(state):
    assert not MooncakeArtifactTransport().should_publish(state)
