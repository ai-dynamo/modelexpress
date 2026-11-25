# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-domain cache discovery, receive ownership, and installation."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modelexpress import p2p_pb2
from modelexpress.metadata import nixl_artifact_transport as transport_module
from modelexpress.metadata.artifact_transport import (
    ArtifactCacheMiss,
    ArtifactTransportContext,
)


@pytest.fixture
def fetching(monkeypatch):
    monkeypatch.setenv("MX_P2P_TOPOLOGY_DOMAIN", "fabric")
    monkeypatch.setenv("MX_P2P_TOPOLOGY_ENFORCEMENT", "adaptive")
    monkeypatch.setenv("MX_P2P_TOPOLOGY", '{"fabric":"a"}')
    monkeypatch.setenv("MX_NIXL_BACKEND", "UCX")
    refs = [
        p2p_pb2.SourceInstanceRef(mx_source_id="cache", worker_id=name, topology=topology)
        for name, topology in [
            ("cross", {"fabric": "b"}), ("same", {"fabric": "a"}), ("unknown", {})
        ]
    ]
    client = MagicMock()
    client.list_sources.return_value = p2p_pb2.ListSourcesResponse(instances=refs)

    def metadata(source_id, worker_id):
        ref = next(ref for ref in refs if ref.worker_id == worker_id)
        return p2p_pb2.GetMetadataResponse(
            found=True,
            mx_source_id=source_id,
            worker_id=worker_id,
            worker=p2p_pb2.WorkerMetadata(
                status=p2p_pb2.SOURCE_STATUS_READY,
                topology=ref.topology, worker_grpc_endpoint=worker_id,
                artifact_source=p2p_pb2.ArtifactSourceMetadata(artifact_id=worker_id),
            ),
        )

    client.get_metadata.side_effect = metadata
    receivers = []

    def receiver(**kwargs):
        manager = MagicMock()
        manager.ucx_mode = kwargs.get("ucx_mode", "auto")
        receivers.append(manager)
        return manager

    monkeypatch.setattr(transport_module, "NixlTransferManager", receiver)
    transfer = MagicMock(
        return_value=p2p_pb2.GetArtifactManifestHeaderResponse(artifact_id="cache")
    )
    monkeypatch.setattr(transport_module, "transfer_artifact_from_worker", transfer)
    publication = MagicMock()
    transport = transport_module.NixlArtifactTransport(
        mx_client=client, nixl_manager=publication
    )
    artifact = SimpleNamespace(name="triton", target_file_paths=lambda: [])
    identity = p2p_pb2.SourceIdentity(mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE)
    return SimpleNamespace(
        client=client, refs=refs, transfer=transfer, receivers=receivers,
        publication=publication, transport=transport, artifact=artifact, identity=identity,
    )


def fetch(case):
    return case.transport.fetch(case.artifact, case.identity, ArtifactTransportContext())


@pytest.mark.parametrize("primary_succeeds", [True, False])
@pytest.mark.parametrize("domain", [None, "fabric"])
def test_one_list_primary_before_tcp_and_private_receivers(
    fetching, monkeypatch, primary_succeeds, domain
):
    if domain is None:
        monkeypatch.delenv("MX_P2P_TOPOLOGY_DOMAIN")

    def transfer(endpoint, source_id, artifact_id, manager, **kwargs):
        if not primary_succeeds and (manager.ucx_mode == "auto" or endpoint == "same"):
            raise ConnectionError("unreachable")
        return p2p_pb2.GetArtifactManifestHeaderResponse(artifact_id=artifact_id)

    fetching.transfer.side_effect = transfer
    result = fetch(fetching)
    attempts = [(c.args[0], c.args[3].ucx_mode) for c in fetching.transfer.call_args_list]
    primary_sources = ["same"] if domain else ["cross", "same", "unknown"]
    expected = [(primary_sources[0], "auto")] if primary_succeeds else (
        [(name, "auto") for name in primary_sources]
        + ([("same", "tcp")] if domain else []) + [("cross", "tcp")]
    )
    assert attempts == expected
    assert result.header.artifact_id == expected[-1][0]
    fetching.client.list_sources.assert_called_once()
    for receiver in fetching.receivers:
        receiver.shutdown.assert_called_once()
    fetching.publication.shutdown.assert_not_called()


def test_no_same_domain_source_goes_directly_to_tcp(fetching):
    fetching.refs[1].topology["fabric"] = "b"
    fetching.client.list_sources.return_value.instances[1].topology["fabric"] = "b"
    fetch(fetching)
    assert len(fetching.receivers) == 1
    assert fetching.receivers[0].ucx_mode == "tcp"
    fetching.client.list_sources.assert_called_once()


def test_each_channel_has_its_own_source_budget(fetching, monkeypatch):
    monkeypatch.setattr(transport_module, "MAX_SOURCE_RETRIES", 1)
    fetching.transfer.side_effect = ConnectionError("unreachable")
    with pytest.raises(ArtifactCacheMiss):
        fetch(fetching)
    attempts = [(c.args[0], c.args[3].ucx_mode) for c in fetching.transfer.call_args_list]
    assert attempts == [
        ("same", "auto"), ("same", "tcp")
    ]
    fetching.client.list_sources.assert_called_once()


def test_incompatible_node_rank_does_not_consume_receive_budget(fetching, monkeypatch):
    monkeypatch.setattr(transport_module, "MAX_SOURCE_RETRIES", 1)
    original = fetching.client.get_metadata.side_effect

    def metadata(source_id, worker_id):
        response = original(source_id, worker_id)
        response.worker.artifact_source.node_rank = 1 if worker_id == "same" else 0
        return response

    fetching.client.get_metadata.side_effect = metadata
    result = fetch(fetching)
    assert result.header.artifact_id == "cache"
    assert fetching.transfer.call_args.args[0] == "cross"
    assert fetching.receivers[0].ucx_mode == "tcp"


@pytest.mark.parametrize("unsupported", ["backend", "discovery"])
def test_unsupported_adaptive_transport_does_not_allocate_receiver(
    fetching, monkeypatch, unsupported
):
    if unsupported == "backend":
        monkeypatch.setenv("MX_NIXL_BACKEND", "LIBFABRIC")
    else:
        fetching.client.REQUIRES_P2P_METADATA = True
    with pytest.raises(ArtifactCacheMiss):
        fetch(fetching)
    fetching.client.list_sources.assert_not_called()
    assert not fetching.receivers


def test_receiver_shutdown_failure_stops_before_tcp(fetching):
    from modelexpress.metadata.artifact_transfer import ArtifactTransferCleanupError

    def fail_shutdown(*args, **kwargs):
        fetching.receivers[-1].shutdown.side_effect = RuntimeError("deregistration failed")
        raise ConnectionError("failed")

    fetching.transfer.side_effect = fail_shutdown
    with pytest.raises(ArtifactTransferCleanupError, match="close artifact receiver"):
        fetch(fetching)
    assert fetching.transfer.call_count == 1


@pytest.mark.parametrize("outcome", ["hit", "miss", "unsafe"])
def test_installation_only_completes_after_success_and_cleanup(
    fetching, monkeypatch, tmp_path, outcome
):
    from modelexpress.metadata import artifact_lifecycle as lifecycle
    from modelexpress.metadata.artifact_transfer import ArtifactTransferCleanupError
    from modelexpress.metadata.artifact_transport import ArtifactInstallStatus

    monkeypatch.setenv("MX_ARTIFACT_TRANSFER", "1")
    monkeypatch.setenv("MX_ARTIFACT_BACKEND", "nixl")
    monkeypatch.setattr(lifecycle, "artifact_lock_root", lambda: tmp_path)
    monkeypatch.setattr(
        lifecycle, "_create_artifact_transport", lambda *a: fetching.transport
    )
    fetching.artifact.roots = ()
    fetching.artifact.install = MagicMock()
    ctx = SimpleNamespace(
        global_rank=0, node_rank=0, accelerator_backend=SimpleNamespace(name="cuda")
    )
    if outcome == "miss":
        fetching.transfer.side_effect = ConnectionError("source unreachable")
    elif outcome == "unsafe":
        fetching.transfer.side_effect = ArtifactTransferCleanupError("READ still pending")

    def install():
        lifecycle.install_artifacts(
            ctx, lambda: [(fetching.artifact, fetching.identity)], engine_label="test"
        )

    if outcome == "unsafe":
        with pytest.raises(ArtifactTransferCleanupError, match="READ still pending"):
            install()
        assert fetching.transfer.call_count == 1
        fetching.receivers[0].shutdown.assert_not_called()
    else:
        install()
    marker = lifecycle.artifact_marker_path(
        fetching.artifact, fetching.identity, "install-attempted"
    )
    state = lifecycle._read_artifact_install_state(marker)
    assert state.status == (
        ArtifactInstallStatus.INSTALLED if outcome == "hit" else ArtifactInstallStatus.ATTEMPTED
    )
    assert fetching.artifact.install.call_count == (1 if outcome == "hit" else 0)
