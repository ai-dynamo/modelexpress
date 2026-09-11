# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the observability contract of ``metadata.artifact_lifecycle``.

An artifact miss is the difference between a warm start and a full recompile,
so the install path has to say which identity it looked for. These tests pin
the identifying fields that an operator needs in order to diff two pods that
fail to pair: the ``mx_source_id`` and the ``compile_config_digest`` that feeds
it.
"""

import logging
import multiprocessing
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from modelexpress import p2p_pb2
from modelexpress.metadata import artifact_lifecycle
from modelexpress.metadata.artifact_transfer import ArtifactCacheRoot
from modelexpress.metadata.artifact_transport import (
    ArtifactCacheMiss,
    ArtifactCacheStale,
    ArtifactInstallState,
    ArtifactInstallStatus,
    ArtifactTransportUnavailable,
)
from modelexpress.metadata.mooncake_artifact_transport import MooncakeArtifactTransport
from modelexpress.metadata.source_id import compute_mx_source_id

LOGGER_NAME = "modelexpress.metadata.artifact_lifecycle"


@pytest.fixture
def artifact_transfer_enabled(monkeypatch):
    monkeypatch.setenv("MX_ARTIFACT_TRANSFER", "1")
    monkeypatch.setenv("MX_P2P_METADATA", "1")


def _identity(digest: str = "") -> p2p_pb2.SourceIdentity:
    return p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        compile_config_digest=digest,
    )


def _run_p2p_install_owner(tmp_dir, started, release):
    """Run one real install in a child process for lease coordination tests."""
    artifact_lifecycle.tempfile.gettempdir = lambda: str(tmp_dir)
    target_root = tmp_dir / "cache"
    header = p2p_pb2.GetArtifactManifestHeaderResponse(artifact_id="artifact-id")

    def discover(*_args, **_kwargs):
        started.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test owner was not released")
        return header

    def install(_header):
        target_root.mkdir(parents=True, exist_ok=True)
        (target_root / "cached").write_text("ready")

    transfer = SimpleNamespace(
        name="torch_compile_cache",
        roots=(
            ArtifactCacheRoot(
                name="primary",
                source_root=target_root,
                target_root=target_root,
            ),
        ),
        discover_and_transfer=discover,
        install=install,
    )
    ctx = SimpleNamespace(
        mx_client=object(),
        nixl_manager=object(),
        node_rank=0,
        accelerator_backend=SimpleNamespace(name="cuda"),
    )
    artifact_lifecycle.install_artifact_once(
        ctx, transfer, _identity(), engine_label="vLLM"
    )


def _crash_during_p2p_install(tmp_dir, started):
    """Terminate while holding the real install lease."""
    artifact_lifecycle.tempfile.gettempdir = lambda: str(tmp_dir)
    target_root = tmp_dir / "cache"

    def discover(*_args, **_kwargs):
        started.set()
        os._exit(0)

    transfer = SimpleNamespace(
        name="torch_compile_cache",
        roots=(
            ArtifactCacheRoot(
                name="primary",
                source_root=target_root,
                target_root=target_root,
            ),
        ),
        discover_and_transfer=discover,
        install=lambda _header: None,
    )
    ctx = SimpleNamespace(
        mx_client=object(),
        nixl_manager=object(),
        node_rank=0,
        accelerator_backend=SimpleNamespace(name="cuda"),
    )
    artifact_lifecycle.install_artifact_once(
        ctx, transfer, _identity(), engine_label="vLLM"
    )


def _read_shared_mooncake_miss(tmp_dir, result):
    artifact_lifecycle.tempfile.gettempdir = lambda: str(tmp_dir)
    transfer = SimpleNamespace(
        name="triton_cache",
        roots=(
            ArtifactCacheRoot(
                "primary",
                tmp_dir / "source",
                tmp_dir / "target",
            ),
        ),
    )
    marker_path = artifact_lifecycle.artifact_marker_path(
        transfer, _identity(), "install-attempted"
    )
    state = artifact_lifecycle._read_artifact_install_state(marker_path)
    transport = MooncakeArtifactTransport()
    result.put(
        (
            state.status.value,
            transport.resolve_install_state(state).value,
            transport.should_publish(state),
        )
    )


def _run_install(identity, *, install_result):
    """Drive ``install_artifacts`` past its guards with a stubbed install."""
    ctx = SimpleNamespace(
        global_rank=0,
        device_id=0,
        nixl_manager=object(),
        mx_client=object(),
    )
    transfer = SimpleNamespace(name="torch_compile_cache")
    if "return_value" in install_result:
        install_result = {
            **install_result,
            "return_value": (install_result["return_value"], "p2p"),
        }

    transport = MagicMock()
    with patch.object(
        artifact_lifecycle, "_metadata_publication_configured", return_value=True
    ), patch.object(
        artifact_lifecycle, "is_nixl_available", return_value=True
    ), patch.object(
        artifact_lifecycle, "_create_artifact_transport", return_value=transport
    ), patch.object(
        artifact_lifecycle, "_install_artifact_via_transports", **install_result
    ):
        artifact_lifecycle.install_artifacts(
            ctx,
            lambda: [(transfer, identity)],
            engine_label="vLLM",
        )


def test_artifact_miss_is_logged_at_info_with_the_identity_it_looked_for(
    artifact_transfer_enabled,
    caplog,
):
    identity = _identity("vllmcfg1-deadbeef")

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        _run_install(
            identity,
            install_result={
                "side_effect": artifact_lifecycle.ArtifactCacheMiss("no source")
            },
        )

    records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "No remote vLLM artifact available" in message
    assert "backend=p2p" in message
    assert compute_mx_source_id(identity) in message
    assert "vllmcfg1-deadbeef" in message


def test_artifact_miss_reports_an_empty_digest_distinguishably(
    artifact_transfer_enabled,
    caplog,
):
    """An unset digest is the default and the cause of cross-config pairing.

    ``%r`` keeps the empty string visible rather than rendering a blank gap.
    """
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        _run_install(
            _identity(),
            install_result={"side_effect": artifact_lifecycle.ArtifactCacheMiss()},
        )

    assert "compile_config_digest=''" in caplog.text


def test_stale_artifact_is_logged_once_at_warning(
    artifact_transfer_enabled,
    caplog,
):
    identity = _identity("vllmcfg1-stale")

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        _run_install(
            identity,
            install_result={
                "side_effect": ArtifactCacheStale(
                    "checksum mismatch; manifest_invalidation=complete"
                )
            },
        )

    records = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "Remote vLLM artifact is stale" in message
    assert "checksum mismatch" in message
    assert "compile_config_digest='vllmcfg1-stale'" in message
    assert "rebuild and republish" in message


def test_successful_install_logs_the_mx_source_id(
    artifact_transfer_enabled,
    caplog,
):
    identity = _identity("vllmcfg1-deadbeef")
    header = SimpleNamespace(artifact_id="artifact-id", total_size=30 * 1024 * 1024)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        _run_install(identity, install_result={"return_value": header})

    message = caplog.text
    assert "artifact install complete" in message
    assert f"mx_source_id={compute_mx_source_id(identity)}" in message
    assert "artifact_id=artifact-id" in message


def test_identities_differing_only_in_digest_get_different_source_ids():
    """Regression guard for the pairing rule the install log now exposes.

    Two workers whose compile configuration differs must not share an artifact
    source pool. ``compile_config_digest`` is the field that separates them, so
    it has to reach ``mx_source_id``.
    """
    assert compute_mx_source_id(_identity("vllmcfg1-aaaa")) != compute_mx_source_id(
        _identity("vllmcfg1-bbbb")
    )
    assert compute_mx_source_id(_identity()) != compute_mx_source_id(
        _identity("vllmcfg1-aaaa")
    )


def test_install_artifact_once_calls_completion_hook(monkeypatch, tmp_path):
    monkeypatch.setattr(
        artifact_lifecycle.tempfile, "gettempdir", lambda: str(tmp_path)
    )
    target_root = tmp_path / "cache"
    header = p2p_pb2.GetArtifactManifestHeaderResponse(artifact_id="artifact-id")
    transfer = SimpleNamespace(
        name="torch_compile_cache",
        roots=(
            ArtifactCacheRoot(
                name="primary",
                source_root=target_root,
                target_root=target_root,
            ),
        ),
        discover_and_transfer=MagicMock(return_value=header),
        install=MagicMock(),
    )
    ctx = SimpleNamespace(
        mx_client=object(),
        nixl_manager=object(),
        node_rank=0,
        accelerator_backend=SimpleNamespace(name="cuda"),
    )
    on_install_completed = MagicMock()

    result = artifact_lifecycle.install_artifact_once(
        ctx,
        transfer,
        _identity(),
        engine_label="vLLM",
        on_install_completed=on_install_completed,
    )

    assert result is header
    transfer.install.assert_called_once_with(header)
    on_install_completed.assert_called_once_with(transfer, _identity())


def test_install_artifact_once_reuses_success_marker_with_target_files(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        artifact_lifecycle.tempfile, "gettempdir", lambda: str(tmp_path)
    )
    target_root = tmp_path / "cache"
    header = p2p_pb2.GetArtifactManifestHeaderResponse(artifact_id="artifact-id")
    transfer = SimpleNamespace(
        name="torch_compile_cache",
        roots=(
            ArtifactCacheRoot(
                name="primary",
                source_root=target_root,
                target_root=target_root,
            ),
        ),
        discover_and_transfer=MagicMock(return_value=header),
        install=MagicMock(
            side_effect=lambda _header: (
                target_root.mkdir(parents=True, exist_ok=True),
                (target_root / "cached").write_text("ready"),
            )
        ),
    )
    ctx = SimpleNamespace(
        mx_client=object(),
        nixl_manager=object(),
        node_rank=0,
        accelerator_backend=SimpleNamespace(name="cuda"),
    )

    assert (
        artifact_lifecycle.install_artifact_once(
            ctx, transfer, _identity(), engine_label="vLLM"
        )
        is header
    )
    assert (
        artifact_lifecycle.install_artifact_once(
            ctx, transfer, _identity(), engine_label="vLLM"
        )
        is None
    )
    transfer.discover_and_transfer.assert_called_once()
    transfer.install.assert_called_once_with(header)


def test_install_artifact_once_skips_existing_attempted_marker(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        artifact_lifecycle.tempfile, "gettempdir", lambda: str(tmp_path)
    )
    target_root = tmp_path / "cache"
    header = p2p_pb2.GetArtifactManifestHeaderResponse(artifact_id="artifact-id")
    transfer = SimpleNamespace(
        name="torch_compile_cache",
        roots=(
            ArtifactCacheRoot(
                name="primary",
                source_root=target_root,
                target_root=target_root,
            ),
        ),
        discover_and_transfer=MagicMock(return_value=header),
        install=MagicMock(
            side_effect=lambda _header: (
                target_root.mkdir(parents=True, exist_ok=True),
                (target_root / "cached").write_text("ready"),
            )
        ),
    )
    marker_path = artifact_lifecycle.artifact_marker_path(
        transfer, _identity(), "install-attempted"
    )
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text("attempted\n")
    ctx = SimpleNamespace(
        mx_client=object(),
        nixl_manager=object(),
        node_rank=0,
        accelerator_backend=SimpleNamespace(name="cuda"),
    )

    result = artifact_lifecycle.install_artifact_once(
        ctx, transfer, _identity(), engine_label="vLLM"
    )

    assert result is None
    transfer.discover_and_transfer.assert_not_called()
    assert marker_path.read_text().strip() == "attempted"


def test_install_artifact_once_keeps_marker_after_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        artifact_lifecycle.tempfile, "gettempdir", lambda: str(tmp_path)
    )
    target_root = tmp_path / "cache"
    transfer = SimpleNamespace(
        name="torch_compile_cache",
        roots=(
            ArtifactCacheRoot(
                name="primary",
                source_root=target_root,
                target_root=target_root,
            ),
        ),
        discover_and_transfer=MagicMock(side_effect=RuntimeError("transfer failed")),
        install=MagicMock(),
    )
    identity = _identity()
    marker_path = artifact_lifecycle.artifact_marker_path(
        transfer, identity, "install-attempted"
    )
    ctx = SimpleNamespace(
        mx_client=object(),
        nixl_manager=object(),
        node_rank=0,
        accelerator_backend=SimpleNamespace(name="cuda"),
    )

    with pytest.raises(RuntimeError, match="transfer failed"):
        artifact_lifecycle.install_artifact_once(
            ctx, transfer, identity, engine_label="vLLM"
        )

    assert marker_path.read_text().strip() == "attempted"


def test_install_artifact_lease_waits_and_rechecks_success(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        artifact_lifecycle.tempfile, "gettempdir", lambda: str(tmp_path)
    )
    context = multiprocessing.get_context("fork")
    started = context.Event()
    release = context.Event()
    owner = context.Process(
        target=_run_p2p_install_owner,
        args=(tmp_path, started, release),
    )
    owner.start()
    result = []
    errors = []
    try:
        assert started.wait(timeout=5)
        target_root = tmp_path / "cache"
        transfer = SimpleNamespace(
            name="torch_compile_cache",
            roots=(
                ArtifactCacheRoot(
                    name="primary",
                    source_root=target_root,
                    target_root=target_root,
                ),
            ),
            discover_and_transfer=MagicMock(),
            install=MagicMock(),
        )
        ctx = SimpleNamespace(
            mx_client=object(),
            nixl_manager=object(),
            node_rank=0,
            accelerator_backend=SimpleNamespace(name="cuda"),
        )

        def contender():
            try:
                result.append(
                    artifact_lifecycle.install_artifact_once(
                        ctx, transfer, _identity(), engine_label="vLLM"
                    )
                )
            except BaseException as exc:  # pragma: no cover - assertion below
                errors.append(exc)

        thread = threading.Thread(target=contender)
        thread.start()
        time.sleep(0.1)
        assert thread.is_alive()

        release.set()
        owner.join(timeout=5)
        thread.join(timeout=5)

        assert owner.exitcode == 0
        assert not thread.is_alive()
        assert not errors
        assert result == [None]
        transfer.discover_and_transfer.assert_not_called()
        transfer.install.assert_not_called()
    finally:
        release.set()
        if owner.is_alive():
            owner.terminate()
        owner.join(timeout=5)


def test_install_artifact_attempt_marker_survives_owner_crash(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        artifact_lifecycle.tempfile, "gettempdir", lambda: str(tmp_path)
    )
    context = multiprocessing.get_context("fork")
    started = context.Event()
    owner = context.Process(
        target=_crash_during_p2p_install,
        args=(tmp_path, started),
    )
    owner.start()
    assert started.wait(timeout=5)
    owner.join(timeout=5)
    assert owner.exitcode == 0

    target_root = tmp_path / "cache"
    header = p2p_pb2.GetArtifactManifestHeaderResponse(artifact_id="artifact-id")
    transfer = SimpleNamespace(
        name="torch_compile_cache",
        roots=(
            ArtifactCacheRoot(
                name="primary",
                source_root=target_root,
                target_root=target_root,
            ),
        ),
        discover_and_transfer=MagicMock(return_value=header),
        install=MagicMock(
            side_effect=lambda _header: (
                target_root.mkdir(parents=True, exist_ok=True),
                (target_root / "cached").write_text("ready"),
            )
        ),
    )
    ctx = SimpleNamespace(
        mx_client=object(),
        nixl_manager=object(),
        node_rank=0,
        accelerator_backend=SimpleNamespace(name="cuda"),
    )

    result = artifact_lifecycle.install_artifact_once(
        ctx, transfer, _identity(), engine_label="vLLM"
    )

    assert result is None
    transfer.discover_and_transfer.assert_not_called()
    transfer.install.assert_not_called()


def test_unified_install_marker_is_backend_neutral():
    """Both transports use the same lifecycle marker contract."""
    transfer = SimpleNamespace(name="cache", roots=())
    assert artifact_lifecycle.artifact_marker_path(
        transfer, _identity(), "install-attempted"
    ).name.startswith("install-attempted-")


def test_p2p_backend_does_not_check_mooncake_availability(monkeypatch):
    monkeypatch.setenv("MX_ARTIFACT_BACKEND", "p2p")
    mooncake_check = MagicMock(side_effect=AssertionError("Mooncake was checked"))
    monkeypatch.setattr(
        artifact_lifecycle.MooncakeArtifactTransport,
        "is_available",
        mooncake_check,
    )
    monkeypatch.setattr(
        artifact_lifecycle, "_p2p_artifact_install_available", lambda *args: True
    )
    ctx = SimpleNamespace(
        global_rank=0,
        mx_client=object(),
        nixl_manager=object(),
        worker_id="worker",
        worker_rank=0,
        device_id=0,
    )

    transport = artifact_lifecycle._create_artifact_transport(
        ctx, "p2p", "vLLM", MagicMock()
    )

    assert isinstance(transport, artifact_lifecycle.P2PArtifactTransport)
    mooncake_check.assert_not_called()


def test_unavailable_mooncake_backend_skips_install_and_publish(
    monkeypatch, caplog
):
    monkeypatch.setenv("MX_ARTIFACT_TRANSFER", "1")
    monkeypatch.setenv("MX_ARTIFACT_BACKEND", "mooncake")
    monkeypatch.setattr(
        artifact_lifecycle.MooncakeArtifactTransport,
        "is_available",
        MagicMock(return_value=False),
    )
    ctx = SimpleNamespace(global_rank=0)
    transfers_factory = MagicMock()
    scheduled_publishers = {}

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        artifact_lifecycle.install_artifacts(
            ctx, transfers_factory, engine_label="vLLM"
        )
        artifact_lifecycle.schedule_artifact_publish(
            ctx,
            transfers_factory,
            engine_label="vLLM",
            ready_fn_factory=MagicMock(),
            artifact_publish_fn=MagicMock(),
            scheduled_publishers=scheduled_publishers,
        )

    transfers_factory.assert_not_called()
    assert caplog.text.count("Mooncake backend is unavailable") == 2
    assert scheduled_publishers == {}


def test_every_device_participates_in_mooncake_lifecycle(monkeypatch):
    monkeypatch.setenv("MX_ARTIFACT_TRANSFER", "1")
    monkeypatch.setenv("MX_ARTIFACT_BACKEND", "mooncake")
    monkeypatch.setattr(
        artifact_lifecycle.MooncakeArtifactTransport,
        "is_available",
        MagicMock(return_value=True),
    )
    ctx = SimpleNamespace(
        global_rank=1,
        worker_rank=1,
        device_id=1,
        accelerator_backend=SimpleNamespace(name="cuda"),
    )
    transfers_factory = MagicMock(return_value=[])
    scheduled_publishers = {}

    artifact_lifecycle.install_artifacts(
        ctx, transfers_factory, engine_label="vLLM"
    )
    artifact_lifecycle.schedule_artifact_publish(
        ctx,
        transfers_factory,
        engine_label="vLLM",
        ready_fn_factory=MagicMock(),
        artifact_publish_fn=MagicMock(),
        scheduled_publishers=scheduled_publishers,
    )

    assert transfers_factory.call_count == 2
    assert scheduled_publishers == {}


def test_install_miss_is_deduplicated_for_one_pod(monkeypatch, tmp_path):
    monkeypatch.setenv("POD_UID", "pod-under-test")
    monkeypatch.setenv("MX_ARTIFACT_BACKEND", "mooncake")
    monkeypatch.setattr(
        artifact_lifecycle.tempfile, "gettempdir", lambda: str(tmp_path)
    )
    transfer = SimpleNamespace(
        name="triton_cache",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
        roots=(
            ArtifactCacheRoot(
                "primary",
                tmp_path / "source",
                tmp_path / "target",
            ),
        ),
        install=MagicMock(),
    )
    transport = MooncakeArtifactTransport()
    transport.fetch = MagicMock(side_effect=ArtifactCacheMiss("missing"))
    monkeypatch.setattr(
        artifact_lifecycle,
        "_create_artifact_transport",
        lambda *args, **kwargs: transport,
    )
    ctx = SimpleNamespace(
        mx_client=object(),
        nixl_manager=object(),
        node_rank=0,
        accelerator_backend=SimpleNamespace(name="cuda"),
    )

    with pytest.raises(ArtifactCacheMiss):
        artifact_lifecycle.install_artifact_once(
            ctx, transfer, _identity(), engine_label="vLLM"
        )
    with pytest.raises(ArtifactCacheMiss, match="cached mooncake miss"):
        artifact_lifecycle.install_artifact_once(
            ctx, transfer, _identity(), engine_label="vLLM"
        )

    transport.fetch.assert_called_once()
    marker_path = artifact_lifecycle.artifact_marker_path(
        transfer, _identity(), "install-attempted"
    )
    state = artifact_lifecycle._read_artifact_install_state(marker_path)
    assert state.status is ArtifactInstallStatus.MISS
    assert state.transport == "mooncake"
    assert state.owner_pid == os.getpid()


def test_mooncake_operational_failure_is_attempted_once_and_not_published(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setenv("MX_ARTIFACT_TRANSFER", "1")
    monkeypatch.setenv("MX_ARTIFACT_BACKEND", "mooncake")
    monkeypatch.setattr(
        artifact_lifecycle.tempfile, "gettempdir", lambda: str(tmp_path)
    )
    transfer = SimpleNamespace(
        name="triton_cache",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
        roots=(
            ArtifactCacheRoot(
                "primary",
                tmp_path / "source",
                tmp_path / "target",
            ),
        ),
        install=MagicMock(),
    )
    identity = _identity()
    transport = MooncakeArtifactTransport()
    transport.fetch = MagicMock(
        side_effect=ArtifactTransportUnavailable("Mooncake setup failed")
    )
    monkeypatch.setattr(
        artifact_lifecycle,
        "_create_artifact_transport",
        lambda *args, **kwargs: transport,
    )
    publish_marker = MagicMock()
    monkeypatch.setattr(
        artifact_lifecycle, "mark_publish_scheduled", publish_marker
    )
    ctx = SimpleNamespace(
        global_rank=0,
        mx_client=object(),
        nixl_manager=object(),
        node_rank=0,
        accelerator_backend=SimpleNamespace(name="cuda"),
    )
    transfers_factory = MagicMock(return_value=[(transfer, identity)])

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        artifact_lifecycle.install_artifacts(
            ctx, transfers_factory, engine_label="vLLM"
        )
        artifact_lifecycle.install_artifacts(
            ctx, transfers_factory, engine_label="vLLM"
        )

    marker_path = artifact_lifecycle.artifact_marker_path(
        transfer, identity, "install-attempted"
    )
    assert marker_path.read_text().strip() == "attempted"
    assert caplog.text.count("Failed to install vLLM artifact triton_cache") == 1
    transport.fetch.assert_called_once()
    transfer.install.assert_not_called()

    artifact_lifecycle.schedule_artifact_publish(
        ctx,
        transfers_factory,
        engine_label="vLLM",
        ready_fn_factory=MagicMock(),
        artifact_publish_fn=MagicMock(),
        scheduled_publishers={},
    )

    publish_marker.assert_not_called()


def test_stale_mooncake_artifact_records_a_publishable_miss(monkeypatch, tmp_path):
    monkeypatch.setenv("MX_ARTIFACT_TRANSFER", "1")
    monkeypatch.setenv("MX_ARTIFACT_BACKEND", "mooncake")
    monkeypatch.setattr(
        artifact_lifecycle.tempfile, "gettempdir", lambda: str(tmp_path)
    )
    transfer = SimpleNamespace(
        name="triton_cache",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
        roots=(ArtifactCacheRoot("primary", tmp_path / "source", tmp_path / "target"),),
        install=MagicMock(),
    )
    identity = _identity()
    transport = MooncakeArtifactTransport()
    transport.fetch = MagicMock(side_effect=ArtifactCacheStale("corrupt chunk"))
    monkeypatch.setattr(
        artifact_lifecycle,
        "_create_artifact_transport",
        lambda *args, **kwargs: transport,
    )
    ctx = SimpleNamespace(
        global_rank=0,
        mx_client=object(),
        nixl_manager=object(),
        node_rank=0,
        accelerator_backend=SimpleNamespace(name="cuda"),
    )

    artifact_lifecycle.install_artifacts(
        ctx,
        lambda: [(transfer, identity)],
        engine_label="vLLM",
    )

    marker_path = artifact_lifecycle.artifact_marker_path(
        transfer, identity, "install-attempted"
    )
    state = artifact_lifecycle._read_artifact_install_state(marker_path)
    assert state.status is ArtifactInstallStatus.MISS
    assert state.transport == "mooncake"
    assert state.owner_pid == os.getpid()
    assert transport.should_publish(state)


def test_mooncake_miss_is_visible_but_not_publishable_by_another_process(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        artifact_lifecycle.tempfile, "gettempdir", lambda: str(tmp_path)
    )
    transfer = SimpleNamespace(
        name="triton_cache",
        roots=(
            ArtifactCacheRoot(
                "primary",
                tmp_path / "source",
                tmp_path / "target",
            ),
        ),
    )
    marker_path = artifact_lifecycle.artifact_marker_path(
        transfer, _identity(), "install-attempted"
    )
    artifact_lifecycle._write_artifact_install_state(
        marker_path, MooncakeArtifactTransport().state_after_cache_miss()
    )
    context = multiprocessing.get_context("fork")
    result = context.Queue()
    reader = context.Process(
        target=_read_shared_mooncake_miss,
        args=(tmp_path, result),
    )

    reader.start()
    reader.join(timeout=5)

    assert reader.exitcode == 0
    assert result.get(timeout=1) == ("miss", "cached_miss", False)


def test_p2p_miss_preserves_original_attempted_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("MX_ARTIFACT_BACKEND", "p2p")
    monkeypatch.setattr(
        artifact_lifecycle.tempfile, "gettempdir", lambda: str(tmp_path)
    )
    transfer = SimpleNamespace(
        name="triton_cache",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
        roots=(ArtifactCacheRoot("primary", tmp_path / "source", tmp_path / "target"),),
        install=MagicMock(),
    )
    transport = artifact_lifecycle.P2PArtifactTransport()
    transport.fetch = MagicMock(side_effect=ArtifactCacheMiss("missing"))
    monkeypatch.setattr(
        artifact_lifecycle,
        "_create_artifact_transport",
        lambda *args, **kwargs: transport,
    )
    ctx = SimpleNamespace(
        mx_client=object(),
        nixl_manager=object(),
        node_rank=0,
        accelerator_backend=SimpleNamespace(name="cuda"),
    )

    with pytest.raises(ArtifactCacheMiss):
        artifact_lifecycle.install_artifact_once(
            ctx, transfer, _identity(), engine_label="vLLM"
        )
    assert (
        artifact_lifecycle.install_artifact_once(
            ctx, transfer, _identity(), engine_label="vLLM"
        )
        is None
    )

    marker_path = artifact_lifecycle.artifact_marker_path(
        transfer, _identity(), "install-attempted"
    )
    assert marker_path.read_text().strip() == "attempted"
    transport.fetch.assert_called_once()


def test_stale_mooncake_miss_is_retried(monkeypatch, tmp_path):
    monkeypatch.setenv("MX_ARTIFACT_BACKEND", "mooncake")
    monkeypatch.setattr(
        artifact_lifecycle.tempfile, "gettempdir", lambda: str(tmp_path)
    )
    target_root = tmp_path / "target"
    transfer = SimpleNamespace(
        name="triton_cache",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
        roots=(ArtifactCacheRoot("primary", tmp_path / "source", target_root),),
        install=MagicMock(),
    )
    identity = _identity()
    marker_path = artifact_lifecycle.artifact_marker_path(
        transfer, identity, "install-attempted"
    )
    artifact_lifecycle._write_artifact_install_state(
        marker_path,
        ArtifactInstallState(
            ArtifactInstallStatus.MISS,
            transport="mooncake",
            owner_pid=999_999_999,
            owner_starttime="1",
        ),
    )
    header = p2p_pb2.GetArtifactManifestHeaderResponse(artifact_id="artifact-id")
    transport = MooncakeArtifactTransport()
    transport.fetch = MagicMock(
        return_value=SimpleNamespace(header=header, transport="mooncake")
    )
    monkeypatch.setattr(
        artifact_lifecycle,
        "_create_artifact_transport",
        lambda *args, **kwargs: transport,
    )
    ctx = SimpleNamespace(
        mx_client=object(),
        nixl_manager=object(),
        node_rank=0,
        accelerator_backend=SimpleNamespace(name="cuda"),
    )

    result = artifact_lifecycle.install_artifact_once(
        ctx, transfer, identity, engine_label="vLLM"
    )

    assert result is header
    transport.fetch.assert_called_once()
    transfer.install.assert_called_once_with(header)
    assert marker_path.read_text().strip() == "artifact-id"


def test_shared_mooncake_miss_can_be_published_by_the_querying_process(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MX_ARTIFACT_TRANSFER", "1")
    monkeypatch.setenv("MX_ARTIFACT_BACKEND", "mooncake")
    monkeypatch.setattr(
        artifact_lifecycle.tempfile, "gettempdir", lambda: str(tmp_path)
    )
    transfer = SimpleNamespace(
        name="triton_cache",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
        roots=(ArtifactCacheRoot("primary", tmp_path / "source", tmp_path / "target"),),
    )
    identity = _identity()
    querying_transport = MooncakeArtifactTransport()
    marker_path = artifact_lifecycle.artifact_marker_path(
        transfer, identity, "install-attempted"
    )
    artifact_lifecycle._write_artifact_install_state(
        marker_path, querying_transport.state_after_cache_miss()
    )
    publishing_transport = MooncakeArtifactTransport()
    monkeypatch.setattr(
        artifact_lifecycle,
        "_create_artifact_transport",
        lambda *args, **kwargs: publishing_transport,
    )
    publish_marker = MagicMock(return_value=tmp_path / "publish.done")
    monkeypatch.setattr(
        artifact_lifecycle, "mark_publish_scheduled", publish_marker
    )

    class FakePublisher:
        def __init__(self, **kwargs):
            self.mx_source_id = None
            self.kwargs = kwargs

        def start(self):
            return None

        def stop(self):
            return None

    monkeypatch.setattr(artifact_lifecycle, "PublisherThread", FakePublisher)
    scheduled = {}
    ctx = _publish_context()

    artifact_lifecycle.schedule_artifact_publish(
        ctx,
        lambda: [(transfer, identity)],
        engine_label="vLLM",
        ready_fn_factory=lambda _roots: lambda: True,
        artifact_publish_fn=MagicMock(),
        scheduled_publishers=scheduled,
    )

    publish_marker.assert_called_once_with(ctx, transfer, identity)
    assert len(scheduled) == 1
    assert next(iter(scheduled.values())).kwargs["retry_publish_on_failure"] is False


def test_p2p_schedule_preserves_publish_retries(monkeypatch, tmp_path):
    monkeypatch.setenv("MX_ARTIFACT_TRANSFER", "1")
    monkeypatch.setenv("MX_ARTIFACT_BACKEND", "p2p")
    transfer = SimpleNamespace(
        name="triton_cache",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
        roots=(),
    )
    transport = artifact_lifecycle.P2PArtifactTransport()
    monkeypatch.setattr(
        artifact_lifecycle,
        "_create_artifact_transport",
        lambda *args, **kwargs: transport,
    )
    monkeypatch.setattr(
        artifact_lifecycle,
        "mark_publish_scheduled",
        lambda *args, **kwargs: tmp_path / "publish.done",
    )
    install_lock = MagicMock(
        side_effect=AssertionError("P2P publication must not read install state")
    )
    monkeypatch.setattr(artifact_lifecycle, "artifact_lock", install_lock)
    publisher = SimpleNamespace(
        mx_source_id=None,
        start=MagicMock(),
        stop=MagicMock(),
    )
    publisher_type = MagicMock(return_value=publisher)
    monkeypatch.setattr(artifact_lifecycle, "PublisherThread", publisher_type)

    artifact_lifecycle.schedule_artifact_publish(
        _publish_context(),
        lambda: [(transfer, _identity())],
        engine_label="vLLM",
        ready_fn_factory=lambda _roots: lambda: True,
        artifact_publish_fn=MagicMock(),
        scheduled_publishers={},
    )

    assert publisher_type.call_args.kwargs["retry_publish_on_failure"] is True
    install_lock.assert_not_called()


def test_mooncake_hit_does_not_compete_for_publish_lease(monkeypatch, tmp_path):
    monkeypatch.setenv("MX_ARTIFACT_TRANSFER", "1")
    monkeypatch.setenv("MX_ARTIFACT_BACKEND", "mooncake")
    monkeypatch.setattr(
        artifact_lifecycle.tempfile, "gettempdir", lambda: str(tmp_path)
    )
    transfer = SimpleNamespace(
        name="triton_cache",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
        roots=(ArtifactCacheRoot("primary", tmp_path / "source", tmp_path / "target"),),
    )
    identity = _identity()
    marker_path = artifact_lifecycle.artifact_marker_path(
        transfer, identity, "install-attempted"
    )
    artifact_lifecycle._write_artifact_install_state(
        marker_path,
        ArtifactInstallState(
            ArtifactInstallStatus.INSTALLED,
            artifact_id="artifact-id",
        ),
    )
    monkeypatch.setattr(
        artifact_lifecycle,
        "_create_artifact_transport",
        lambda *args, **kwargs: MooncakeArtifactTransport(),
    )
    publish_marker = MagicMock()
    monkeypatch.setattr(
        artifact_lifecycle, "mark_publish_scheduled", publish_marker
    )

    artifact_lifecycle.schedule_artifact_publish(
        _publish_context(),
        lambda: [(transfer, identity)],
        engine_label="vLLM",
        ready_fn_factory=MagicMock(),
        artifact_publish_fn=MagicMock(),
        scheduled_publishers={},
    )

    publish_marker.assert_not_called()


def _publish_context():
    return SimpleNamespace(
        global_rank=0,
        device_id=0,
        worker_rank=0,
        worker_id="worker",
        node_rank=0,
        mx_client=object(),
        nixl_manager=object(),
        accelerator_backend=SimpleNamespace(name="cuda"),
    )


def test_publish_artifact_uses_selected_transport(monkeypatch, tmp_path):
    bundle = SimpleNamespace(
        artifact_id="artifact",
        manifest=SimpleNamespace(files=[]),
    )
    transfer = SimpleNamespace(
        name="cache",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
        roots=(SimpleNamespace(source_root=tmp_path, optional=False),),
        prepare_source=MagicMock(return_value=bundle),
    )
    handle = SimpleNamespace(identifier="published", transport="test", stop=MagicMock())
    transport = SimpleNamespace(publish=MagicMock(return_value=handle))
    monkeypatch.setattr(artifact_lifecycle, "has_files", lambda _path: True)
    monkeypatch.setattr(
        artifact_lifecycle,
        "_create_artifact_transport",
        lambda *args, **kwargs: transport,
    )

    result = artifact_lifecycle.publish_artifact(
        _publish_context(),
        transfer,
        _identity(),
        engine_label="vLLM",
        accelerator="cuda",
        published_sources={},
    )

    assert result is handle
    transport.publish.assert_called_once()
    assert transfer.prepare_source.call_count == 1
