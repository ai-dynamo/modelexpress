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
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from modelexpress import p2p_pb2
from modelexpress.metadata import artifact_lifecycle
from modelexpress.metadata.artifact_transfer import ArtifactCacheRoot
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


def _run_install(identity, *, install_result):
    """Drive ``install_artifacts`` past its guards with a stubbed install."""
    ctx = SimpleNamespace(
        global_rank=0,
        device_id=0,
        nixl_manager=object(),
        mx_client=object(),
    )
    transfer = SimpleNamespace(name="torch_compile_cache")

    with patch.object(
        artifact_lifecycle, "_metadata_publication_configured", return_value=True
    ), patch.object(
        artifact_lifecycle, "is_nixl_available", return_value=True
    ), patch.object(
        artifact_lifecycle, "install_artifact_once", **install_result
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
        _run_install(identity, install_result={"side_effect": LookupError("no source")})

    records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "No ready vLLM artifact source" in message
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
        _run_install(_identity(), install_result={"side_effect": LookupError()})

    assert "compile_config_digest=''" in caplog.text


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


@pytest.fixture
def mooncake_publish_state():
    artifact_lifecycle._prepared_artifact_bundles.clear()
    artifact_lifecycle._prepared_artifact_locks.clear()
    artifact_lifecycle._mooncake_publish_needed.clear()
    yield
    artifact_lifecycle._prepared_artifact_bundles.clear()
    artifact_lifecycle._prepared_artifact_locks.clear()
    artifact_lifecycle._mooncake_publish_needed.clear()


def _mooncake_publish_context():
    return SimpleNamespace(
        global_rank=0,
        node_rank=0,
        accelerator_backend=SimpleNamespace(name="cuda"),
    )


def _mooncake_publish_transfer(tmp_path, bundle):
    return SimpleNamespace(
        name="torch_compile_cache",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
        roots=(SimpleNamespace(source_root=tmp_path, optional=False),),
        prepare_source=MagicMock(return_value=bundle),
    )


def _publish_after_mooncake(
    ctx,
    transfer,
    identity,
    p2p_publish_fn,
):
    artifact_lifecycle._mark_mooncake_publish_needed(ctx, transfer, identity)
    return artifact_lifecycle._publish_mooncake_then_p2p_artifact(
        ctx,
        transfer,
        identity,
        engine_label="vLLM",
        p2p_publish_fn=p2p_publish_fn,
        p2p_publish_available=True,
        log=MagicMock(),
    )


def test_mooncake_prepared_bundle_is_released_when_p2p_publish_fails(
    monkeypatch,
    tmp_path,
    mooncake_publish_state,
):
    monkeypatch.setenv("MX_ARTIFACT_BACKEND", "mooncake")
    monkeypatch.setattr(artifact_lifecycle, "has_files", lambda _path: True)
    monkeypatch.setattr(artifact_lifecycle, "publish_to_mooncake", MagicMock())
    bundle = SimpleNamespace()
    transfer = _mooncake_publish_transfer(tmp_path, bundle)
    identity = _identity()

    with pytest.raises(RuntimeError, match="P2P publish failed"):
        _publish_after_mooncake(
            _mooncake_publish_context(),
            transfer,
            identity,
            MagicMock(side_effect=RuntimeError("P2P publish failed")),
        )

    assert artifact_lifecycle._prepared_artifact_bundles == {}


def test_mooncake_prepared_bundle_is_released_when_p2p_callback_bypasses_publish(
    monkeypatch,
    tmp_path,
    mooncake_publish_state,
):
    monkeypatch.setenv("MX_ARTIFACT_BACKEND", "mooncake")
    monkeypatch.setattr(artifact_lifecycle, "has_files", lambda _path: True)
    monkeypatch.setattr(artifact_lifecycle, "publish_to_mooncake", MagicMock())
    bundle = SimpleNamespace()
    transfer = _mooncake_publish_transfer(tmp_path, bundle)
    identity = _identity()
    p2p_publish_fn = MagicMock(
        return_value=SimpleNamespace(endpoint=SimpleNamespace(mx_source_id="source"))
    )

    assert (
        _publish_after_mooncake(
            _mooncake_publish_context(), transfer, identity, p2p_publish_fn
        )
        == "source"
    )
    assert artifact_lifecycle._prepared_artifact_bundles == {}


def test_p2p_publish_consumes_mooncake_prepared_bundle(
    monkeypatch,
    tmp_path,
    mooncake_publish_state,
):
    bundle = SimpleNamespace(manifest=SimpleNamespace(files=[]), artifact_id="artifact")
    transfer = _mooncake_publish_transfer(tmp_path, bundle)
    identity = _identity()
    prepared_key = artifact_lifecycle._prepared_artifact_key(transfer, identity)
    artifact_lifecycle._prepared_artifact_bundles[prepared_key] = bundle
    published = SimpleNamespace(endpoint=SimpleNamespace(mx_source_id="source"))
    ctx = SimpleNamespace(
        global_rank=0,
        device_id=0,
        mx_client=object(),
        nixl_manager=object(),
        worker_id="worker",
        worker_rank=0,
        node_rank=0,
    )

    monkeypatch.setattr(artifact_lifecycle, "has_files", lambda _path: True)
    monkeypatch.setattr(
        artifact_lifecycle, "_get_worker_server", lambda _device: object()
    )
    publish_source = MagicMock(return_value=published)
    monkeypatch.setattr(artifact_lifecycle, "publish_artifact_source", publish_source)

    result = artifact_lifecycle.publish_artifact(
        ctx,
        transfer,
        identity,
        engine_label="vLLM",
        accelerator="cuda",
        published_sources={},
    )

    assert result is published
    transfer.prepare_source.assert_not_called()
    assert artifact_lifecycle._prepared_artifact_bundles == {}
    assert publish_source.call_args.args[2] is bundle


def test_mooncake_prepared_bundles_are_serialized_per_transfer(
    monkeypatch,
    tmp_path,
    mooncake_publish_state,
):
    monkeypatch.setenv("MX_ARTIFACT_BACKEND", "mooncake")
    monkeypatch.setattr(artifact_lifecycle, "has_files", lambda _path: True)
    monkeypatch.setattr(artifact_lifecycle, "publish_to_mooncake", MagicMock())
    bundles = [SimpleNamespace(name="first"), SimpleNamespace(name="second")]
    transfer = _mooncake_publish_transfer(tmp_path, bundles)
    transfer.prepare_source.side_effect = bundles
    identity = _identity()
    first_callback_started = threading.Event()
    release_first_callback = threading.Event()
    second_callback_started = threading.Event()
    results = []

    def p2p_publish_fn(_transfer, _identity):
        if not first_callback_started.is_set():
            first_callback_started.set()
            assert release_first_callback.wait(timeout=1)
        else:
            second_callback_started.set()
        return SimpleNamespace(endpoint=SimpleNamespace(mx_source_id="source"))

    ctx = _mooncake_publish_context()
    artifact_lifecycle._mark_mooncake_publish_needed(ctx, transfer, identity)
    first = threading.Thread(
        target=lambda: results.append(
            _publish_after_mooncake(ctx, transfer, identity, p2p_publish_fn)
        )
    )
    first.start()
    assert first_callback_started.wait(timeout=1)

    # Simulate a rescheduled publisher for the same artifact while the old
    # publisher is still inside its synchronous P2P callback.
    artifact_lifecycle._mark_mooncake_publish_needed(ctx, transfer, identity)
    second = threading.Thread(
        target=lambda: results.append(
            _publish_after_mooncake(ctx, transfer, identity, p2p_publish_fn)
        )
    )
    second.start()
    assert not second_callback_started.wait(timeout=0.1)

    release_first_callback.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert sorted(results) == ["source", "source"]
    assert transfer.prepare_source.call_count == 2
    assert artifact_lifecycle._prepared_artifact_bundles == {}
