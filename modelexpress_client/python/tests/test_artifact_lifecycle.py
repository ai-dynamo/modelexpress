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
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from modelexpress import p2p_pb2
from modelexpress.metadata import artifact_lifecycle
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
