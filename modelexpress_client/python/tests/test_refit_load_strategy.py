# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dedicated RL cold-start loading policy."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from modelexpress.adapter import StrategyFailed
from modelexpress.load_strategy import LoadResult
from modelexpress_rl import (
    ObjectStorageSource,
    ObjectStorageType,
    WeightPayloadFormat,
    WeightVersion,
    WeightVersionState,
)
from modelexpress_rl.inference.load_strategy import (
    DesiredVersionP2PStrategy,
    DesiredVersionS3Strategy,
    RLLoadStrategyChain,
    _resolve_s3_replay_chain,
)


def _context():
    ctx = MagicMock()
    ctx.global_rank = 0
    ctx.identity.model_name = "test-model"
    ctx.identity.revision = "base"
    ctx.mx_server_url = "http://mx:8001"
    return ctx


def _version(uid, payload_format, *, base=None):
    return WeightVersion(
        version_id=uid,
        model_name="test-model",
        payload_format=payload_format,
        base_version_id=base,
        object_storage=ObjectStorageSource(
            storage_type=ObjectStorageType.S3,
            uri=f"s3://weights/{uid}/model.safetensors.index.json",
        ),
        expected_source_slots=(),
        layout_signature="layout",
        state=WeightVersionState.READY,
        created_at_unix_ms=1,
    )


def test_desired_version_does_not_use_version_agnostic_fallbacks(monkeypatch):
    monkeypatch.setenv("MX_REFIT_DESIRED_VERSION_UID", "version-7")
    model = nn.Linear(1, 1)
    ctx = _context()
    unavailable_p2p = MagicMock(name="desired_version_p2p")
    unavailable_p2p.is_available.return_value = False
    unavailable_s3 = MagicMock(name="desired_version_s3")
    unavailable_s3.is_available.return_value = False
    fallback = MagicMock(name="default")
    fallback.is_available.return_value = True
    fallback.load.return_value = LoadResult(value=model, model=model)

    with patch(
        "modelexpress_rl.inference.load_strategy.DesiredVersionP2PStrategy",
        return_value=unavailable_p2p,
    ), patch(
        "modelexpress_rl.inference.load_strategy.DesiredVersionS3Strategy",
        return_value=unavailable_s3,
    ), patch(
        "modelexpress_rl.inference.load_strategy.ModelStreamerStrategy",
        return_value=fallback,
    ), patch(
        "modelexpress_rl.inference.load_strategy.DefaultStrategy",
        return_value=fallback,
    ), pytest.raises(RuntimeError, match="No loading strategy succeeded"):
        RLLoadStrategyChain.run(model, ctx)

    fallback.load.assert_not_called()


def test_desired_p2p_is_skipped_without_desired_version(monkeypatch):
    monkeypatch.delenv("MX_REFIT_DESIRED_VERSION_UID", raising=False)
    assert DesiredVersionP2PStrategy().is_available(_context()) is False


def test_desired_p2p_uses_exact_revision(monkeypatch):
    monkeypatch.setenv("MX_REFIT_DESIRED_VERSION_UID", "version-7")
    ctx = _context()
    result = LoadResult(value=nn.Linear(1, 1))

    with patch(
        "modelexpress.load_strategy.rdma_strategy.RdmaStrategy.load",
        return_value=result,
    ) as load:
        assert DesiredVersionP2PStrategy().load(result, ctx) is result

    load.assert_called_once_with(result, ctx)
    assert ctx.identity.revision == "version-7"


def test_desired_p2p_restores_revision_on_miss(monkeypatch):
    monkeypatch.setenv("MX_REFIT_DESIRED_VERSION_UID", "version-7")
    ctx = _context()
    result = LoadResult(value=nn.Linear(1, 1))

    with patch(
        "modelexpress.load_strategy.rdma_strategy.RdmaStrategy.load",
        side_effect=StrategyFailed("miss", mutated=False),
    ), pytest.raises(StrategyFailed):
        DesiredVersionP2PStrategy().load(result, ctx)

    assert ctx.identity.revision == "base"


def test_missing_optional_sources_reaches_engine_default(monkeypatch):
    monkeypatch.delenv("MX_REFIT_DESIRED_VERSION_UID", raising=False)
    monkeypatch.delenv("MX_REFIT_CHECKPOINT_DIR", raising=False)
    monkeypatch.delenv("MX_MODEL_URI", raising=False)
    model = nn.Linear(1, 1)
    ctx = _context()
    unavailable = MagicMock()
    unavailable.is_available.return_value = False
    fallback = MagicMock()
    fallback.name = "default"
    fallback.is_available.return_value = True
    fallback.load.return_value = LoadResult(value=model, model=model)

    with patch(
        "modelexpress_rl.inference.load_strategy.DesiredVersionP2PStrategy",
        return_value=unavailable,
    ), patch(
        "modelexpress_rl.inference.load_strategy.DesiredVersionS3Strategy",
        return_value=unavailable,
    ), patch(
        "modelexpress_rl.inference.load_strategy.ModelStreamerStrategy",
        return_value=unavailable,
    ), patch(
        "modelexpress_rl.inference.load_strategy.DefaultStrategy",
        return_value=fallback,
    ):
        assert RLLoadStrategyChain.run(model, ctx) is model

    fallback.load.assert_called_once()


def test_desired_s3_is_skipped_without_cache_directory(monkeypatch):
    monkeypatch.setenv("MX_REFIT_DESIRED_VERSION_UID", "version-7")
    monkeypatch.delenv("MX_REFIT_CHECKPOINT_DIR", raising=False)
    assert DesiredVersionS3Strategy().is_available(_context()) is False


def test_resolve_s3_chain_returns_full_root_then_deltas():
    root = _version("root", WeightPayloadFormat.FULL_HF_CHECKPOINT)
    delta = _version("delta", WeightPayloadFormat.XOR_DELTA, base="root")
    client = MagicMock()
    client.__enter__.return_value = client
    client.get_weight_version.side_effect = [delta, root]

    with patch(
        "modelexpress_rl.inference.load_strategy.ModelExpressControlClient.connect",
        return_value=client,
    ):
        assert _resolve_s3_replay_chain(_context(), "delta") == (root, delta)


def test_desired_s3_loads_materialized_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("MX_REFIT_DESIRED_VERSION_UID", "delta")
    monkeypatch.setenv("MX_REFIT_CHECKPOINT_DIR", str(tmp_path))
    ctx = _context()
    root = _version("root", WeightPayloadFormat.FULL_HF_CHECKPOINT)
    delta = _version("delta", WeightPayloadFormat.XOR_DELTA, base="root")
    result = LoadResult(value=nn.Linear(1, 1))
    loaded = LoadResult(value=nn.Linear(1, 1))
    method = MagicMock()
    method.installation_context.return_value = nullcontext()
    method.prepare_chain.return_value = SimpleNamespace(
        checkpoint=SimpleNamespace(path=Path("/cache/delta"))
    )
    s3 = MagicMock()
    strategy = DesiredVersionS3Strategy()

    with patch(
        "modelexpress_rl.inference.load_strategy._resolve_s3_replay_chain",
        return_value=(root, delta),
    ), patch(
        "modelexpress_rl.inference.load_strategy.S3Client", return_value=s3
    ), patch(
        "modelexpress_rl.inference.load_strategy.bootstrap_s3_checkpoint",
        return_value=Path("/cache/root"),
    ), patch(
        "modelexpress_rl.inference.load_strategy.CanonicalDeltaUpdateMethod",
        return_value=method,
    ), patch.object(strategy, "load_uri", return_value=loaded) as load_uri:
        assert strategy.load(result, ctx) is loaded

    load_uri.assert_called_once_with(result, ctx, "/cache/delta")
    assert ctx.identity.revision == "delta"
    method.release.assert_called_once_with(method.prepare_chain.return_value)
    method.close.assert_called_once()
    s3.close.assert_called_once()
