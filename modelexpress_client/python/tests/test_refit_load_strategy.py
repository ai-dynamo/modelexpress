# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dedicated RL cold-start loading policy."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from modelexpress.adapter import EngineAdapter, StrategyFailed, StrategyRecoveryError
from modelexpress.load_strategy import LoadResult
from modelexpress_rl import (
    ObjectStorageSource,
    ObjectStorageType,
    WeightPayloadFormat,
    WeightVersion,
    WeightVersionState,
)
from modelexpress_rl import envs as rl_envs
from modelexpress_rl.inference.load_strategy import (
    DesiredVersionP2PStrategy,
    DesiredVersionS3Strategy,
    RLLoadStrategyChain,
    _resolve_s3_replay_chain,
)


def _context():
    """Build a default load context for cold-start strategy tests."""
    ctx = MagicMock()
    ctx.global_rank = 0
    ctx.identity.model_name = "test-model"
    ctx.identity.revision = "base"
    ctx.mx_server_url = "http://mx:8001"
    ctx.local_rank = 0
    ctx.desired_version_uid = rl_envs.MX_REFIT_DESIRED_VERSION_UID
    ctx.adapter = EngineAdapter()
    return ctx


class _DistributedAdapter(EngineAdapter):
    def __init__(self, gather, *, broadcast=None):
        """Configure deterministic collective results for a test."""
        self._gather = gather
        self._broadcast = broadcast

    def all_gather_state(self, state):
        """Return the states supplied by the configured gather callback."""
        return tuple(self._gather(state))

    def broadcast_state(self, state):
        """Return the configured rank-zero value or the local state."""
        return self._broadcast if self._broadcast is not None else state


def _version(
    uid,
    payload_format,
    *,
    base=None,
    model_name="test-model",
    state=WeightVersionState.READY,
):
    return WeightVersion(
        version_id=uid,
        model_name=model_name,
        payload_format=payload_format,
        base_version_id=base,
        object_storage=ObjectStorageSource(
            storage_type=ObjectStorageType.S3,
            uri=f"s3://weights/{uid}/model.safetensors.index.json",
        ),
        expected_source_slots=(),
        layout_signature="layout",
        state=state,
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


def test_distributed_cold_start_rejects_desired_version_disagreement(monkeypatch):
    """Reject desired-version disagreement reported by another rank."""
    monkeypatch.setenv("MX_REFIT_DESIRED_VERSION_UID", "version-7")
    model = nn.Linear(1, 1)
    ctx = _context()
    ctx.adapter = _DistributedAdapter(
        lambda state: (
            state,
            ("desired_version", "version-8", "version-8", True),
        )
    )

    with pytest.raises(
        StrategyRecoveryError,
        match="desired-version disagreement",
    ):
        RLLoadStrategyChain.run(model, ctx)


def test_distributed_cold_start_rejects_local_config_different_from_rank_zero(
    monkeypatch,
):
    """Reject a local desired version that differs from rank zero."""
    monkeypatch.setenv("MX_REFIT_DESIRED_VERSION_UID", "version-8")
    model = nn.Linear(1, 1)
    ctx = _context()
    ctx.adapter = _DistributedAdapter(lambda state: (state,), broadcast="version-7")

    with pytest.raises(
        StrategyRecoveryError,
        match="desired-version disagreement",
    ):
        RLLoadStrategyChain.run(model, ctx)


def test_p2p_rejects_desired_version_change_during_load(monkeypatch):
    """Reject configuration drift while a desired-version P2P load runs."""
    monkeypatch.setenv("MX_REFIT_DESIRED_VERSION_UID", "version-7")
    ctx = _context()
    result = LoadResult(value=nn.Linear(1, 1))
    client = MagicMock()
    client.__enter__.return_value = client
    client.get_weight_version.return_value = _version(
        "version-7", WeightPayloadFormat.FULL_HF_CHECKPOINT
    )

    def change_desired(*_args):
        """Change the configured version while returning a successful load."""
        monkeypatch.setenv("MX_REFIT_DESIRED_VERSION_UID", "version-8")
        return result

    with patch(
        "modelexpress_rl.inference.load_strategy.ModelExpressControlClient.connect",
        return_value=client,
    ), patch(
        "modelexpress.load_strategy.rdma_strategy.RdmaStrategy.load",
        side_effect=change_desired,
    ), pytest.raises(
        StrategyRecoveryError,
        match="desired-version disagreement",
    ):
        DesiredVersionP2PStrategy().load(result, ctx)


def test_p2p_rejects_rank_load_outcome_disagreement(monkeypatch):
    """Reject disagreement between rank-local P2P load outcomes."""
    monkeypatch.setenv("MX_REFIT_DESIRED_VERSION_UID", "version-7")
    ctx = _context()

    def gather(state):
        """Inject a peer P2P miss into the gathered outcomes."""
        phase, desired, configured, result = state
        peer_result = "miss" if phase == "p2p_result" else result
        return state, (phase, desired, configured, peer_result)

    ctx.adapter = _DistributedAdapter(gather)
    result = LoadResult(value=nn.Linear(1, 1))
    client = MagicMock()
    client.__enter__.return_value = client
    client.get_weight_version.return_value = _version(
        "version-7", WeightPayloadFormat.FULL_HF_CHECKPOINT
    )

    with patch(
        "modelexpress_rl.inference.load_strategy.ModelExpressControlClient.connect",
        return_value=client,
    ), patch(
        "modelexpress.load_strategy.rdma_strategy.RdmaStrategy.load",
        return_value=result,
    ), pytest.raises(
        StrategyRecoveryError,
        match="result disagreement during P2P load",
    ):
        DesiredVersionP2PStrategy().load(result, ctx)


def test_desired_p2p_is_skipped_without_desired_version(monkeypatch):
    monkeypatch.delenv("MX_REFIT_DESIRED_VERSION_UID", raising=False)
    assert DesiredVersionP2PStrategy().is_available(_context()) is False


def test_desired_p2p_uses_exact_revision(monkeypatch):
    monkeypatch.setenv("MX_REFIT_DESIRED_VERSION_UID", "version-7")
    ctx = _context()
    result = LoadResult(value=nn.Linear(1, 1))
    client = MagicMock()
    client.__enter__.return_value = client
    client.get_weight_version.return_value = _version(
        "version-7", WeightPayloadFormat.FULL_HF_CHECKPOINT
    )

    with patch(
        "modelexpress_rl.inference.load_strategy.ModelExpressControlClient.connect",
        return_value=client,
    ), patch(
        "modelexpress.load_strategy.rdma_strategy.RdmaStrategy.load",
        return_value=result,
    ) as load:
        assert DesiredVersionP2PStrategy().load(result, ctx) is result

    load.assert_called_once_with(result, ctx)
    client.get_weight_version.assert_called_once_with("version-7")
    assert ctx.identity.revision == "version-7"


@pytest.mark.parametrize(
    ("version", "message"),
    [
        (
            _version(
                "version-7",
                WeightPayloadFormat.FULL_HF_CHECKPOINT,
                state=WeightVersionState.STAGING,
            ),
            "not READY",
        ),
        (
            _version(
                "version-7",
                WeightPayloadFormat.FULL_HF_CHECKPOINT,
                model_name="other-model",
            ),
            "model_name does not match",
        ),
        (
            _version("other-version", WeightPayloadFormat.FULL_HF_CHECKPOINT),
            "requested revision 'version-7' but MX returned 'other-version'",
        ),
    ],
)
def test_desired_p2p_requires_matching_ready_control_record(
    monkeypatch, version, message
):
    monkeypatch.setenv("MX_REFIT_DESIRED_VERSION_UID", "version-7")
    ctx = _context()
    result = LoadResult(value=nn.Linear(1, 1))
    client = MagicMock()
    client.__enter__.return_value = client
    client.get_weight_version.return_value = version

    with patch(
        "modelexpress_rl.inference.load_strategy.ModelExpressControlClient.connect",
        return_value=client,
    ), patch(
        "modelexpress.load_strategy.rdma_strategy.RdmaStrategy.load"
    ) as load, pytest.raises(StrategyFailed, match=message):
        DesiredVersionP2PStrategy().load(result, ctx)

    load.assert_not_called()
    assert ctx.identity.revision == "base"


def test_desired_p2p_restores_revision_on_miss(monkeypatch):
    monkeypatch.setenv("MX_REFIT_DESIRED_VERSION_UID", "version-7")
    ctx = _context()
    result = LoadResult(value=nn.Linear(1, 1))
    client = MagicMock()
    client.__enter__.return_value = client
    client.get_weight_version.return_value = _version(
        "version-7", WeightPayloadFormat.FULL_HF_CHECKPOINT
    )

    with patch(
        "modelexpress_rl.inference.load_strategy.ModelExpressControlClient.connect",
        return_value=client,
    ), patch(
        "modelexpress.load_strategy.rdma_strategy.RdmaStrategy.load",
        side_effect=StrategyFailed("miss", mutated=False),
    ), pytest.raises(StrategyFailed):
        DesiredVersionP2PStrategy().load(result, ctx)

    assert ctx.identity.revision == "base"


def test_missing_optional_sources_reaches_engine_default(monkeypatch, caplog):
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

    with caplog.at_level("WARNING"), patch(
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
    assert "RL initial load has no desired version" in caplog.text


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


def test_resolve_s3_chain_rejects_mismatched_returned_uid():
    client = MagicMock()
    client.__enter__.return_value = client
    client.get_weight_version.return_value = _version(
        "other", WeightPayloadFormat.FULL_HF_CHECKPOINT
    )

    with patch(
        "modelexpress_rl.inference.load_strategy.ModelExpressControlClient.connect",
        return_value=client,
    ), pytest.raises(
        RuntimeError,
        match="requested revision 'desired' but MX returned 'other'",
    ):
        _resolve_s3_replay_chain(_context(), "desired")


def test_desired_s3_loads_materialized_checkpoint(monkeypatch, tmp_path):
    """Load and activate a checkpoint materialized from the desired S3 chain."""
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
    method.installation_context.assert_called_once_with(
        method.prepare_chain.return_value,
        activate=False,
    )
    method.activate.assert_called_once_with(method.prepare_chain.return_value)
    method.release.assert_called_once_with(method.prepare_chain.return_value)
    method.close.assert_called_once()
    s3.close.assert_called_once()


def test_desired_s3_does_not_activate_until_every_rank_loads(monkeypatch, tmp_path):
    """Do not activate a checkpoint when another rank fails engine loading."""
    monkeypatch.setenv("MX_REFIT_DESIRED_VERSION_UID", "delta")
    monkeypatch.setenv("MX_REFIT_CHECKPOINT_DIR", str(tmp_path))
    ctx = _context()

    def gather(state):
        """Inject a peer engine-load failure into gathered phase state."""
        phase, desired, configured, result = state
        peer_result = False if phase == "s3_engine_loaded" else result
        return state, (phase, desired, configured, peer_result)

    ctx.adapter = _DistributedAdapter(gather)
    root = _version("root", WeightPayloadFormat.FULL_HF_CHECKPOINT)
    delta = _version("delta", WeightPayloadFormat.XOR_DELTA, base="root")
    result = LoadResult(value=nn.Linear(1, 1))
    method = MagicMock()
    method.installation_context.return_value = nullcontext()
    method.prepare_chain.return_value = SimpleNamespace(
        checkpoint=SimpleNamespace(path=Path("/cache/delta"))
    )

    with patch(
        "modelexpress_rl.inference.load_strategy._resolve_s3_replay_chain",
        return_value=(root, delta),
    ), patch.object(
        DesiredVersionS3Strategy,
        "_prepare",
        return_value=(method, method.prepare_chain.return_value),
    ), patch.object(
        DesiredVersionS3Strategy,
        "load_uri",
        return_value=result,
    ), pytest.raises(
        StrategyRecoveryError,
        match="at least one rank failed to load",
    ):
        DesiredVersionS3Strategy().load(result, ctx)

    method.activate.assert_not_called()


def test_desired_s3_only_local_leader_activates(monkeypatch, tmp_path):
    """Allow only the node-local leader to prepare and activate checkpoints."""
    monkeypatch.setenv("MX_REFIT_DESIRED_VERSION_UID", "delta")
    monkeypatch.setenv("MX_REFIT_CHECKPOINT_DIR", str(tmp_path))
    ctx = _context()
    events = []

    def gather(state):
        """Record phases and report successful leader-only operations."""
        phase, desired, configured, result = state
        events.append(phase)
        peer_result = (
            (True, True)
            if phase in {
                "s3_local_leader_prepared",
                "s3_checkpoint_activated",
            }
            else result
        )
        return state, (phase, desired, configured, peer_result)

    ctx.adapter = _DistributedAdapter(gather)
    ctx.local_rank = 1
    root = _version("root", WeightPayloadFormat.FULL_HF_CHECKPOINT)
    delta = _version("delta", WeightPayloadFormat.XOR_DELTA, base="root")
    result = LoadResult(value=nn.Linear(1, 1))
    method = MagicMock()
    method.installation_context.return_value = nullcontext()
    method.prepare_chain.return_value = SimpleNamespace(
        checkpoint=SimpleNamespace(path=Path("/cache/delta"))
    )

    def prepare(*_args):
        """Record an unexpected follower preparation attempt."""
        events.append("prepare")
        return method, method.prepare_chain.return_value

    with patch(
        "modelexpress_rl.inference.load_strategy._resolve_s3_replay_chain",
        return_value=(root, delta),
    ), patch.object(
        DesiredVersionS3Strategy,
        "_prepare",
        side_effect=prepare,
    ), patch.object(
        DesiredVersionS3Strategy,
        "load_uri",
        return_value=result,
    ):
        assert DesiredVersionS3Strategy().load(result, ctx) is result

    method.activate.assert_not_called()
    assert events.index("s3_local_leader_prepared") < events.index("prepare")
