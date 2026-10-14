# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from modelexpress_rl import (
    ModelExpressGeneratorClient,
    TrainerMeshRef,
    WeightPayloadFormat,
    WeightVersion,
    WeightVersionRef,
    WeightVersionState,
)
from modelexpress.adapter import StrategyRecoveryError
from modelexpress_rl.control import TrainerMesh
from modelexpress_rl.inference.session import WeightUpdateSession
from modelexpress_rl.inference.plan import WeightSource


def test_prepare_none_reuses_runtime_without_creating_update():
    client = ModelExpressGeneratorClient()
    client._runtime = Mock()
    client._runtime.session = Mock(prepare=Mock(return_value={}))
    client.prepare()
    client._runtime.session.prepare.assert_called_once_with(None)
    assert client._active_handle is None


def test_prepare_rejects_active_handle_even_for_same_mesh():
    client = ModelExpressGeneratorClient()
    client._runtime = Mock()
    client._active_handle = object()
    with pytest.raises(RuntimeError, match="active"):
        client.prepare(TrainerMeshRef("mesh", 1))
    client._runtime.session.prepare.assert_not_called()


def test_prepare_requires_open_initialized_client():
    client = ModelExpressGeneratorClient()
    with pytest.raises(RuntimeError, match="initialized"):
        client.prepare()
    client._runtime = Mock()
    client._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        client.prepare()


def test_preparation_failure_skips_only_trainer_for_one_stage():
    session = WeightUpdateSession(
        planner=SimpleNamespace(source_order=(WeightSource.TRAINER, WeightSource.OBJECT_STORAGE)),
        start_lease=Mock(),
    )
    session._stage_source = Mock(return_value="staged")
    version = SimpleNamespace(version_id="v")
    assert session.stage(version, preparation_errors={WeightSource.TRAINER: RuntimeError("mesh unavailable")}) == "staged"
    assert session._stage_source.call_args.kwargs["source_kind"] is WeightSource.OBJECT_STORAGE
    session._stage_source.reset_mock()
    session.stage(version)
    assert session._stage_source.call_args.kwargs["source_kind"] is WeightSource.TRAINER


def test_preparation_failure_without_fallback_is_reported():
    session = WeightUpdateSession(
        planner=SimpleNamespace(source_order=(WeightSource.TRAINER,)), start_lease=Mock(),
    )
    with pytest.raises(RuntimeError, match="mesh unavailable"):
        session.stage(SimpleNamespace(version_id="v"), preparation_errors={WeightSource.TRAINER: RuntimeError("mesh unavailable")})
    session._start_lease.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [None, RuntimeError("mesh unavailable"), StrategyRecoveryError("unsafe capture")],
)
def test_stage_calls_prepare_and_only_allows_recoverable_fallback(failure):
    client = ModelExpressGeneratorClient()
    runtime = Mock(initial_version_id=None)
    runtime.session.prepare.return_value = (
        {WeightSource.TRAINER: failure}
        if isinstance(failure, RuntimeError) and not isinstance(failure, StrategyRecoveryError)
        else {}
    )
    if isinstance(failure, StrategyRecoveryError):
        runtime.session.prepare.side_effect = failure
    client._runtime = runtime
    ready = WeightVersion(
        version_id="v",
        model_name="model",
        payload_format=WeightPayloadFormat.FULL_TENSOR,
        layout_signature="",
        state=WeightVersionState.READY,
        created_at_unix_ms=0,
        trainer_mesh_id="mesh",
        trainer_mesh_generation=1,
    )
    client._fetch_ready_version = Mock(return_value=ready)
    mesh = TrainerMesh("mesh", "model", 1, {})
    client._fetch_mesh = Mock(return_value=mesh)
    if isinstance(failure, StrategyRecoveryError):
        with pytest.raises(StrategyRecoveryError, match="unsafe capture"):
            client.stage_weight(version=WeightVersionRef("v"))
        runtime.session.stage.assert_not_called()
        assert client._active_handle is None
    else:
        staged = client.stage_weight(version=WeightVersionRef("v"))
        assert staged is client._active_handle
        errors = {WeightSource.TRAINER: failure} if failure else {}
        runtime.session.stage.assert_called_once_with(ready, preparation_errors=errors)
    assert runtime.session.prepare.call_args.args[0] is mesh
    assert callable(runtime.session.prepare.call_args.kwargs["validate_mesh"])


def test_explicit_prepare_reports_failure():
    client = ModelExpressGeneratorClient()
    client._runtime = Mock()
    client._fetch_mesh = Mock(return_value=TrainerMesh("mesh", "model", 1, {}))
    client._runtime.session.prepare.return_value = {
        WeightSource.TRAINER: RuntimeError("mesh unavailable")
    }
    with pytest.raises(RuntimeError, match="mesh unavailable"):
        client.prepare(TrainerMeshRef("mesh", 1))
    assert client._active_handle is None


def test_unsafe_preparation_fences_later_attempts():
    client = ModelExpressGeneratorClient()
    client._runtime = Mock()
    client._fetch_mesh = Mock(return_value=TrainerMesh("mesh", "model", 1, {}))
    client._runtime.session.prepare.side_effect = StrategyRecoveryError("unsafe capture")
    with pytest.raises(StrategyRecoveryError, match="unsafe capture"):
        client.prepare(TrainerMeshRef("mesh", 1))
    client._runtime.session.prepare.side_effect = None
    with pytest.raises(StrategyRecoveryError, match="restart"):
        client.prepare(None)
    assert client._runtime.session.prepare.call_count == 1


def test_closed_client_cannot_fallback_during_staging():
    client = ModelExpressGeneratorClient()
    client._closed = True
    client._runtime = Mock()
    with pytest.raises(RuntimeError, match="closed"):
        client.stage_weight(version=WeightVersionRef("v"))
    client._runtime.session.stage.assert_not_called()
