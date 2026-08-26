# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from modelexpress_rl.inference.engines.vllm.weight_transfer_engine import (
    ModelExpressWeightTransferEngine,
)


def _engine(monkeypatch, client=None, *, initialize=True):
    client = client or MagicMock()
    monkeypatch.setattr(
        "modelexpress_rl.inference.engines.vllm.weight_transfer_engine."
        "ModelExpressGeneratorClient.initialize",
        lambda config: client,
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        model_config=SimpleNamespace(model="test/model"),
    )
    engine = ModelExpressWeightTransferEngine(
        SimpleNamespace(),
        vllm_config,
        torch.device("cpu"),
        torch.nn.Linear(2, 2),
    )
    if initialize:
        engine.init_transfer_engine(engine.init_info_cls())
    return engine, client


def test_weight_transfer_engine_initializes_client_in_init_hook(monkeypatch):
    client = MagicMock()
    initialize = MagicMock(return_value=client)
    monkeypatch.setattr(
        "modelexpress_rl.inference.engines.vllm.weight_transfer_engine."
        "ModelExpressGeneratorClient.initialize",
        initialize,
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        model_config=SimpleNamespace(model="test/model"),
    )
    model = torch.nn.Linear(2, 2)
    engine = ModelExpressWeightTransferEngine(
        SimpleNamespace(), vllm_config, torch.device("cpu"), model
    )

    initialize.assert_not_called()
    engine.init_transfer_engine(engine.init_info_cls())

    config = initialize.call_args.args[0]
    assert config.model_name == "test/model"
    assert config.engine_context.model is model


def test_weight_transfer_engine_applies_one_exact_version(monkeypatch):
    engine, client = _engine(monkeypatch)
    staged = client.stage_weight.return_value

    engine.start_weight_update()
    engine.update_weights({"version_id": "version-a"})
    engine.finish_weight_update()

    version = client.stage_weight.call_args.kwargs["version"]
    assert version.version_id == "version-a"
    client.apply_weight.assert_called_once_with(staged)
    staged.release.assert_called_once_with()


def test_weight_transfer_engine_warns_for_out_of_order_updates(monkeypatch, caplog):
    engine, _client = _engine(monkeypatch)

    engine.update_weights({"version_id": "version-a"})

    engine.start_weight_update()
    engine.update_weights({"version_id": "version-a"})
    engine.update_weights({"version_id": "version-b"})

    assert "weight update has not been started" in caplog.text
    assert "weight update already received a version" in caplog.text


def test_weight_transfer_engine_warns_for_duplicate_lifecycle_calls(
    monkeypatch, caplog
):
    engine, client = _engine(monkeypatch)

    engine.init_transfer_engine(engine.init_info_cls())
    engine.start_weight_update()
    engine.start_weight_update()
    engine.finish_weight_update()
    engine.finish_weight_update()

    assert client.stage_weight.call_count == 0
    assert "weight transfer engine is already initialized" in caplog.text
    assert "weight update is already active" in caplog.text
    assert "weight update has not received a version" in caplog.text
    assert "weight update has not been started" in caplog.text


def test_weight_transfer_engine_releases_failed_update(monkeypatch):
    client = MagicMock()
    client.apply_weight.side_effect = RuntimeError("apply failed")
    engine, client = _engine(monkeypatch, client)
    staged = client.stage_weight.return_value

    engine.start_weight_update()
    with pytest.raises(RuntimeError, match="apply failed"):
        engine.update_weights({"version_id": "version-a"})

    staged.release.assert_called_once_with()
    engine.start_weight_update()


def test_weight_transfer_engine_preserves_apply_error_when_release_fails(monkeypatch):
    client = MagicMock()
    client.apply_weight.side_effect = RuntimeError("apply failed")
    client.stage_weight.return_value.release.side_effect = RuntimeError(
        "release failed"
    )
    engine, _client = _engine(monkeypatch, client)

    engine.start_weight_update()
    with pytest.raises(RuntimeError, match="apply failed"):
        engine.update_weights({"version_id": "version-a"})

    engine.start_weight_update()


def test_weight_transfer_engine_shutdown_is_idempotent(monkeypatch):
    engine, client = _engine(monkeypatch)
    staged = client.stage_weight.return_value
    engine.start_weight_update()
    engine.update_weights({"version_id": "version-a"})

    engine.shutdown()
    engine.shutdown()

    staged.release.assert_called_once_with()
    client.close.assert_called_once_with()


def test_weight_transfer_engine_shutdown_before_initialization(monkeypatch):
    engine, client = _engine(monkeypatch, initialize=False)

    engine.shutdown()
    engine.start_weight_update()

    client.close.assert_not_called()


def test_weight_transfer_engine_ignores_vllm_trainer_transport(caplog):
    ModelExpressWeightTransferEngine.trainer_send_weights(iter(()), {})

    assert "ModelExpressTrainerClient" in caplog.text


@pytest.mark.parametrize("version_id", ["", "   ", None, 1])
def test_weight_transfer_engine_rejects_invalid_version_id(version_id):
    with pytest.raises(ValueError, match="version_id is required"):
        ModelExpressWeightTransferEngine.update_info_cls(version_id=version_id)


def test_vllm_plugin_registers_weight_transfer_engine(monkeypatch):
    from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

    from modelexpress.engines.vllm import registration

    calls = []
    monkeypatch.setattr(WeightTransferEngineFactory, "_registry", {})
    monkeypatch.setattr(
        WeightTransferEngineFactory,
        "register_engine",
        lambda name, module_path, class_name: calls.append(
            (name, module_path, class_name)
        ),
    )

    registration.register_plugin_weight_transfer_engine()

    assert calls == [
        (
            "modelexpress",
            "modelexpress_rl.inference.engines.vllm.weight_transfer_engine",
            "ModelExpressWeightTransferEngine",
        )
    ]
