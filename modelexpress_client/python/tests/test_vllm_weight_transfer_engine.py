# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from modelexpress_rl import ObjectStorageType
from modelexpress_rl.inference.engines.vllm import weight_transfer_engine
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
    assert config.object_storage is None


def test_weight_transfer_engine_parses_vime_object_storage_init_info(monkeypatch):
    client = MagicMock()
    initialize = MagicMock(return_value=client)
    monkeypatch.setattr(
        weight_transfer_engine.ModelExpressGeneratorClient,
        "initialize",
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

    init_info = {
        "model_name": "policy",
        "initial_base_version_id": "base-a",
        "seed_checkpoint_path": "/models/launch",
        "refit_checkpoint_dir": "/cache/modelexpress",
        "server_url": "mx:8001",
        "object_storage_type": "S3",
        "object_storage_endpoint_url": "http://minio:9000",
        "object_storage_region_name": "us-west-2",
        "registration_ttl_seconds": 90,
        "lease_ttl_seconds": 60,
        "max_transfer_attempts": 4,
        "max_replay_chain_length": 17,
        "rpc_timeout_seconds": 12.5,
    }
    engine.init_transfer_engine(engine.init_info_cls(**init_info))

    config = initialize.call_args.args[0]
    assert config.model_name == "policy"
    assert config.engine_context.model is model
    assert config.server_url == "mx:8001"
    assert config.registration_ttl_seconds == 90
    assert config.lease_ttl_seconds == 60
    assert config.max_transfer_attempts == 4
    assert config.max_replay_chain_length == 17
    assert config.rpc_timeout_seconds == 12.5
    assert config.object_storage.storage_type is ObjectStorageType.S3
    assert config.object_storage.initial_base_version_id == "base-a"
    assert config.object_storage.seed_checkpoint_path == "/models/launch"
    assert config.object_storage.refit_checkpoint_dir == "/cache/modelexpress"
    assert config.object_storage.refit_checkpoint_max_size_gb == 500
    assert config.object_storage.endpoint_url == "http://minio:9000"
    assert config.object_storage.region_name == "us-west-2"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"object_storage_type": "S3"},
        {"initial_base_version_id": "base-a"},
        {"seed_checkpoint_path": "/models/launch"},
        {"refit_checkpoint_dir": "/cache/modelexpress"},
        {"object_storage_endpoint_url": "http://minio:9000"},
        {"object_storage_region_name": "us-west-2"},
    ],
)
def test_weight_transfer_engine_requires_complete_object_storage_init(
    monkeypatch,
    kwargs,
):
    engine, _client = _engine(monkeypatch, initialize=False)

    with pytest.raises(ValueError, match="object storage requires"):
        engine.init_transfer_engine(engine.init_info_cls(**kwargs))


def test_weight_transfer_engine_rejects_unknown_object_storage_type(monkeypatch):
    engine, _client = _engine(monkeypatch, initialize=False)

    with pytest.raises(ValueError, match="unsupported object_storage_type"):
        engine.init_transfer_engine(
            engine.init_info_cls(
                object_storage_type="UNKNOWN",
                initial_base_version_id="base-a",
                seed_checkpoint_path="/models/launch",
                refit_checkpoint_dir="/cache/modelexpress",
            )
        )


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


def test_weight_transfer_engine_logs_lifecycle(monkeypatch, caplog):
    engine, client = _engine(monkeypatch, initialize=False)
    staged = SimpleNamespace(
        version_id="version-a",
        metrics={},
        release=MagicMock(),
    )
    client.stage_weight.return_value = staged
    client.apply_weight.return_value = {}

    with caplog.at_level(logging.INFO, logger=weight_transfer_engine.__name__):
        engine.init_transfer_engine(engine.init_info_cls())
        engine.start_weight_update()
        engine.update_weights({"version_id": "version-a"})
        engine.finish_weight_update()

    messages = [record.getMessage() for record in caplog.records]
    assert "ModelExpress weight transfer initialized model=test/model" in messages
    assert "ModelExpress weight update started" in messages
    assert "ModelExpress weight update receiving version=version-a" in messages
    assert "ModelExpress weight update applied version=version-a" in messages
    assert "ModelExpress weight update finished version=version-a" in messages


def test_weight_transfer_engine_logs_receiver_metrics(monkeypatch, caplog):
    engine, client = _engine(monkeypatch)
    staged = SimpleNamespace(
        metrics={
            "perf/mx_receive_prepare_time": 2.0,
            "perf/not_numeric": "ignored",
            "receiver/detail": 7.0,
        },
        release=MagicMock(),
    )
    client.stage_weight.return_value = staged
    client.apply_weight.return_value = {
        "perf/mx_receive_install_time": 3.0,
        "receiver/attempts": 1,
    }
    clock = iter([10.0, 14.5])
    monkeypatch.setattr(weight_transfer_engine, "perf_counter", lambda: next(clock))

    with caplog.at_level(logging.INFO, logger=weight_transfer_engine.__name__):
        engine.start_weight_update()
        engine.update_weights({"version_id": "version-a"})

    assert [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("ModelExpress receiver metric")
    ] == [
        "ModelExpress receiver metric perf/mx_receive_install_time=3.0",
        "ModelExpress receiver metric perf/mx_receive_prepare_time=2.0",
        "ModelExpress receiver metric perf/mx_receive_stage_weight_time=4.5",
    ]
    staged.release.assert_not_called()

    engine.finish_weight_update()
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
