# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import logging.handlers
from unittest.mock import Mock, patch

from modelexpress import configure_trtllm_logging
from modelexpress.engines.trtllm.loader import MxModelLoader


def _loader_kwargs():
    return {
        "model_config": object(),
        "load_config": object(),
        "checkpoint_loader": object(),
        "checkpoint_dir": "/model",
        "native_loader_kwargs": {},
        "mapping": object(),
        "source_identity": object(),
        "prepare_post_transform_receiver": lambda _model: None,
        "transform_protocol_version": 1,
        "p2p_enabled": True,
        "mx_server_url": "mx:8001",
    }


def test_loader_runs_shared_strategy_chain(monkeypatch):
    context = Mock()
    context.adapter.rdma_loaded = True
    context.adapter.rdma_transform_protocol_version = 1
    context.adapter.current_model = None
    model = Mock()

    monkeypatch.setattr(
        "modelexpress.engines.trtllm.loader.build_trtllm_load_context",
        lambda **kwargs: context,
    )
    run = Mock(return_value=model)
    monkeypatch.setattr(
        "modelexpress.engines.trtllm.loader.LoadStrategyChain.run",
        run,
    )

    loader = MxModelLoader(**_loader_kwargs())

    assert loader.load_model(model) == {}
    run.assert_called_once_with(model, context)
    assert loader.p2p_succeeded


def test_loader_logs_same_start_and_completion_milestones(monkeypatch, caplog):
    context = Mock()
    context.global_rank = 3
    context.identity.model_name = "meta-llama/Llama-3.1-8B-Instruct"
    context.p2p_enabled = True
    context.adapter.rdma_loaded = True
    context.adapter.rdma_transform_protocol_version = 1
    context.adapter.current_model = None
    model = Mock()

    monkeypatch.setattr(
        "modelexpress.engines.trtllm.loader.build_trtllm_load_context",
        lambda **kwargs: context,
    )
    monkeypatch.setattr(
        "modelexpress.engines.trtllm.loader.LoadStrategyChain.run",
        Mock(return_value=model),
    )

    with (
        patch("modelexpress.engines.trtllm.loader.configure_trtllm_logging"),
        caplog.at_level(
            logging.INFO,
            logger="modelexpress.engines.trtllm.loader",
        ),
    ):
        loader = MxModelLoader(**_loader_kwargs())
        loader.load_model(model)

    messages = [record.getMessage() for record in caplog.records]
    assert messages[0] == (
        "[Worker 3] TRT-LLM MxModelLoader starting "
        "(model=meta-llama/Llama-3.1-8B-Instruct, p2p_enabled=True)"
    )
    assert messages[1].startswith(
        "[Worker 3] TRT-LLM MxModelLoader.load_model() COMPLETE in "
    )


def test_trtllm_logging_exposes_shared_strategy_logs(monkeypatch):
    monkeypatch.delenv("MODEL_EXPRESS_LOG_LEVEL", raising=False)
    mx_logger = logging.getLogger("modelexpress")
    trtllm_logger = logging.getLogger("TRT-LLM")
    saved_mx = (list(mx_logger.handlers), mx_logger.level)
    saved_trtllm = (list(trtllm_logger.handlers), trtllm_logger.level)
    handler = logging.handlers.MemoryHandler(capacity=10)

    try:
        mx_logger.handlers.clear()
        mx_logger.setLevel(logging.NOTSET)
        trtllm_logger.handlers.clear()
        trtllm_logger.addHandler(handler)
        trtllm_logger.setLevel(logging.INFO)

        configure_trtllm_logging()
        logging.getLogger("modelexpress.load_strategy").info(
            "Eligible loaders: ['rdma', 'default']"
        )

        assert mx_logger.handlers == [handler]
        assert mx_logger.level == logging.INFO
        assert len(handler.buffer) == 1
        assert handler.buffer[0].getMessage() == (
            "Eligible loaders: ['rdma', 'default']"
        )
    finally:
        mx_logger.handlers.clear()
        mx_logger.handlers.extend(saved_mx[0])
        mx_logger.setLevel(saved_mx[1])
        trtllm_logger.handlers.clear()
        trtllm_logger.handlers.extend(saved_trtllm[0])
        trtllm_logger.setLevel(saved_trtllm[1])


def test_loader_preserves_native_engine_value(monkeypatch):
    context = Mock()
    context.adapter.native_loaded = True
    context.adapter.rdma_loaded = False
    context.adapter.current_model = None
    model = Mock()
    native_value = Mock(name="ConsumableWeightsDict")

    monkeypatch.setattr(
        "modelexpress.engines.trtllm.loader.build_trtllm_load_context",
        lambda **kwargs: context,
    )
    monkeypatch.setattr(
        "modelexpress.engines.trtllm.loader.LoadStrategyChain.run",
        Mock(return_value=native_value),
    )
    loader = MxModelLoader(**_loader_kwargs())

    assert loader.load_model(model) is native_value


def test_native_publish_uses_existing_context(monkeypatch):
    context = Mock()
    context.adapter.native_loaded = True
    context.adapter.rdma_loaded = False
    context.adapter.current_model = None
    manager = context.nixl_manager
    model = Mock()

    monkeypatch.setattr(
        "modelexpress.engines.trtllm.loader.build_trtllm_load_context",
        lambda **kwargs: context,
    )
    register = Mock()
    publish = Mock()
    monkeypatch.setattr("modelexpress.engines.trtllm.loader.register_tensors", register)
    monkeypatch.setattr("modelexpress.engines.trtllm.loader.publish_metadata", publish)

    loader = MxModelLoader(**_loader_kwargs())
    loader.publish_model(model)

    context.accelerator_backend.synchronize.assert_called_once()
    manager.shutdown.assert_not_called()
    register.assert_called_once()
    publish.assert_called_once_with(context)


def test_rdma_target_rebuilds_registration_before_publish(monkeypatch):
    context = Mock()
    context.adapter.rdma_loaded = True
    context.adapter.current_model = None
    receive_manager = context.nixl_manager
    context.tensors = {"receive.weight": Mock()}
    model = Mock()

    monkeypatch.setattr(
        "modelexpress.engines.trtllm.loader.build_trtllm_load_context",
        lambda **kwargs: context,
    )
    register = Mock()
    publish = Mock()
    monkeypatch.setattr("modelexpress.engines.trtllm.loader.register_tensors", register)
    monkeypatch.setattr("modelexpress.engines.trtllm.loader.publish_metadata", publish)

    loader = MxModelLoader(**_loader_kwargs())
    loader.publish_model(model)

    receive_manager.shutdown.assert_called_once_with()
    assert context.nixl_manager is None
    assert context.tensors == {}
    context.accelerator_backend.synchronize.assert_called_once_with()
    register.assert_called_once()
    publish.assert_called_once_with(context)


def test_native_publish_is_best_effort_when_synchronization_fails(monkeypatch):
    context = Mock()
    context.global_rank = 0
    context.adapter.current_model = None
    context.accelerator_backend.synchronize.side_effect = ValueError("sync")

    monkeypatch.setattr(
        "modelexpress.engines.trtllm.loader.build_trtllm_load_context",
        lambda **kwargs: context,
    )
    register = Mock()
    monkeypatch.setattr(
        "modelexpress.engines.trtllm.loader.register_tensors",
        register,
    )
    loader = MxModelLoader(**_loader_kwargs())

    loader.publish_model(Mock())

    register.assert_not_called()


def test_cleanup_releases_all_resources_when_shutdown_fails(monkeypatch):
    context = Mock()
    manager = context.nixl_manager
    context.nixl_manager.shutdown.side_effect = RuntimeError("nixl")
    context.mx_client.close.side_effect = RuntimeError("client")

    monkeypatch.setattr(
        "modelexpress.engines.trtllm.loader.build_trtllm_load_context",
        lambda **kwargs: context,
    )
    unpublish = Mock(side_effect=RuntimeError("metadata"))
    monkeypatch.setattr(
        "modelexpress.engines.trtllm.loader.unpublish_metadata",
        unpublish,
    )
    loader = MxModelLoader(**_loader_kwargs())

    loader.cleanup()

    unpublish.assert_called_once_with(context)
    manager.shutdown.assert_called_once_with()
    assert context.nixl_manager is None
    context.mx_client.close.assert_called_once_with()
