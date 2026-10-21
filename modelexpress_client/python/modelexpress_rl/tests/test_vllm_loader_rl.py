# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RL policy selection for the shared vLLM loader."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from modelexpress_rl.inference.engines.vllm.loader import RLMxModelLoader


def test_rl_loader_runs_rl_chain():
    loader = object.__new__(RLMxModelLoader)
    model, ctx = MagicMock(), MagicMock()

    with patch(
        "modelexpress_rl.inference.engines.vllm.loader.RLLoadStrategyChain.run",
        return_value=model,
    ) as run:
        assert loader._run_load_strategy_chain(model, ctx) is model

    run.assert_called_once_with(model, ctx)


def test_rl_loader_defers_download():
    loader = object.__new__(RLMxModelLoader)
    with patch("modelexpress.engines.vllm.loader.DefaultModelLoader") as default:
        assert loader.download_model(MagicMock()) is None
    default.assert_not_called()


def test_rl_drafter_is_rejected_before_initialization():
    loader = object.__new__(RLMxModelLoader)
    model_config = MagicMock(dtype=MagicMock(), runner_type="draft")

    with patch(
        "modelexpress.engines.vllm.loader.build_vllm_load_context"
    ) as build_context, patch(
        "modelexpress.engines.vllm.loader.initialize_model"
    ) as initialize, pytest.raises(
        ValueError,
        match="RL initial loading does not support speculative draft models",
    ):
        loader.load_model(MagicMock(), model_config)

    build_context.assert_not_called()
    initialize.assert_not_called()


def test_rl_log_emits_once_when_vllm_handler_arrives_late(monkeypatch):
    from modelexpress_rl.inference.engines import configure_engine_logging

    vllm_logger = logging.getLogger("vllm")
    mx_logger = logging.getLogger("modelexpress")
    rl_logger = logging.getLogger("modelexpress_rl")
    handler = logging.Handler()
    handler.emit = MagicMock()
    old_vllm_handlers = list(vllm_logger.handlers)
    old_vllm_level = vllm_logger.level
    old_mx_handlers = list(mx_logger.handlers)
    old_mx_level = mx_logger.level
    old_mx_propagate = mx_logger.propagate
    old_rl_handlers = list(rl_logger.handlers)
    old_rl_level = rl_logger.level
    monkeypatch.delenv("MODEL_EXPRESS_LOG_LEVEL", raising=False)
    try:
        vllm_logger.handlers.clear()
        vllm_logger.setLevel(logging.DEBUG)
        mx_logger.handlers.clear()
        mx_logger.setLevel(logging.NOTSET)
        mx_logger.propagate = False
        rl_logger.handlers.clear()
        rl_logger.setLevel(logging.NOTSET)

        configure_engine_logging("vllm")
        vllm_logger.addHandler(handler)
        configure_engine_logging("vllm")
        configure_engine_logging("vllm")
        logging.getLogger("modelexpress_rl.inference.session").info("RL ready")

        handler.emit.assert_called_once()
        assert handler.emit.call_args.args[0].getMessage() == "RL ready"
    finally:
        vllm_logger.handlers[:] = old_vllm_handlers
        vllm_logger.setLevel(old_vllm_level)
        mx_logger.handlers[:] = old_mx_handlers
        mx_logger.setLevel(old_mx_level)
        mx_logger.propagate = old_mx_propagate
        rl_logger.handlers[:] = old_rl_handlers
        rl_logger.setLevel(old_rl_level)


@pytest.mark.parametrize("policy, uses_rl", [("INFERENCE", False), ("RL", True)])
def test_plugin_registers_rl_loader_only_for_rl_policy(monkeypatch, policy, uses_rl):
    from modelexpress.engines.vllm.loader import MxModelLoader
    from modelexpress_rl.inference.engines.vllm import registration
    from vllm.model_executor import model_loader

    monkeypatch.setenv("MX_LOAD_STRATEGY_CHAIN", policy)
    registry = {"modelexpress": MxModelLoader, "mx": MxModelLoader}
    monkeypatch.setattr(model_loader, "_LOAD_FORMAT_TO_MODEL_LOADER", registry, raising=False)
    monkeypatch.setattr(registration, "register_modelexpress", lambda: None)
    monkeypatch.setattr(registration, "register_plugin_weight_transfer_engine", lambda: None)
    registration.register()

    expected = RLMxModelLoader if uses_rl else MxModelLoader
    assert registry == {"modelexpress": expected, "mx": expected}


@pytest.mark.parametrize(
    "old_entry",
    [
        ("modelexpress_rl.inference.engines.vllm.weight_transfer_engine", "OldEngine"),
        ("vllm.distributed.weight_transfer.native", "NativeEngine"),
    ],
)
def test_plugin_preserves_weight_transfer_entry_and_is_idempotent(monkeypatch, old_entry):
    from modelexpress_rl.inference.engines.vllm import registration
    from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

    registry = {
        "modelexpress": old_entry,
        "other": ("other.engine", "OtherEngine"),
    }
    monkeypatch.setattr(WeightTransferEngineFactory, "_registry", registry)
    monkeypatch.setenv("MX_LOAD_STRATEGY_CHAIN", "INFERENCE")
    monkeypatch.setattr(registration, "register_modelexpress", lambda: None)
    monkeypatch.setattr(registration, "configure_engine_logging", lambda engine: None)

    registration.register()
    registration.register()

    assert registry["modelexpress"] is old_entry
    assert registry["other"] == ("other.engine", "OtherEngine")


def test_plugin_registers_missing_weight_transfer_entry_once(monkeypatch):
    from modelexpress_rl.inference.engines.vllm import registration
    from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

    monkeypatch.setattr(WeightTransferEngineFactory, "_registry", {})
    registration.register_plugin_weight_transfer_engine()
    entry = WeightTransferEngineFactory._registry["modelexpress"]
    registration.register_plugin_weight_transfer_engine()
    assert WeightTransferEngineFactory._registry["modelexpress"] is entry
