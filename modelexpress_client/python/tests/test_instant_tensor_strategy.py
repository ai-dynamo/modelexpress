# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for InstantTensorStrategy."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from modelexpress import p2p_pb2
from modelexpress.adapter import EngineAdapter, StrategyFailed
from modelexpress.load_strategy.context import LoadResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeAdapter(EngineAdapter):
    """Adapter that implements the InstantTensor capability."""

    def __init__(self, *, is_cuda_alike: bool = True):
        self._is_cuda_alike = is_cuda_alike

    def discover_tensors(self, result: LoadResult):
        return {}

    def is_cuda_alike(self) -> bool:
        return self._is_cuda_alike

    def after_weight_iter_load(self, result: LoadResult):
        return result

    def apply_weight_iter(self, result: LoadResult, weights_iter):
        if result.model is not None:
            result.model.load_weights(weights_iter)
        return result

    def build_instanttensor_weight_iter(self, model=None):
        return iter(())


class _NoInstantTensorAdapter(EngineAdapter):
    """Adapter that does NOT implement the InstantTensor capability."""

    def is_cuda_alike(self) -> bool:
        return True


class _IterOnlyAdapter(EngineAdapter):
    """Implements the iterator but not apply_weight_iter (should be ineligible)."""

    def is_cuda_alike(self) -> bool:
        return True

    def build_instanttensor_weight_iter(self, model=None):
        return iter(())


def _make_load_context(**overrides):
    """Return a LoadContext with mocked dependencies."""
    from modelexpress.load_strategy import LoadContext

    defaults = dict(
        model_config=MagicMock(),
        load_config=MagicMock(),
        target_device=torch.device("cpu"),
        global_rank=0,
        worker_rank=0,
        device_id=0,
        identity=p2p_pb2.SourceIdentity(
            model_name="test-model",
            tensor_parallel_size=1,
        ),
        mx_client=MagicMock(),
        worker_id="test-worker",
        adapter=_FakeAdapter(),
    )
    defaults.update(overrides)
    return LoadContext(**defaults)


def _make_strategy():
    from modelexpress.load_strategy.instant_tensor_strategy import InstantTensorStrategy
    return InstantTensorStrategy()


# ---------------------------------------------------------------------------
# TestInstantTensorIsAvailable
# ---------------------------------------------------------------------------


class TestInstantTensorIsAvailable:
    def test_available_by_default(self):
        ctx = _make_load_context()
        strategy = _make_strategy()
        with patch.dict("os.environ", {}, clear=True):
            with patch("importlib.util.find_spec", return_value=MagicMock()):
                assert strategy.is_available(ctx) is True

    def test_available_when_explicitly_enabled(self):
        ctx = _make_load_context()
        strategy = _make_strategy()
        with patch.dict("os.environ", {"MX_INSTANT_TENSOR": "1"}):
            with patch("importlib.util.find_spec", return_value=MagicMock()):
                assert strategy.is_available(ctx) is True

    def test_unavailable_when_disabled(self):
        ctx = _make_load_context()
        strategy = _make_strategy()
        with patch.dict("os.environ", {"MX_INSTANT_TENSOR": "0"}):
            with patch("importlib.util.find_spec", return_value=MagicMock()):
                assert strategy.is_available(ctx) is False

    def test_unavailable_no_package(self):
        ctx = _make_load_context()
        strategy = _make_strategy()
        with patch.dict("os.environ", {"MX_INSTANT_TENSOR": "1"}):
            with patch("importlib.util.find_spec", return_value=None):
                assert strategy.is_available(ctx) is False

    def test_unavailable_not_cuda(self):
        ctx = _make_load_context(adapter=_FakeAdapter(is_cuda_alike=False))
        strategy = _make_strategy()
        with patch.dict("os.environ", {"MX_INSTANT_TENSOR": "1"}):
            with patch("importlib.util.find_spec", return_value=MagicMock()):
                assert strategy.is_available(ctx) is False

    def test_unavailable_when_adapter_lacks_capability(self):
        ctx = _make_load_context(adapter=_NoInstantTensorAdapter())
        strategy = _make_strategy()
        with patch.dict("os.environ", {"MX_INSTANT_TENSOR": "1"}):
            with patch("importlib.util.find_spec", return_value=MagicMock()):
                assert strategy.is_available(ctx) is False

    def test_unavailable_when_adapter_lacks_apply_weight_iter(self):
        # Implements the iterator but not apply_weight_iter: must fall through
        # cleanly instead of hitting the gated apply_weight_iter default at load.
        ctx = _make_load_context(adapter=_IterOnlyAdapter())
        strategy = _make_strategy()
        with patch.dict("os.environ", {"MX_INSTANT_TENSOR": "1"}):
            with patch("importlib.util.find_spec", return_value=MagicMock()):
                assert strategy.is_available(ctx) is False


# ---------------------------------------------------------------------------
# TestInstantTensorLoad
# ---------------------------------------------------------------------------


class TestInstantTensorLoad:
    @patch("modelexpress.load_strategy.instant_tensor_strategy.register_tensors")
    def test_success_path(self, mock_register):
        model = MagicMock()
        adapter = _FakeAdapter()
        adapter.build_instanttensor_weight_iter = MagicMock(
            return_value=iter([
                ("layer.0.weight", torch.randn(4, 4)),
                ("layer.1.weight", torch.randn(4, 4)),
            ])
        )
        ctx = _make_load_context(adapter=adapter)
        strategy = _make_strategy()

        result = strategy.load(model, ctx)

        assert isinstance(result, LoadResult)
        assert result.model is model
        adapter.build_instanttensor_weight_iter.assert_called_once_with(model=model)
        model.load_weights.assert_called_once()
        mock_register.assert_called_once_with(result, ctx)

    @patch("modelexpress.load_strategy.instant_tensor_strategy.register_tensors")
    def test_iterator_setup_failure_is_not_mutated(self, mock_register):
        model = MagicMock()
        adapter = _FakeAdapter()
        adapter.build_instanttensor_weight_iter = MagicMock(
            side_effect=RuntimeError("no safetensors found")
        )
        ctx = _make_load_context(adapter=adapter)
        strategy = _make_strategy()

        with pytest.raises(StrategyFailed, match="no safetensors found") as exc:
            strategy.load(model, ctx)

        assert exc.value.mutated is False
        mock_register.assert_not_called()

    @patch("modelexpress.load_strategy.instant_tensor_strategy.register_tensors")
    def test_apply_weight_iter_failure_is_mutated(self, mock_register):
        model = MagicMock()
        adapter = _FakeAdapter()
        adapter.build_instanttensor_weight_iter = MagicMock(
            return_value=iter([("layer.0.weight", torch.randn(4, 4))])
        )
        adapter.apply_weight_iter = MagicMock(side_effect=RuntimeError("partial load"))
        ctx = _make_load_context(adapter=adapter)
        strategy = _make_strategy()

        with pytest.raises(StrategyFailed, match="partial load") as exc:
            strategy.load(model, ctx)

        assert exc.value.mutated is True
        mock_register.assert_not_called()

    @patch("modelexpress.load_strategy.instant_tensor_strategy.register_tensors")
    def test_after_weight_iter_failure_is_mutated(self, mock_register):
        model = MagicMock()
        adapter = _FakeAdapter()
        adapter.build_instanttensor_weight_iter = MagicMock(
            return_value=iter([("layer.0.weight", torch.randn(4, 4))])
        )
        adapter.after_weight_iter_load = MagicMock(side_effect=RuntimeError("post load"))
        ctx = _make_load_context(adapter=adapter)
        strategy = _make_strategy()

        with pytest.raises(StrategyFailed, match="post load") as exc:
            strategy.load(model, ctx)

        assert exc.value.mutated is True
        mock_register.assert_not_called()


# ---------------------------------------------------------------------------
# TestVllmInstantTensorIterator
# ---------------------------------------------------------------------------


class TestVllmInstantTensorIterator:
    def _make_adapter(self, *, extra_config=None):
        from modelexpress.engines.vllm.adapter import VllmAdapter

        load_config = SimpleNamespace(
            device=None,
            load_format="modelexpress",
            model_loader_extra_config=extra_config,
        )
        vllm_config = MagicMock()
        vllm_config.load_config = load_config
        vllm_config.device_config.device = "cuda"
        vllm_config.parallel_config.tensor_parallel_size = 8
        model_config = SimpleNamespace(revision="main", model="/models/qwen")
        return VllmAdapter(vllm_config, model_config), model_config

    def _patch_default_loader(self, tensors):
        loader_instance = MagicMock()
        loader_instance.get_all_weights.return_value = iter(tensors)
        loader_cls = MagicMock(return_value=loader_instance)
        module = SimpleNamespace(DefaultModelLoader=loader_cls)
        return patch.dict(
            sys.modules,
            {"vllm.model_executor.model_loader.default_loader": module},
        ), loader_cls, loader_instance

    def test_sets_instanttensor_format_and_calls_get_all_weights(self):
        tensor = torch.randn(2, 2)
        adapter, model_config = self._make_adapter()
        patcher, loader_cls, loader_instance = self._patch_default_loader([("w", tensor)])
        model = MagicMock()

        with patcher:
            weights = list(adapter.build_instanttensor_weight_iter(model=model))

        native_load_config = loader_cls.call_args.args[0]
        assert native_load_config.load_format == "instanttensor"
        loader_instance.get_all_weights.assert_called_once_with(model_config, model)
        assert weights[0][0] == "w"
        assert weights[0][1] is tensor

    def test_requires_model(self):
        adapter, _model_config = self._make_adapter()
        with pytest.raises(RuntimeError, match="requires the initialized model"):
            adapter.build_instanttensor_weight_iter(model=None)


# ---------------------------------------------------------------------------
# TestChainOrder
# ---------------------------------------------------------------------------


class TestChainOrder:
    @patch(
        "modelexpress.load_strategy.rdma_strategy.RdmaStrategy.is_available",
        return_value=True,
    )
    @patch(
        "modelexpress.load_strategy.instant_tensor_strategy."
        "InstantTensorStrategy.is_available",
        return_value=True,
    )
    @patch(
        "modelexpress.load_strategy.model_streamer_strategy."
        "ModelStreamerStrategy.is_available",
        return_value=True,
    )
    @patch(
        "modelexpress.load_strategy.gds_strategy.GdsStrategy.is_available",
        return_value=True,
    )
    @patch(
        "modelexpress.load_strategy.default_strategy.DefaultStrategy.is_available",
        return_value=True,
    )
    def test_instant_tensor_runs_after_rdma_before_model_streamer(
        self, _d, _g, _ms, _it, _r
    ):
        from modelexpress.load_strategy import LoadStrategyChain

        call_order: list[str] = []

        def track_load(strategy_name):
            def _load(self_or_model, *args, **kwargs):
                call_order.append(strategy_name)
                if strategy_name != "default":
                    raise StrategyFailed(f"{strategy_name} miss")
                return args[0]
            return _load

        model = MagicMock()
        ctx = _make_load_context()

        with patch(
            "modelexpress.load_strategy.rdma_strategy.RdmaStrategy.load",
            track_load("rdma"),
        ), patch(
            "modelexpress.load_strategy.instant_tensor_strategy."
            "InstantTensorStrategy.load",
            track_load("instant_tensor"),
        ), patch(
            "modelexpress.load_strategy.model_streamer_strategy."
            "ModelStreamerStrategy.load",
            track_load("model_streamer"),
        ), patch(
            "modelexpress.load_strategy.gds_strategy.GdsStrategy.load",
            track_load("gds"),
        ), patch(
            "modelexpress.load_strategy.default_strategy.DefaultStrategy.load",
            track_load("default"),
        ):
            LoadStrategyChain.run(model, ctx)

        assert call_order == [
            "rdma",
            "instant_tensor",
            "model_streamer",
            "gds",
            "default",
        ]
