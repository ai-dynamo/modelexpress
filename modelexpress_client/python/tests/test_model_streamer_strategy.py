# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ModelStreamerStrategy."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from modelexpress import p2p_pb2
from modelexpress.adapter import EngineAdapter
from modelexpress.adapter import StrategyFailed
from modelexpress.load_strategy.context import LoadResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeAdapter(EngineAdapter):
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

    def build_model_streamer_weight_iter(self, model_uri: str, model=None):
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


# ---------------------------------------------------------------------------
# TestModelStreamerIsAvailable
# ---------------------------------------------------------------------------


class TestModelStreamerIsAvailable:
    def _make_strategy(self):
        from modelexpress.load_strategy.model_streamer_strategy import ModelStreamerStrategy
        return ModelStreamerStrategy()

    def test_available_with_s3_uri(self):
        ctx = _make_load_context()
        strategy = self._make_strategy()
        with patch.dict("os.environ", {"MX_MODEL_URI": "s3://bucket/model"}):
            with patch("importlib.util.find_spec", return_value=MagicMock()):
                assert strategy.is_available(ctx) is True

    def test_available_with_local_path_env(self):
        ctx = _make_load_context()
        strategy = self._make_strategy()
        with patch.dict("os.environ", {"MX_MODEL_URI": "/models/deepseek"}):
            with patch("importlib.util.find_spec", return_value=MagicMock()):
                assert strategy.is_available(ctx) is True

    def test_available_with_gcs_uri(self):
        ctx = _make_load_context()
        strategy = self._make_strategy()
        with patch.dict("os.environ", {"MX_MODEL_URI": "gs://bucket/model"}):
            with patch("importlib.util.find_spec", return_value=MagicMock()):
                assert strategy.is_available(ctx) is True

    def test_available_with_local_path(self):
        ctx = _make_load_context()
        strategy = self._make_strategy()
        with patch.dict("os.environ", {"MX_MODEL_URI": "/models/deepseek-ai/DeepSeek-V3"}):
            with patch("importlib.util.find_spec", return_value=MagicMock()):
                assert strategy.is_available(ctx) is True

    def test_available_with_hf_model_id(self):
        ctx = _make_load_context()
        strategy = self._make_strategy()
        with patch.dict("os.environ", {"MX_MODEL_URI": "deepseek-ai/DeepSeek-V3"}):
            with patch("importlib.util.find_spec", return_value=MagicMock()):
                assert strategy.is_available(ctx) is True

    def test_unavailable_no_env(self):
        ctx = _make_load_context()
        strategy = self._make_strategy()
        with patch.dict("os.environ", {}, clear=True):
            with patch("importlib.util.find_spec", return_value=MagicMock()):
                assert strategy.is_available(ctx) is False

    def test_unavailable_no_package(self):
        ctx = _make_load_context()
        strategy = self._make_strategy()
        with patch.dict("os.environ", {"MX_MODEL_URI": "s3://bucket/model"}):
            with patch("importlib.util.find_spec", return_value=None):
                assert strategy.is_available(ctx) is False


# ---------------------------------------------------------------------------
# TestModelStreamerLoad
# ---------------------------------------------------------------------------


class TestModelStreamerLoad:
    def _make_strategy(self):
        from modelexpress.load_strategy.model_streamer_strategy import ModelStreamerStrategy
        return ModelStreamerStrategy()

    def _make_ctx_with_uri(self, *, model_weights=None, model="ignored"):
        model_config = MagicMock()
        model_config.model_weights = model_weights
        model_config.model = model
        return _make_load_context(model_config=model_config)

    @patch("modelexpress.load_strategy.model_streamer_strategy.register_tensors")
    def test_success_path_s3(self, mock_register):
        model = MagicMock()
        ctx = self._make_ctx_with_uri(model_weights="s3://bucket/model")
        strategy = self._make_strategy()

        with patch(
            "modelexpress.load_strategy.model_streamer_strategy."
            "ModelStreamerStrategy._stream_weights"
        ) as mock_stream:
            mock_stream.return_value = iter([
                ("layer.0.weight", torch.randn(4, 4)),
                ("layer.1.weight", torch.randn(4, 4)),
            ])
            result = strategy.load(model, ctx)

        assert isinstance(result, LoadResult)
        assert result.model is model
        mock_stream.assert_called_once_with("s3://bucket/model", ctx, model)
        model.load_weights.assert_called_once()
        mock_register.assert_called_once_with(result, ctx)

    @patch("modelexpress.load_strategy.model_streamer_strategy.register_tensors")
    def test_success_path_local_falls_back_to_model(self, mock_register):
        """When model_weights is unset, the URI comes from model_config.model."""
        model = MagicMock()
        ctx = self._make_ctx_with_uri(model_weights=None, model="/models/llama")
        strategy = self._make_strategy()

        with patch(
            "modelexpress.load_strategy.model_streamer_strategy."
            "ModelStreamerStrategy._stream_weights"
        ) as mock_stream:
            mock_stream.return_value = iter([
                ("layer.0.weight", torch.randn(4, 4)),
            ])
            result = strategy.load(model, ctx)

        assert isinstance(result, LoadResult)
        mock_stream.assert_called_once_with("/models/llama", ctx, model)

    @patch("modelexpress.load_strategy.model_streamer_strategy.register_tensors")
    def test_uri_from_model_weights_not_from_env(self, mock_register):
        """The streaming URI comes from model_config, not from MX_MODEL_URI."""
        model = MagicMock()
        ctx = self._make_ctx_with_uri(
            model_weights="s3://bucket/from-config", model="/ignored"
        )
        strategy = self._make_strategy()

        with patch.dict("os.environ", {"MX_MODEL_URI": "s3://other/path-in-env"}):
            with patch(
                "modelexpress.load_strategy.model_streamer_strategy."
                "ModelStreamerStrategy._stream_weights"
            ) as mock_stream:
                mock_stream.side_effect = RuntimeError("expected")
                with pytest.raises(StrategyFailed, match="expected"):
                    strategy.load(model, ctx)

        mock_stream.assert_called_once_with("s3://bucket/from-config", ctx, model)

    @patch("modelexpress.load_strategy.model_streamer_strategy.register_tensors")
    def test_raises_strategy_failed_on_error(self, mock_register):
        model = MagicMock()
        ctx = self._make_ctx_with_uri(model_weights="s3://bucket/model")
        strategy = self._make_strategy()

        with patch(
            "modelexpress.load_strategy.model_streamer_strategy."
            "ModelStreamerStrategy._stream_weights",
            side_effect=RuntimeError("S3 connection failed"),
        ):
            with pytest.raises(StrategyFailed, match="S3 connection failed") as exc:
                strategy.load(model, ctx)

        assert exc.value.mutated is False
        mock_register.assert_not_called()

    @patch("modelexpress.load_strategy.model_streamer_strategy.register_tensors")
    def test_apply_weight_iter_failure_is_mutated(self, mock_register):
        model = MagicMock()
        adapter = _FakeAdapter()
        adapter.apply_weight_iter = MagicMock(side_effect=RuntimeError("partial load"))
        ctx = _make_load_context(adapter=adapter)
        strategy = self._make_strategy()

        with patch.dict("os.environ", {"MX_MODEL_URI": "s3://bucket/model"}):
            with patch(
                "modelexpress.load_strategy.model_streamer_strategy."
                "ModelStreamerStrategy._stream_weights",
                return_value=iter([("layer.0.weight", torch.randn(4, 4))]),
            ):
                with pytest.raises(StrategyFailed, match="partial load") as exc:
                    strategy.load(model, ctx)

        assert exc.value.mutated is True
        mock_register.assert_not_called()

    @patch("modelexpress.load_strategy.model_streamer_strategy.register_tensors")
    def test_after_weight_iter_failure_is_mutated(self, mock_register):
        model = MagicMock()
        adapter = _FakeAdapter()
        adapter.after_weight_iter_load = MagicMock(side_effect=RuntimeError("post load"))
        ctx = _make_load_context(adapter=adapter)
        strategy = self._make_strategy()

        with patch.dict("os.environ", {"MX_MODEL_URI": "s3://bucket/model"}):
            with patch(
                "modelexpress.load_strategy.model_streamer_strategy."
                "ModelStreamerStrategy._stream_weights",
                return_value=iter([("layer.0.weight", torch.randn(4, 4))]),
            ):
                with pytest.raises(StrategyFailed, match="post load") as exc:
                    strategy.load(model, ctx)

        assert exc.value.mutated is True
        mock_register.assert_not_called()


# ---------------------------------------------------------------------------
# TestStreamWeights
# ---------------------------------------------------------------------------


class TestStreamWeights:
    def _make_strategy(self):
        from modelexpress.load_strategy.model_streamer_strategy import ModelStreamerStrategy
        return ModelStreamerStrategy()

    def test_delegates_to_adapter_without_cloning(self):
        strategy = self._make_strategy()
        tensor = torch.randn(2, 2)
        adapter = _FakeAdapter()
        adapter.build_model_streamer_weight_iter = MagicMock(
            return_value=iter([("w", tensor)])
        )
        ctx = _make_load_context(adapter=adapter)

        model = torch.nn.Module()
        weights = list(strategy._stream_weights("s3://bucket/model", ctx, model))

        adapter.build_model_streamer_weight_iter.assert_called_once_with(
            "s3://bucket/model",
            model=model,
        )
        assert weights[0][0] == "w"
        assert weights[0][1] is tensor


class TestVllmModelStreamerIterator:
    def _make_adapter(self, *, tp_size: int = 8, extra_config=None):
        from modelexpress.engines.vllm.adapter import VllmAdapter

        load_config = SimpleNamespace(
            device=None,
            model_loader_extra_config=extra_config,
        )
        vllm_config = MagicMock()
        vllm_config.load_config = load_config
        vllm_config.device_config.device = "cuda"
        vllm_config.parallel_config.tensor_parallel_size = tp_size
        model_config = SimpleNamespace(revision="main")
        return VllmAdapter(vllm_config, model_config), load_config

    def _patch_runai_loader(self, tensors):
        loader_instance = MagicMock()
        loader_instance._get_weights_iterator.return_value = iter(tensors)
        loader_cls = MagicMock(return_value=loader_instance)
        module = SimpleNamespace(RunaiModelStreamerLoader=loader_cls)
        return patch.dict(
            sys.modules,
            {"vllm.model_executor.model_loader.runai_streamer_loader": module},
        ), loader_cls, loader_instance

    def test_uses_vllm_native_iterator_and_enables_distributed(self):
        tensor = torch.randn(2, 2)
        adapter, _load_config = self._make_adapter(tp_size=8)
        patcher, loader_cls, loader_instance = self._patch_runai_loader([("w", tensor)])

        with patcher, patch.dict("os.environ", {"MX_MS_DISTRIBUTED": "1"}):
            weights = list(adapter.build_model_streamer_weight_iter("az://models/model"))

        native_load_config = loader_cls.call_args.args[0]
        assert native_load_config.model_loader_extra_config["distributed"] is True
        loader_instance._get_weights_iterator.assert_called_once_with(
            "az://models/model", "main"
        )
        assert weights[0][0] == "w"
        assert weights[0][1] is tensor

    def test_preserves_existing_extra_config_when_distributed_disabled(self):
        adapter, _load_config = self._make_adapter(
            tp_size=1,
            extra_config={"concurrency": 4},
        )
        patcher, loader_cls, _loader_instance = self._patch_runai_loader([])

        with patcher, patch.dict("os.environ", {"MX_MS_DISTRIBUTED": "1"}):
            list(adapter.build_model_streamer_weight_iter("az://models/model"))

        native_load_config = loader_cls.call_args.args[0]
        assert native_load_config.model_loader_extra_config == {"concurrency": 4}

    def test_distributed_disabled_by_default_even_with_tp_gt_one(self):
        adapter, _load_config = self._make_adapter(
            tp_size=8,
            extra_config={"concurrency": 4},
        )
        patcher, loader_cls, _loader_instance = self._patch_runai_loader([])

        with patcher, patch.dict("os.environ", {}, clear=True):
            list(adapter.build_model_streamer_weight_iter("az://models/model"))

        native_load_config = loader_cls.call_args.args[0]
        assert native_load_config.model_loader_extra_config == {"concurrency": 4}


# ---------------------------------------------------------------------------
# TestChainOrder
# ---------------------------------------------------------------------------


class TestChainOrder:
    @patch(
        "modelexpress.load_strategy.rdma_strategy.RdmaStrategy.is_available",
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
    def test_model_streamer_in_chain(self, _d, _g, _ms, _r):
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

        assert call_order == ["rdma", "model_streamer", "gds", "default"]

    @patch(
        "modelexpress.load_strategy.rdma_strategy.RdmaStrategy.is_available",
        return_value=True,
    )
    @patch(
        "modelexpress.load_strategy.model_streamer_strategy."
        "ModelStreamerStrategy.is_available",
        return_value=False,
    )
    @patch(
        "modelexpress.load_strategy.gds_strategy.GdsStrategy.is_available",
        return_value=True,
    )
    @patch(
        "modelexpress.load_strategy.default_strategy.DefaultStrategy.is_available",
        return_value=True,
    )
    def test_skips_unavailable_strategy(self, _d, _g, _ms, _r):
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
            "modelexpress.load_strategy.gds_strategy.GdsStrategy.load",
            track_load("gds"),
        ), patch(
            "modelexpress.load_strategy.default_strategy.DefaultStrategy.load",
            track_load("default"),
        ):
            LoadStrategyChain.run(model, ctx)

        assert call_order == ["rdma", "gds", "default"]
