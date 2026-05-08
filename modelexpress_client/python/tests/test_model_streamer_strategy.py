# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ModelStreamerStrategy."""

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
    def discover_tensors(self, result: LoadResult):
        return {}

    def after_weight_iter_load(self, result: LoadResult):
        return result

    def apply_weight_iter(self, result: LoadResult, weights_iter):
        if result.model is not None:
            result.model.load_weights(weights_iter)
        return result


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
        mock_stream.assert_called_once_with("s3://bucket/model", ctx)
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
        mock_stream.assert_called_once_with("/models/llama", ctx)

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

        mock_stream.assert_called_once_with("s3://bucket/from-config", ctx)

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

    def _make_streamer_module(self, file_uris, tensors):
        mock_streamer_instance = MagicMock()
        mock_streamer_instance.__enter__ = MagicMock(return_value=mock_streamer_instance)
        mock_streamer_instance.__exit__ = MagicMock(return_value=False)
        mock_streamer_instance.files_to_tensors_metadata = {f: [MagicMock()] for f in file_uris}
        mock_streamer_instance.get_tensors.return_value = iter(tensors)
        mock_streamer_cls = MagicMock(return_value=mock_streamer_instance)
        return (
            MagicMock(
                list_safetensors=MagicMock(return_value=file_uris),
                SafetensorsStreamer=mock_streamer_cls,
            ),
            mock_streamer_instance,
        )

    def test_raises_when_no_files(self):
        strategy = self._make_strategy()
        ctx = _make_load_context()

        mock_list = MagicMock(return_value=[])
        mock_streamer = MagicMock()

        with patch.dict("sys.modules", {
            "runai_model_streamer": MagicMock(
                list_safetensors=mock_list,
                SafetensorsStreamer=mock_streamer,
            ),
        }):
            with pytest.raises(FileNotFoundError, match="No safetensors files found"):
                list(strategy._stream_weights("s3://empty-bucket/model", ctx))

    def _make_ctx_with_tp(
        self,
        tp_size: int,
        device_id: int = 2,
        device: str = "cuda",
    ):
        return _make_load_context(
            device_id=device_id,
            target_device=torch.device(device),
            identity=p2p_pb2.SourceIdentity(
                model_name="test-model",
                tensor_parallel_size=tp_size,
            ),
        )

    def test_distributed_disabled_by_default(self):
        strategy = self._make_strategy()
        ctx = self._make_ctx_with_tp(tp_size=4, device_id=2)
        file_uris = ["s3://bucket/model.safetensors"]
        tensors = [("w", torch.randn(2, 2))]

        module, streamer_instance = self._make_streamer_module(file_uris, tensors)
        with patch.dict("sys.modules", {"runai_model_streamer": module}):
            with patch.dict("os.environ", {}, clear=True):
                list(strategy._stream_weights("s3://bucket/model", ctx))

        streamer_instance.stream_files.assert_called_once_with(file_uris)

    def test_distributed_enabled_by_mx_ms_distributed_1(self):
        strategy = self._make_strategy()
        ctx = self._make_ctx_with_tp(tp_size=4, device_id=2)
        file_uris = ["s3://bucket/model.safetensors"]
        tensors = [("w", torch.randn(2, 2))]

        module, streamer_instance = self._make_streamer_module(file_uris, tensors)
        with patch.dict("sys.modules", {"runai_model_streamer": module}):
            with patch.dict("os.environ", {"MX_MS_DISTRIBUTED": "1"}):
                list(strategy._stream_weights("s3://bucket/model", ctx))

        streamer_instance.stream_files.assert_called_once_with(
            file_uris, device="cuda:2", is_distributed=True
        )

    def test_distributed_disabled_when_tp_is_1(self):
        strategy = self._make_strategy()
        ctx = self._make_ctx_with_tp(tp_size=1, device_id=0)
        file_uris = ["s3://bucket/model.safetensors"]
        tensors = [("w", torch.randn(2, 2))]

        module, streamer_instance = self._make_streamer_module(file_uris, tensors)
        with patch.dict("sys.modules", {"runai_model_streamer": module}):
            with patch.dict("os.environ", {"MX_MS_DISTRIBUTED": "1"}):
                list(strategy._stream_weights("s3://bucket/model", ctx))

        streamer_instance.stream_files.assert_called_once_with(file_uris)

    def test_distributed_disabled_on_non_cuda_platform(self):
        strategy = self._make_strategy()
        ctx = self._make_ctx_with_tp(tp_size=4, device_id=2, device="cpu")
        file_uris = ["s3://bucket/model.safetensors"]
        tensors = [("w", torch.randn(2, 2))]

        module, streamer_instance = self._make_streamer_module(file_uris, tensors)
        with patch.dict("sys.modules", {"runai_model_streamer": module}):
            with patch.dict("os.environ", {"MX_MS_DISTRIBUTED": "1"}):
                list(strategy._stream_weights("s3://bucket/model", ctx))

        streamer_instance.stream_files.assert_called_once_with(file_uris)

    def test_distributed_device_id_used_in_cuda_device(self):
        strategy = self._make_strategy()
        ctx = self._make_ctx_with_tp(tp_size=8, device_id=5)
        file_uris = ["s3://bucket/model.safetensors"]
        tensors = [("w", torch.randn(2, 2))]

        module, streamer_instance = self._make_streamer_module(file_uris, tensors)
        with patch.dict("sys.modules", {"runai_model_streamer": module}):
            with patch.dict("os.environ", {"MX_MS_DISTRIBUTED": "1"}):
                list(strategy._stream_weights("s3://bucket/model", ctx))

        streamer_instance.stream_files.assert_called_once_with(
            file_uris, device="cuda:5", is_distributed=True
        )


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
