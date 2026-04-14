# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ModelStreamerStrategy."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from modelexpress import p2p_pb2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_load_context(**overrides):
    """Return a LoadContext with mocked dependencies."""
    from modelexpress.load_strategy import LoadContext

    defaults = dict(
        vllm_config=MagicMock(),
        model_config=MagicMock(),
        load_config=MagicMock(),
        target_device=torch.device("cpu"),
        global_rank=0,
        device_id=0,
        identity=p2p_pb2.SourceIdentity(model_name="test-model"),
        mx_client=MagicMock(),
        worker_id="test-worker",
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

    @patch("modelexpress.load_strategy.model_streamer_strategy.publish_metadata")
    @patch("modelexpress.load_strategy.model_streamer_strategy.register_tensors")
    @patch("modelexpress.load_strategy.model_streamer_strategy.capture_tensor_attrs")
    def test_success_path_s3(self, mock_capture, mock_register, mock_publish):
        mock_capture.return_value.__enter__ = MagicMock()
        mock_capture.return_value.__exit__ = MagicMock(return_value=False)

        model = MagicMock()
        ctx = _make_load_context()
        strategy = self._make_strategy()

        with patch.dict("os.environ", {"MX_MODEL_URI": "s3://bucket/model"}):
            with patch(
                "modelexpress.load_strategy.model_streamer_strategy."
                "ModelStreamerStrategy._stream_weights"
            ) as mock_stream:
                mock_stream.return_value = iter([
                    ("layer.0.weight", torch.randn(4, 4)),
                    ("layer.1.weight", torch.randn(4, 4)),
                ])
                with patch(
                    "vllm.model_executor.model_loader.utils.process_weights_after_loading"
                ):
                    result = strategy.load(model, ctx)

        assert result is True
        model.load_weights.assert_called_once()
        mock_register.assert_called_once_with(model, ctx)
        mock_publish.assert_called_once_with(ctx)

    @patch("modelexpress.load_strategy.model_streamer_strategy.publish_metadata")
    @patch("modelexpress.load_strategy.model_streamer_strategy.register_tensors")
    @patch("modelexpress.load_strategy.model_streamer_strategy.capture_tensor_attrs")
    def test_success_path_local(self, mock_capture, mock_register, mock_publish):
        """load() works with a local path in MX_MODEL_URI."""
        mock_capture.return_value.__enter__ = MagicMock()
        mock_capture.return_value.__exit__ = MagicMock(return_value=False)

        model = MagicMock()
        ctx = _make_load_context()
        strategy = self._make_strategy()

        with patch.dict("os.environ", {"MX_MODEL_URI": "/models/llama"}):
            with patch(
                "modelexpress.load_strategy.model_streamer_strategy."
                "ModelStreamerStrategy._stream_weights"
            ) as mock_stream:
                mock_stream.return_value = iter([
                    ("layer.0.weight", torch.randn(4, 4)),
                ])
                with patch(
                    "vllm.model_executor.model_loader.utils.process_weights_after_loading"
                ):
                    result = strategy.load(model, ctx)

        assert result is True
        mock_stream.assert_called_once_with("/models/llama", ctx)

    @patch("modelexpress.load_strategy.model_streamer_strategy.publish_metadata")
    @patch("modelexpress.load_strategy.model_streamer_strategy.register_tensors")
    def test_env_overrides_model_config(self, mock_register, mock_publish):
        """MX_MODEL_URI takes priority over model_config.model."""
        model = MagicMock()
        model_config = MagicMock()
        model_config.model = "/models/local-llama"
        ctx = _make_load_context(model_config=model_config)
        strategy = self._make_strategy()

        with patch.dict("os.environ", {"MX_MODEL_URI": "s3://bucket/model"}):
            with patch(
                "modelexpress.load_strategy.model_streamer_strategy."
                "ModelStreamerStrategy._stream_weights"
            ) as mock_stream:
                mock_stream.side_effect = RuntimeError("expected")
                strategy.load(model, ctx)

        mock_stream.assert_called_once_with("s3://bucket/model", ctx)

    @patch("modelexpress.load_strategy.model_streamer_strategy.publish_metadata")
    @patch("modelexpress.load_strategy.model_streamer_strategy.register_tensors")
    def test_returns_false_on_error(self, mock_register, mock_publish):
        model = MagicMock()
        ctx = _make_load_context()
        strategy = self._make_strategy()

        with patch.dict("os.environ", {"MX_MODEL_URI": "s3://bucket/model"}):
            with patch(
                "modelexpress.load_strategy.model_streamer_strategy."
                "ModelStreamerStrategy._stream_weights",
                side_effect=RuntimeError("S3 connection failed"),
            ):
                result = strategy.load(model, ctx)

        assert result is False
        mock_register.assert_not_called()
        mock_publish.assert_not_called()


# ---------------------------------------------------------------------------
# TestStreamWeights
# ---------------------------------------------------------------------------


class TestStreamWeights:
    def _make_strategy(self):
        from modelexpress.load_strategy.model_streamer_strategy import ModelStreamerStrategy
        return ModelStreamerStrategy()

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
                return strategy_name == "default"
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
                return strategy_name == "default"
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
