# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ModelStreamerStrategy."""

import json
import sys
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
import torch

from modelexpress import p2p_pb2


# ---------------------------------------------------------------------------
# boto3 / botocore mock helpers (not installed in dev environment)
# ---------------------------------------------------------------------------


class _MockClientError(Exception):
    """Stand-in for botocore.exceptions.ClientError."""

    def __init__(self, error_response, operation_name):
        self.response = error_response
        self.operation_name = operation_name
        super().__init__(f"{operation_name}: {error_response}")


def _boto3_modules():
    """Return a dict of mock modules to inject into sys.modules for boto3/botocore."""
    mock_boto3 = MagicMock()
    mock_botocore = MagicMock()
    mock_botocore_exceptions = MagicMock()
    mock_botocore_exceptions.ClientError = _MockClientError
    mock_botocore.exceptions = mock_botocore_exceptions
    return {
        "boto3": mock_boto3,
        "botocore": mock_botocore,
        "botocore.exceptions": mock_botocore_exceptions,
    }


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

    def test_available_when_env_and_package(self):
        ctx = _make_load_context()
        strategy = self._make_strategy()
        with patch.dict("os.environ", {"MX_S3_URI": "s3://bucket/model"}):
            with patch("importlib.util.find_spec", return_value=MagicMock()):
                assert strategy.is_available(ctx) is True

    def test_unavailable_no_env(self):
        ctx = _make_load_context()
        strategy = self._make_strategy()
        with patch.dict("os.environ", {}, clear=True):
            assert strategy.is_available(ctx) is False

    def test_unavailable_empty_env(self):
        ctx = _make_load_context()
        strategy = self._make_strategy()
        with patch.dict("os.environ", {"MX_S3_URI": ""}):
            assert strategy.is_available(ctx) is False

    def test_unavailable_no_package(self):
        ctx = _make_load_context()
        strategy = self._make_strategy()
        with patch.dict("os.environ", {"MX_S3_URI": "s3://bucket/model"}):
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
    @patch(
        "modelexpress.load_strategy.model_streamer_strategy."
        "ModelStreamerStrategy._resolve_s3_safetensors"
    )
    def test_success_path(self, mock_resolve, mock_capture, mock_register, mock_publish):
        mock_resolve.return_value = ["s3://bucket/model/shard-00001.safetensors"]
        mock_capture.return_value.__enter__ = MagicMock()
        mock_capture.return_value.__exit__ = MagicMock(return_value=False)

        mock_streamer_instance = MagicMock()
        mock_streamer_instance.get_tensors.return_value = [
            ("layer.0.weight", torch.randn(4, 4)),
            ("layer.1.weight", torch.randn(4, 4)),
        ]

        model = MagicMock()
        ctx = _make_load_context()
        strategy = self._make_strategy()

        with patch.dict("os.environ", {"MX_S3_URI": "s3://bucket/model"}):
            with patch.dict("sys.modules", {"runai_model_streamer": MagicMock()}):
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
    def test_returns_false_on_error(self, mock_register, mock_publish):
        model = MagicMock()
        ctx = _make_load_context()
        strategy = self._make_strategy()

        with patch.dict("os.environ", {"MX_S3_URI": "s3://bucket/model"}):
            with patch(
                "modelexpress.load_strategy.model_streamer_strategy."
                "ModelStreamerStrategy._stream_weights",
                side_effect=RuntimeError("S3 connection failed"),
            ):
                with patch(
                    "vllm.model_executor.model_loader.utils.process_weights_after_loading"
                ):
                    result = strategy.load(model, ctx)

        assert result is False
        mock_register.assert_not_called()
        mock_publish.assert_not_called()


# ---------------------------------------------------------------------------
# TestResolveS3Safetensors
# ---------------------------------------------------------------------------


class TestResolveS3Safetensors:
    def _make_strategy(self):
        from modelexpress.load_strategy.model_streamer_strategy import ModelStreamerStrategy
        return ModelStreamerStrategy()

    def test_resolves_from_index(self):
        strategy = self._make_strategy()

        index_data = {
            "weight_map": {
                "model.layer.0.weight": "model-00001-of-00002.safetensors",
                "model.layer.1.weight": "model-00001-of-00002.safetensors",
                "model.layer.2.weight": "model-00002-of-00002.safetensors",
            }
        }
        body = BytesIO(json.dumps(index_data).encode())

        mock_client = MagicMock()
        mock_client.get_object.return_value = {"Body": body}

        modules = _boto3_modules()
        modules["boto3"].client.return_value = mock_client

        with patch.dict(sys.modules, modules):
            result = strategy._resolve_s3_safetensors("s3://my-bucket/models/llama")

        assert result == [
            "s3://my-bucket/models/llama/model-00001-of-00002.safetensors",
            "s3://my-bucket/models/llama/model-00002-of-00002.safetensors",
        ]
        mock_client.get_object.assert_called_once_with(
            Bucket="my-bucket",
            Key="models/llama/model.safetensors.index.json",
        )

    def test_fallback_to_listing(self):
        strategy = self._make_strategy()

        error_response = {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}

        mock_client = MagicMock()
        mock_client.get_object.side_effect = _MockClientError(error_response, "GetObject")

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "models/llama/model-00001.safetensors"},
                    {"Key": "models/llama/model-00002.safetensors"},
                    {"Key": "models/llama/config.json"},
                ]
            }
        ]
        mock_client.get_paginator.return_value = mock_paginator

        modules = _boto3_modules()
        modules["boto3"].client.return_value = mock_client

        with patch.dict(sys.modules, modules):
            result = strategy._resolve_s3_safetensors("s3://my-bucket/models/llama")

        assert result == [
            "s3://my-bucket/models/llama/model-00001.safetensors",
            "s3://my-bucket/models/llama/model-00002.safetensors",
        ]

    def test_raises_no_files(self):
        strategy = self._make_strategy()

        error_response = {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}

        mock_client = MagicMock()
        mock_client.get_object.side_effect = _MockClientError(error_response, "GetObject")

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Contents": []}]
        mock_client.get_paginator.return_value = mock_paginator

        modules = _boto3_modules()
        modules["boto3"].client.return_value = mock_client

        with patch.dict(sys.modules, modules):
            with pytest.raises(FileNotFoundError, match="No .safetensors files found"):
                strategy._resolve_s3_safetensors("s3://my-bucket/models/llama")

    def test_rejects_non_s3_uri(self):
        strategy = self._make_strategy()
        with pytest.raises(ValueError, match="Expected s3:// URI"):
            strategy._resolve_s3_safetensors("gs://bucket/model")


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
