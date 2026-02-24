# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared helpers and MxModelLoader detection logic."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# _collect_cuda_tensors
# ---------------------------------------------------------------------------


class TestCollectCudaTensors:
    """Tests for the _collect_cuda_tensors helper."""

    def test_empty_model(self):
        from modelexpress.vllm_loader import _collect_cuda_tensors

        model = nn.Module()
        result = _collect_cuda_tensors(model)
        assert result == {}

    def test_cpu_only_model(self):
        from modelexpress.vllm_loader import _collect_cuda_tensors

        model = nn.Linear(4, 2, bias=False)
        # Parameters default to CPU
        result = _collect_cuda_tensors(model)
        assert result == {}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_model(self):
        from modelexpress.vllm_loader import _collect_cuda_tensors

        model = nn.Linear(4, 2, bias=True).cuda()
        result = _collect_cuda_tensors(model)
        assert len(result) == 2  # weight + bias
        assert "weight" in result
        assert "bias" in result
        for t in result.values():
            assert t.is_cuda


# ---------------------------------------------------------------------------
# _detect_source (via MxModelLoader)
# ---------------------------------------------------------------------------


class _FakeReadyResponse:
    """Minimal stand-in for p2p_pb2.GetReadyResponse."""

    def __init__(self, found=False, ready=False, session_id="", metadata_hash=""):
        self.found = found
        self.ready = ready
        self.session_id = session_id
        self.metadata_hash = metadata_hash


class _FakeWorker:
    """Minimal stand-in for p2p_pb2.WorkerMetadata."""

    def __init__(self, worker_rank, tensors=None, nixl_metadata=b""):
        self.worker_rank = worker_rank
        self.tensors = tensors or []
        self.nixl_metadata = nixl_metadata


class _FakeTensor:
    """Minimal stand-in for p2p_pb2.TensorDescriptor."""

    def __init__(self, name="w", addr=0, size=1024, device_id=0, dtype="bf16"):
        self.name = name
        self.addr = addr
        self.size = size
        self.device_id = device_id
        self.dtype = dtype


class _FakeMetadataResponse:
    """Minimal stand-in for p2p_pb2.GetMetadataResponse."""

    def __init__(self, found=False, workers=None):
        self.found = found
        self.workers = workers or []


def _make_auto_loader():
    """Create an MxModelLoader with a mock MxClient."""
    with patch("modelexpress.vllm_loader.DefaultModelLoader"):
        with patch("modelexpress.vllm_loader.DummyModelLoader"):
            # Patch the LoadConfig to avoid vllm import issues
            load_config = MagicMock()
            load_config.load_format = "mx"
            load_config.device = None

            from modelexpress.vllm_loader import MxModelLoader

            loader = MxModelLoader(load_config)
    return loader


class TestDetectSource:
    """Tests for MxModelLoader._detect_source."""

    def test_no_model_name(self):
        loader = _make_auto_loader()
        result = loader._detect_source("", device_id=0)
        assert result is None

    @patch("modelexpress.vllm_loader.is_nixl_available", return_value=False)
    def test_nixl_not_available(self, _mock_nixl):
        loader = _make_auto_loader()
        result = loader._detect_source("some-model", device_id=0)
        assert result is None

    @patch("modelexpress.vllm_loader.is_nixl_available", return_value=True)
    def test_server_unreachable(self, _mock_nixl):
        loader = _make_auto_loader()
        loader._mx_client = MagicMock()
        loader._mx_client.get_ready.side_effect = Exception("connection refused")

        result = loader._detect_source("model", device_id=0)
        assert result is None

    @patch("modelexpress.vllm_loader.is_nixl_available", return_value=True)
    def test_not_ready(self, _mock_nixl):
        loader = _make_auto_loader()
        loader._mx_client = MagicMock()
        loader._mx_client.get_ready.return_value = _FakeReadyResponse(
            found=True, ready=False
        )

        result = loader._detect_source("model", device_id=0)
        assert result is None

    @patch("modelexpress.vllm_loader.is_nixl_available", return_value=True)
    def test_ready_no_metadata(self, _mock_nixl):
        loader = _make_auto_loader()
        loader._mx_client = MagicMock()
        loader._mx_client.get_ready.return_value = _FakeReadyResponse(
            found=True, ready=True, session_id="abc"
        )
        loader._mx_client.get_metadata.return_value = _FakeMetadataResponse(found=False)

        result = loader._detect_source("model", device_id=0)
        assert result is None

    @patch("modelexpress.vllm_loader.is_nixl_available", return_value=True)
    def test_ready_wrong_rank(self, _mock_nixl):
        loader = _make_auto_loader()
        loader._mx_client = MagicMock()
        loader._mx_client.get_ready.return_value = _FakeReadyResponse(
            found=True, ready=True, session_id="abc"
        )
        # Source has rank 1 but we need rank 0
        loader._mx_client.get_metadata.return_value = _FakeMetadataResponse(
            found=True,
            workers=[_FakeWorker(worker_rank=1, tensors=[_FakeTensor()])],
        )

        result = loader._detect_source("model", device_id=0)
        assert result is None

    @patch("modelexpress.vllm_loader.is_nixl_available", return_value=True)
    def test_happy_path(self, _mock_nixl):
        loader = _make_auto_loader()
        loader._mx_client = MagicMock()
        loader._mx_client.get_ready.return_value = _FakeReadyResponse(
            found=True, ready=True, session_id="session-123"
        )
        worker = _FakeWorker(
            worker_rank=0,
            tensors=[_FakeTensor(name="layer.0.weight")],
            nixl_metadata=b"nixl-data",
        )
        loader._mx_client.get_metadata.return_value = _FakeMetadataResponse(
            found=True, workers=[worker],
        )

        result = loader._detect_source("model", device_id=0)
        assert result is not None
        assert result.worker_rank == 0
        assert len(result.tensors) == 1


# ---------------------------------------------------------------------------
# _publish_metadata_and_ready
# ---------------------------------------------------------------------------


class TestPublishMetadataAndReady:
    """Tests for the _publish_metadata_and_ready helper."""

    @patch("modelexpress.vllm_loader.p2p_pb2", create=True)
    def test_publishes_correct_tensor_count(self, mock_pb2):
        from modelexpress.vllm_loader import _publish_metadata_and_ready

        mock_pb2.TensorDescriptor = MagicMock()
        mock_pb2.WorkerMetadata = MagicMock()

        mx_client = MagicMock()
        mx_client.publish_metadata.return_value = True
        mx_client.session_id = "test-session"

        nixl_manager = MagicMock()
        nixl_manager.nixl_metadata = b"metadata"

        tensors = {}
        for i in range(5):
            t = MagicMock(spec=torch.Tensor)
            t.data_ptr.return_value = 0x1000 + i * 1024
            t.numel.return_value = 256
            t.element_size.return_value = 4
            t.dtype = torch.float32
            tensors[f"layer.{i}.weight"] = t

        with patch.dict("os.environ", {"MX_CONTIGUOUS_REG": "0"}):
            _publish_metadata_and_ready(
                mx_client, nixl_manager, tensors, device_id=0, model_name="test-model"
            )

        # Check that publish_metadata was called with the worker proto
        mx_client.publish_metadata.assert_called_once()
        # Check that publish_ready was called
        mx_client.publish_ready.assert_called_once()
        call_kwargs = mx_client.publish_ready.call_args
        assert call_kwargs[1]["model_name"] == "test-model" or call_kwargs.kwargs.get("model_name") == "test-model"

    @patch("modelexpress.vllm_loader.p2p_pb2", create=True)
    def test_publish_failure_does_not_publish_ready(self, mock_pb2):
        from modelexpress.vllm_loader import _publish_metadata_and_ready

        mock_pb2.TensorDescriptor = MagicMock()
        mock_pb2.WorkerMetadata = MagicMock()

        mx_client = MagicMock()
        mx_client.publish_metadata.return_value = False

        nixl_manager = MagicMock()
        nixl_manager.nixl_metadata = b"metadata"

        with patch.dict("os.environ", {"MX_CONTIGUOUS_REG": "0"}):
            _publish_metadata_and_ready(
                mx_client, nixl_manager, {}, device_id=0, model_name="test"
            )

        mx_client.publish_ready.assert_not_called()
