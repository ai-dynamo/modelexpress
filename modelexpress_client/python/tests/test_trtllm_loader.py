# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for TensorRT-LLM loader module."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from modelexpress.trtllm_loader import (
    MxTrtllmSourcePublisher,
    MxTrtllmTargetLoader,
    _parse_server_address,
    create_trtllm_from_mx,
)


class TestParseServerAddress:
    """Tests for _parse_server_address helper."""
    
    def test_no_prefix(self):
        assert _parse_server_address("localhost:8001") == "localhost:8001"
    
    def test_http_prefix(self):
        assert _parse_server_address("http://localhost:8001") == "localhost:8001"
    
    def test_https_prefix(self):
        assert _parse_server_address("https://localhost:8001") == "localhost:8001"
    
    def test_hostname_with_port(self):
        assert _parse_server_address("modelexpress-server:8001") == "modelexpress-server:8001"


class TestMxTrtllmSourcePublisher:
    """Tests for MxTrtllmSourcePublisher class."""
    
    def test_init(self):
        """Test publisher initialization."""
        publisher = MxTrtllmSourcePublisher(
            checkpoint_dir="/tmp/test",
            model_name="test-model",
            mx_server="localhost:8001"
        )
        
        assert publisher.checkpoint_dir == Path("/tmp/test")
        assert publisher.model_name == "test-model"
        assert publisher.mx_server == "localhost:8001"
        assert publisher._initialized is False
    
    def test_init_strips_http_prefix(self):
        """Test that http:// prefix is stripped from server address."""
        publisher = MxTrtllmSourcePublisher(
            checkpoint_dir="/tmp/test",
            model_name="test-model",
            mx_server="http://localhost:8001"
        )
        
        assert publisher.mx_server == "localhost:8001"
    
    def test_missing_config_raises_error(self):
        """Test that missing config.json raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            publisher = MxTrtllmSourcePublisher(
                checkpoint_dir=tmpdir,
                model_name="test-model",
                mx_server="localhost:8001"
            )
            
            with pytest.raises(FileNotFoundError, match="Config not found"):
                publisher.initialize()
    
    def test_missing_weights_raises_error(self):
        """Test that missing weights file raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config but no weights
            config = {
                "mapping": {"world_size": 1, "tp_size": 1, "pp_size": 1}
            }
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config, f)
            
            publisher = MxTrtllmSourcePublisher(
                checkpoint_dir=tmpdir,
                model_name="test-model",
                mx_server="localhost:8001"
            )
            
            with pytest.raises(FileNotFoundError, match="Weights not found"):
                publisher.initialize()
    
    def test_context_manager(self):
        """Test context manager interface."""
        publisher = MxTrtllmSourcePublisher(
            checkpoint_dir="/tmp/test",
            model_name="test-model",
            mx_server="localhost:8001"
        )
        
        # Mock initialize and shutdown
        publisher.initialize = MagicMock()
        publisher.shutdown = MagicMock()
        
        with publisher as p:
            assert p is publisher
            publisher.initialize.assert_called_once()
        
        publisher.shutdown.assert_called_once()


class TestMxTrtllmTargetLoader:
    """Tests for MxTrtllmTargetLoader class."""
    
    def test_init_defaults(self):
        """Test loader initialization with defaults."""
        loader = MxTrtllmTargetLoader(
            model_name="test-model"
        )
        
        assert loader.model_name == "test-model"
        assert loader.mx_server == "modelexpress-server:8001"
        assert loader.output_dir == Path("/tmp/mx_trtllm")
        assert "gemm_plugin" in loader.build_config
    
    def test_init_custom_config(self):
        """Test loader initialization with custom config."""
        loader = MxTrtllmTargetLoader(
            model_name="test-model",
            mx_server="http://custom-server:9001",
            output_dir="/custom/output",
            build_config={"max_batch_size": "16"}
        )
        
        assert loader.mx_server == "custom-server:9001"
        assert loader.output_dir == Path("/custom/output")
        assert loader.build_config["max_batch_size"] == "16"
    
    def test_parse_dtype(self):
        """Test dtype string parsing."""
        loader = MxTrtllmTargetLoader(model_name="test")
        
        assert loader._parse_dtype("torch.float16") == torch.float16
        assert loader._parse_dtype("torch.float32") == torch.float32
        assert loader._parse_dtype("torch.bfloat16") == torch.bfloat16
        assert loader._parse_dtype("torch.int8") == torch.int8
        assert loader._parse_dtype("unknown") == torch.float16  # default
    
    def test_dtype_size(self):
        """Test dtype size calculation."""
        loader = MxTrtllmTargetLoader(model_name="test")
        
        assert loader._dtype_size(torch.float16) == 2
        assert loader._dtype_size(torch.float32) == 4
        assert loader._dtype_size(torch.int8) == 1
    
    def test_transfer_stats_initially_empty(self):
        """Test that transfer stats are initially empty."""
        loader = MxTrtllmTargetLoader(model_name="test")
        assert loader.get_transfer_stats() == {}


class TestCreateTrtllmFromMx:
    """Tests for create_trtllm_from_mx convenience function."""
    
    @patch('modelexpress.trtllm_loader.MxTrtllmTargetLoader')
    def test_creates_loader_and_calls_load(self, mock_loader_class):
        """Test that function creates loader and calls load."""
        mock_loader = MagicMock()
        mock_loader.load.return_value = "/path/to/engine"
        mock_loader_class.return_value = mock_loader
        
        result = create_trtllm_from_mx(
            model_name="test-model",
            mx_server="localhost:8001",
            output_dir="/tmp/test",
            skip_build=True
        )
        
        mock_loader_class.assert_called_once_with(
            model_name="test-model",
            mx_server="localhost:8001",
            output_dir="/tmp/test",
            build_config=None,
        )
        mock_loader.load.assert_called_once_with(skip_build=True)
        assert result == "/path/to/engine"


# Integration tests (require NIXL and GPU)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)
class TestTrtllmLoaderIntegration:
    """Integration tests requiring GPU."""
    
    def test_source_publisher_with_mock_checkpoint(self):
        """Test source publisher with a mock checkpoint."""
        pytest.skip("Requires NIXL and gRPC server")
    
    def test_target_loader_with_mock_source(self):
        """Test target loader with a mock source."""
        pytest.skip("Requires NIXL and gRPC server")
