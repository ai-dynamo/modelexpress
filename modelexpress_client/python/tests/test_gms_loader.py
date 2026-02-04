# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MxGmsSourceLoader."""

from unittest.mock import MagicMock, patch

import pytest
import torch


class TestMxGmsSourceLoaderInit:
    """Tests for MxGmsSourceLoader initialization."""

    def test_init_modifies_load_format(self, mocker):
        """Test that __init__ modifies load_format to 'auto'."""
        # Mock DefaultModelLoader to avoid vLLM import issues
        mock_default_loader = mocker.patch(
            "modelexpress.gms_loader.DefaultModelLoader.__init__",
            return_value=None,
        )

        from modelexpress.gms_loader import MxGmsSourceLoader

        # Create a mock LoadConfig
        mock_load_config = MagicMock()
        mock_load_config.load_format = "mx-gms-source"

        loader = MxGmsSourceLoader(mock_load_config)

        # Verify DefaultModelLoader was called with modified config
        call_args = mock_default_loader.call_args
        modified_config = call_args[0][1]  # Second positional arg (after self)
        assert modified_config.load_format == "auto"

        # Verify internal state
        assert loader._gms_client is None
        assert loader._pool is None

    def test_gms_client_property_returns_none_initially(self, mocker):
        """Test gms_client property returns None before load_model."""
        mocker.patch(
            "modelexpress.gms_loader.DefaultModelLoader.__init__",
            return_value=None,
        )

        from modelexpress.gms_loader import MxGmsSourceLoader

        mock_load_config = MagicMock()
        loader = MxGmsSourceLoader(mock_load_config)

        assert loader.gms_client is None


class TestMxGmsSourceLoaderLoadModel:
    """Tests for MxGmsSourceLoader.load_model()."""

    @pytest.fixture
    def loader_with_mocks(self, mocker):
        """Create a loader with all dependencies mocked."""
        # Mock vLLM imports
        mocker.patch(
            "modelexpress.gms_loader.DefaultModelLoader.__init__",
            return_value=None,
        )
        mocker.patch("modelexpress.gms_loader.DefaultModelLoader.load_weights")

        # Mock GMS functions
        mock_gms_client = MagicMock()
        mock_gms_client.total_bytes = 2 * (1 << 30)  # 2 GiB
        mock_gms_client.mappings = {0x1000: MagicMock()}
        mock_gms_client.commit.return_value = True

        mock_pool = MagicMock()

        mocker.patch(
            "modelexpress.gms_loader.get_or_create_gms_client_memory_manager",
            return_value=(mock_gms_client, mock_pool),
        )
        mocker.patch(
            "modelexpress.gms_loader.get_socket_path",
            return_value="/tmp/gms_test.sock",
        )
        mocker.patch("modelexpress.gms_loader.register_module_tensors")

        # Mock vLLM model loading utilities
        mock_model = MagicMock(spec=torch.nn.Module)
        mock_model.eval.return_value = mock_model
        mocker.patch(
            "modelexpress.gms_loader.initialize_model",
            return_value=mock_model,
        )
        mocker.patch("modelexpress.gms_loader.process_weights_after_loading")
        mocker.patch("modelexpress.gms_loader.set_default_torch_dtype")
        mocker.patch("modelexpress.gms_loader.use_mem_pool")

        # Mock torch CUDA functions
        mocker.patch("torch.cuda.synchronize")
        mocker.patch("torch.cuda.empty_cache")
        mocker.patch("torch.device")

        from modelexpress.gms_loader import MxGmsSourceLoader

        mock_load_config = MagicMock()
        loader = MxGmsSourceLoader(mock_load_config)

        return loader, mock_gms_client, mock_pool, mock_model

    def test_load_model_connects_to_gms_as_rw(self, loader_with_mocks, mocker):
        """Test that load_model connects to GMS in RW mode."""
        loader, mock_gms_client, _, _ = loader_with_mocks

        mock_get_gms = mocker.patch(
            "modelexpress.gms_loader.get_or_create_gms_client_memory_manager",
            return_value=(mock_gms_client, MagicMock()),
        )
        mocker.patch(
            "modelexpress.gms_loader.RequestedLockType"
        ).RW = "RW"

        mock_vllm_config = MagicMock()
        mock_vllm_config.device_config.device = "cuda:0"
        mock_vllm_config.load_config.device = None
        mock_model_config = MagicMock()

        loader.load_model(mock_vllm_config, mock_model_config)

        # Verify GMS connection was requested
        mock_get_gms.assert_called_once()

    def test_load_model_clears_existing_allocations(self, loader_with_mocks, mocker):
        """Test that load_model clears existing GMS allocations."""
        loader, mock_gms_client, _, _ = loader_with_mocks

        mocker.patch(
            "modelexpress.gms_loader.get_or_create_gms_client_memory_manager",
            return_value=(mock_gms_client, MagicMock()),
        )
        mocker.patch("modelexpress.gms_loader.RequestedLockType")

        mock_vllm_config = MagicMock()
        mock_vllm_config.device_config.device = "cuda:0"
        mock_vllm_config.load_config.device = None
        mock_model_config = MagicMock()

        loader.load_model(mock_vllm_config, mock_model_config)

        mock_gms_client.clear_all.assert_called_once()

    def test_load_model_registers_tensors(self, loader_with_mocks, mocker):
        """Test that load_model registers tensors with GMS."""
        loader, mock_gms_client, _, mock_model = loader_with_mocks

        mocker.patch(
            "modelexpress.gms_loader.get_or_create_gms_client_memory_manager",
            return_value=(mock_gms_client, MagicMock()),
        )
        mocker.patch("modelexpress.gms_loader.RequestedLockType")
        mock_register = mocker.patch("modelexpress.gms_loader.register_module_tensors")

        mock_vllm_config = MagicMock()
        mock_vllm_config.device_config.device = "cuda:0"
        mock_vllm_config.load_config.device = None
        mock_model_config = MagicMock()

        loader.load_model(mock_vllm_config, mock_model_config)

        mock_register.assert_called_once_with(mock_gms_client, mock_model)

    def test_load_model_commits_and_switches_to_read(self, loader_with_mocks, mocker):
        """Test that load_model commits and switches to read mode."""
        loader, mock_gms_client, _, _ = loader_with_mocks

        mocker.patch(
            "modelexpress.gms_loader.get_or_create_gms_client_memory_manager",
            return_value=(mock_gms_client, MagicMock()),
        )
        mocker.patch("modelexpress.gms_loader.RequestedLockType")

        mock_vllm_config = MagicMock()
        mock_vllm_config.device_config.device = "cuda:0"
        mock_vllm_config.load_config.device = None
        mock_model_config = MagicMock()

        loader.load_model(mock_vllm_config, mock_model_config)

        mock_gms_client.commit.assert_called_once()
        mock_gms_client.switch_to_read.assert_called_once()

    def test_load_model_raises_on_commit_failure(self, loader_with_mocks, mocker):
        """Test that load_model raises RuntimeError if commit fails."""
        loader, mock_gms_client, _, _ = loader_with_mocks
        mock_gms_client.commit.return_value = False

        mocker.patch(
            "modelexpress.gms_loader.get_or_create_gms_client_memory_manager",
            return_value=(mock_gms_client, MagicMock()),
        )
        mocker.patch("modelexpress.gms_loader.RequestedLockType")

        mock_vllm_config = MagicMock()
        mock_vllm_config.device_config.device = "cuda:0"
        mock_vllm_config.load_config.device = None
        mock_model_config = MagicMock()

        with pytest.raises(RuntimeError, match="GMS commit failed"):
            loader.load_model(mock_vllm_config, mock_model_config)

    def test_load_model_returns_model_in_eval_mode(self, loader_with_mocks, mocker):
        """Test that load_model returns model in eval mode."""
        loader, mock_gms_client, _, mock_model = loader_with_mocks

        mocker.patch(
            "modelexpress.gms_loader.get_or_create_gms_client_memory_manager",
            return_value=(mock_gms_client, MagicMock()),
        )
        mocker.patch("modelexpress.gms_loader.RequestedLockType")

        mock_vllm_config = MagicMock()
        mock_vllm_config.device_config.device = "cuda:0"
        mock_vllm_config.load_config.device = None
        mock_model_config = MagicMock()

        result = loader.load_model(mock_vllm_config, mock_model_config)

        mock_model.eval.assert_called_once()
        assert result == mock_model


class TestMxGmsSourceLoaderClose:
    """Tests for MxGmsSourceLoader.close()."""

    def test_close_closes_gms_client(self, mocker):
        """Test that close() closes the GMS client."""
        mocker.patch(
            "modelexpress.gms_loader.DefaultModelLoader.__init__",
            return_value=None,
        )

        from modelexpress.gms_loader import MxGmsSourceLoader

        mock_load_config = MagicMock()
        loader = MxGmsSourceLoader(mock_load_config)

        # Simulate having a GMS client
        mock_client = MagicMock()
        loader._gms_client = mock_client
        loader._pool = MagicMock()

        loader.close()

        mock_client.close.assert_called_once()
        assert loader._gms_client is None
        assert loader._pool is None

    def test_close_handles_no_client(self, mocker):
        """Test that close() handles case with no GMS client."""
        mocker.patch(
            "modelexpress.gms_loader.DefaultModelLoader.__init__",
            return_value=None,
        )

        from modelexpress.gms_loader import MxGmsSourceLoader

        mock_load_config = MagicMock()
        loader = MxGmsSourceLoader(mock_load_config)

        # Should not raise
        loader.close()

        assert loader._gms_client is None
