# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for GMS sidecar."""

from unittest.mock import MagicMock

import pytest


class TestMxGmsSidecarConfig:
    """Tests for MxGmsSidecarConfig."""

    def test_config_validates_required_model(self):
        """Test that model is required."""
        from modelexpress.gms_sidecar import MxGmsSidecarConfig

        with pytest.raises(Exception):  # Pydantic ValidationError
            MxGmsSidecarConfig()

    def test_config_accepts_valid_model(self, mocker):
        """Test that valid model is accepted."""
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.device_count", return_value=1)

        from modelexpress.gms_sidecar import MxGmsSidecarConfig

        config = MxGmsSidecarConfig(model="test-model")
        assert config.model == "test-model"
        assert config.device == 0
        assert config.dtype.value == "auto"

    def test_config_validates_device_availability(self, mocker):
        """Test that device is validated against available devices."""
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.device_count", return_value=2)

        from modelexpress.gms_sidecar import MxGmsSidecarConfig

        # Valid device
        config = MxGmsSidecarConfig(model="test-model", device=1)
        assert config.device == 1

        # Invalid device (index too high)
        with pytest.raises(ValueError, match="Device 5 not available"):
            MxGmsSidecarConfig(model="test-model", device=5)

    def test_config_validates_cuda_available(self, mocker):
        """Test that CUDA availability is checked."""
        mocker.patch("torch.cuda.is_available", return_value=False)

        from modelexpress.gms_sidecar import MxGmsSidecarConfig

        with pytest.raises(ValueError, match="CUDA is not available"):
            MxGmsSidecarConfig(model="test-model")

    def test_config_all_fields(self, mocker):
        """Test configuration with all fields."""
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.device_count", return_value=4)

        from modelexpress.gms_sidecar import (
            DType,
            MxGmsSidecarConfig,
        )

        config = MxGmsSidecarConfig(
            model="meta-llama/Llama-3.2-1B",
            device=2,
            socket_path="/tmp/custom.sock",
            dtype=DType.BFLOAT16,
            trust_remote_code=True,
            stay_running=True,
            revision="main",
            max_model_len=4096,
        )

        assert config.model == "meta-llama/Llama-3.2-1B"
        assert config.device == 2
        assert config.socket_path == "/tmp/custom.sock"
        assert config.dtype == DType.BFLOAT16
        assert config.trust_remote_code is True
        assert config.stay_running is True
        assert config.revision == "main"
        assert config.max_model_len == 4096


class TestDType:
    """Tests for DType enum."""

    def test_dtype_values(self):
        """Test DType enum values."""
        from modelexpress.gms_sidecar import DType

        assert DType.AUTO.value == "auto"
        assert DType.FLOAT16.value == "float16"
        assert DType.BFLOAT16.value == "bfloat16"
        assert DType.FLOAT32.value == "float32"

    def test_dtype_from_string(self):
        """Test DType creation from string."""
        from modelexpress.gms_sidecar import DType

        assert DType("auto") == DType.AUTO
        assert DType("float16") == DType.FLOAT16
        assert DType("bfloat16") == DType.BFLOAT16
        assert DType("float32") == DType.FLOAT32


class TestParseArgs:
    """Tests for parse_args function."""

    def test_parse_args_minimal(self, mocker):
        """Test parse_args with minimal arguments."""
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.device_count", return_value=1)
        mocker.patch(
            "sys.argv",
            ["gms_sidecar", "--model", "test-model"],
        )

        from modelexpress.gms_sidecar import parse_args

        config = parse_args()
        assert config.model == "test-model"
        assert config.device == 0

    def test_parse_args_all_options(self, mocker):
        """Test parse_args with all options."""
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.device_count", return_value=4)
        mocker.patch(
            "sys.argv",
            [
                "gms_sidecar",
                "--model",
                "test-model",
                "--device",
                "2",
                "--socket-path",
                "/tmp/test.sock",
                "--dtype",
                "bfloat16",
                "--trust-remote-code",
                "--stay-running",
                "--revision",
                "v1.0",
                "--max-model-len",
                "2048",
            ],
        )

        from modelexpress.gms_sidecar import DType, parse_args

        config = parse_args()
        assert config.model == "test-model"
        assert config.device == 2
        assert config.socket_path == "/tmp/test.sock"
        assert config.dtype == DType.BFLOAT16
        assert config.trust_remote_code is True
        assert config.stay_running is True
        assert config.revision == "v1.0"
        assert config.max_model_len == 2048


class TestBuildVllmConfigs:
    """Tests for build_vllm_configs function."""

    def test_build_vllm_configs(self, mocker):
        """Test build_vllm_configs creates proper configs."""
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.device_count", return_value=1)

        # Mock vLLM imports
        mock_engine_args = MagicMock()
        mock_vllm_config = MagicMock()
        mock_model_config = MagicMock()
        mock_load_config = MagicMock()

        mock_vllm_config.model_config = mock_model_config
        mock_vllm_config.load_config = mock_load_config
        mock_engine_args.create_engine_config.return_value = mock_vllm_config

        mocker.patch(
            "modelexpress.gms_sidecar.AsyncEngineArgs",
            return_value=mock_engine_args,
        )

        from modelexpress.gms_sidecar import (
            MxGmsSidecarConfig,
            build_vllm_configs,
        )

        config = MxGmsSidecarConfig(model="test-model")
        vllm_config, model_config, load_config = build_vllm_configs(config)

        assert vllm_config == mock_vllm_config
        assert model_config == mock_model_config
        assert load_config == mock_load_config


class TestMain:
    """Tests for main function."""

    def test_main_returns_zero_on_success(self, mocker):
        """Test main returns 0 on successful execution."""
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.device_count", return_value=1)
        mocker.patch("torch.cuda.set_device")
        mocker.patch(
            "sys.argv",
            ["gms_sidecar", "--model", "test-model"],
        )

        # Mock build_vllm_configs
        mock_vllm_config = MagicMock()
        mock_model_config = MagicMock()
        mock_load_config = MagicMock()
        mocker.patch(
            "modelexpress.gms_sidecar.build_vllm_configs",
            return_value=(mock_vllm_config, mock_model_config, mock_load_config),
        )

        # Mock loader
        mock_model = MagicMock()
        mock_model.parameters.return_value = [
            MagicMock(numel=MagicMock(return_value=1e9))
        ]
        mock_loader = MagicMock()
        mock_loader.load_model.return_value = mock_model
        mocker.patch(
            "modelexpress.gms_sidecar.MxGmsSourceLoader",
            return_value=mock_loader,
        )

        from modelexpress.gms_sidecar import main

        result = main()
        assert result == 0

    def test_main_returns_one_on_config_error(self, mocker):
        """Test main returns 1 on configuration error."""
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch(
            "sys.argv",
            ["gms_sidecar", "--model", "test-model"],
        )

        from modelexpress.gms_sidecar import main

        result = main()
        assert result == 1

    def test_main_returns_one_on_load_error(self, mocker):
        """Test main returns 1 on model load error."""
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.device_count", return_value=1)
        mocker.patch("torch.cuda.set_device")
        mocker.patch(
            "sys.argv",
            ["gms_sidecar", "--model", "test-model"],
        )

        # Mock build_vllm_configs
        mock_vllm_config = MagicMock()
        mock_model_config = MagicMock()
        mock_load_config = MagicMock()
        mocker.patch(
            "modelexpress.gms_sidecar.build_vllm_configs",
            return_value=(mock_vllm_config, mock_model_config, mock_load_config),
        )

        # Mock loader to raise exception
        mock_loader = MagicMock()
        mock_loader.load_model.side_effect = RuntimeError("Load failed")
        mocker.patch(
            "modelexpress.gms_sidecar.MxGmsSourceLoader",
            return_value=mock_loader,
        )

        from modelexpress.gms_sidecar import main

        result = main()
        assert result == 1
        mock_loader.close.assert_called_once()
