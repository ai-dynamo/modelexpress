# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GMS CLI dispatcher."""

from unittest.mock import MagicMock, patch

import pytest

from modelexpress.gms.config import EngineType, GmsConfig, GmsMode


class TestParseArgs:
    """Tests for parse_args function."""

    def test_parse_args_minimal(self, mocker):
        mocker.patch("sys.argv", ["gms", "--model", "test-model"])

        from modelexpress.gms.main import parse_args

        config = parse_args()
        assert config.model == "test-model"
        assert config.engine == EngineType.VLLM
        assert config.mode == GmsMode.SOURCE
        assert config.tp_size == 1
        assert config.ep_size == 1

    def test_parse_args_full(self, mocker):
        mocker.patch(
            "sys.argv",
            [
                "gms",
                "--model",
                "deepseek-v3",
                "--engine",
                "vllm",
                "--mode",
                "target",
                "--tp-size",
                "4",
                "--ep-size",
                "2",
                "--mx-server",
                "mx:8001",
                "--model-name",
                "ds-v3",
                "--dtype",
                "bfloat16",
                "--trust-remote-code",
                "--revision",
                "main",
                "--max-model-len",
                "4096",
                "--weight-source",
                "gds",
                "--s3-bucket",
                "bucket",
                "--s3-prefix",
                "prefix/",
                "--cache-endpoint",
                "http://cache",
            ],
        )

        from modelexpress.gms.main import parse_args

        config = parse_args()
        assert config.model == "deepseek-v3"
        assert config.mode == GmsMode.TARGET
        assert config.tp_size == 4
        assert config.ep_size == 2
        assert config.mx_server == "mx:8001"
        assert config.model_name == "ds-v3"
        assert config.trust_remote_code is True
        assert config.max_model_len == 4096


class TestGetLauncher:
    """Tests for get_launcher function."""

    def test_returns_vllm_module(self):
        pytest.importorskip("torch")
        from modelexpress.gms.main import get_launcher

        launcher = get_launcher(EngineType.VLLM)
        assert hasattr(launcher, "run")

    def test_raises_for_unsupported(self):
        from modelexpress.gms.main import get_launcher

        with pytest.raises(ValueError, match="Unsupported engine"):
            get_launcher(EngineType.SGLANG)


class TestMain:
    """Tests for main function."""

    @patch("modelexpress.gms.main.get_launcher")
    @patch("modelexpress.gms.main.parse_args")
    def test_returns_zero_on_success(self, mock_parse, mock_get_launcher):
        from modelexpress.gms.main import main

        mock_config = GmsConfig(model="test-model")
        mock_parse.return_value = mock_config

        mock_launcher = MagicMock()
        mock_get_launcher.return_value = mock_launcher

        result = main()
        assert result == 0
        mock_launcher.run.assert_called_once_with(mock_config)

    @patch("modelexpress.gms.main.get_launcher")
    @patch("modelexpress.gms.main.parse_args")
    def test_returns_one_on_launcher_error(self, mock_parse, mock_get_launcher):
        from modelexpress.gms.main import main

        mock_config = GmsConfig(model="test-model", engine=EngineType.SGLANG)
        mock_parse.return_value = mock_config

        mock_get_launcher.side_effect = ValueError("Unsupported engine")

        result = main()
        assert result == 1
