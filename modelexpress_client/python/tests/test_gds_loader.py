# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GDS loader and transfer manager."""

import json
import struct
from unittest.mock import MagicMock, mock_open, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# GDS availability detection
# ---------------------------------------------------------------------------


class TestIsGdsAvailable:
    """Tests for system-level GDS detection."""

    @patch("modelexpress.gds_transfer.NIXL_AVAILABLE", False)
    def test_nixl_not_installed(self):
        from modelexpress.gds_transfer import is_gds_available
        assert is_gds_available() is False

    @patch("modelexpress.gds_transfer.NIXL_AVAILABLE", True)
    @patch("modelexpress.gds_transfer._nvidia_fs_loaded", return_value=False)
    def test_no_nvidia_fs_module(self, _mock_fs):
        from modelexpress.gds_transfer import is_gds_available
        assert is_gds_available() is False

    @patch("modelexpress.gds_transfer.NIXL_AVAILABLE", True)
    @patch("modelexpress.gds_transfer._nvidia_fs_loaded", return_value=True)
    @patch("modelexpress.gds_transfer._cufile_loadable", return_value=False)
    def test_no_libcufile(self, _mock_cufile, _mock_fs):
        from modelexpress.gds_transfer import is_gds_available
        assert is_gds_available() is False

    @patch("modelexpress.gds_transfer.NIXL_AVAILABLE", True)
    @patch("modelexpress.gds_transfer._nvidia_fs_loaded", return_value=True)
    @patch("modelexpress.gds_transfer._cufile_loadable", return_value=True)
    def test_all_present(self, _mock_cufile, _mock_fs):
        from modelexpress.gds_transfer import is_gds_available
        assert is_gds_available() is True


class TestNvidiaFsLoaded:
    """Tests for nvidia_fs kernel module detection."""

    def test_module_present(self):
        content = (
            "nvidia_fs 12345 0 - Live 0xffffffff\n"
            "nvidia 67890 1 - Live 0xfffffffe\n"
        )
        from modelexpress.gds_transfer import _nvidia_fs_loaded
        with patch("builtins.open", mock_open(read_data=content)):
            assert _nvidia_fs_loaded() is True

    def test_module_absent(self):
        content = "nvidia 67890 1 - Live 0xfffffffe\n"
        from modelexpress.gds_transfer import _nvidia_fs_loaded
        with patch("builtins.open", mock_open(read_data=content)):
            assert _nvidia_fs_loaded() is False

    def test_proc_not_readable(self):
        from modelexpress.gds_transfer import _nvidia_fs_loaded
        with patch("builtins.open", side_effect=OSError("not readable")):
            assert _nvidia_fs_loaded() is False


class TestCufileLoadable:
    """Tests for libcufile.so detection."""

    @patch("ctypes.CDLL")
    def test_loadable(self, mock_cdll):
        from modelexpress.gds_transfer import _cufile_loadable
        assert _cufile_loadable() is True
        mock_cdll.assert_called_once_with("libcufile.so")

    @patch("ctypes.CDLL", side_effect=OSError("not found"))
    def test_not_loadable(self, _mock_cdll):
        from modelexpress.gds_transfer import _cufile_loadable
        assert _cufile_loadable() is False


# ---------------------------------------------------------------------------
# GdsTransferManager
# ---------------------------------------------------------------------------


class TestGdsTransferManager:
    """Tests for the GdsTransferManager class."""

    def test_not_available_raises(self):
        with patch("modelexpress.gds_transfer.NIXL_AVAILABLE", False):
            from modelexpress.gds_transfer import GdsTransferManager
            mgr = GdsTransferManager(agent_name="test")
            with pytest.raises(RuntimeError, match="not available"):
                mgr.initialize()

    def test_batch_load_requires_init(self):
        with patch("modelexpress.gds_transfer.NIXL_AVAILABLE", True):
            from modelexpress.gds_transfer import GdsTransferManager
            mgr = GdsTransferManager(agent_name="test")
            with pytest.raises(RuntimeError, match="not initialized"):
                mgr.batch_load_file(0, 100, [], torch.device("cpu"))


# ---------------------------------------------------------------------------
# Safetensors header parsing
# ---------------------------------------------------------------------------


class TestParseSafetensorsHeader:
    """Tests for safetensors header parsing."""

    def test_parses_header(self, tmp_path):
        from modelexpress.gds_loader import MxGdsLoader

        header = {
            "weight": {
                "dtype": "F32",
                "shape": [4, 2],
                "data_offsets": [0, 32],
            }
        }
        header_bytes = json.dumps(header).encode()
        file_path = tmp_path / "test.safetensors"
        with open(file_path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(b"\x00" * 32)

        loader = MxGdsLoader()
        parsed = loader._parse_safetensors_header(str(file_path))
        assert "weight" in parsed
        assert parsed["weight"]["dtype"] == "F32"
        assert parsed["weight"]["shape"] == [4, 2]
        assert parsed["weight"]["size"] == 32
        assert parsed["weight"]["file_offset"] == 8 + len(header_bytes)


# ---------------------------------------------------------------------------
# Model path resolution
# ---------------------------------------------------------------------------


class TestResolveModelPath:
    """Tests for model path resolution."""

    def test_local_directory_returned_as_is(self, tmp_path):
        from modelexpress.gds_loader import MxGdsLoader
        result = MxGdsLoader._resolve_model_path(str(tmp_path))
        assert result == str(tmp_path.resolve())

    @patch("huggingface_hub.snapshot_download")
    def test_hf_model_calls_snapshot_download(self, mock_download):
        from modelexpress.gds_loader import MxGdsLoader
        mock_download.return_value = "/cache/models/org/model"
        result = MxGdsLoader._resolve_model_path("org/model")
        mock_download.assert_called_once_with("org/model", revision=None)
        assert result == "/cache/models/org/model"


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


class TestResolveSafetensorsFiles:
    """Tests for safetensors file discovery."""

    def test_sharded_index(self, tmp_path):
        from modelexpress.gds_loader import MxGdsLoader

        # Create index file
        index = {
            "weight_map": {
                "layer.0.weight": "model-00001.safetensors",
                "layer.1.weight": "model-00002.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        loader = MxGdsLoader()
        result = loader._resolve_safetensors_files(str(tmp_path))
        assert len(result) == 2

    def test_no_files_raises(self, tmp_path):
        from modelexpress.gds_loader import MxGdsLoader
        loader = MxGdsLoader()
        with pytest.raises(FileNotFoundError, match=r"No \.safetensors"):
            loader._resolve_safetensors_files(str(tmp_path))


# ---------------------------------------------------------------------------
# GdsStrategy integration
# ---------------------------------------------------------------------------


class TestGdsStrategyIntegration:
    """Tests for GdsStrategy loading behavior."""

    def _make_context(self):
        from modelexpress.load_strategy import LoadContext
        return LoadContext(
            vllm_config=MagicMock(),
            model_config=MagicMock(),
            load_config=MagicMock(),
            target_device=torch.device("cpu"),
            global_rank=0,
            device_id=0,
            identity=MagicMock(),
            mx_client=MagicMock(),
            worker_id="test-worker",
        )

    @patch("modelexpress.gds_transfer.is_gds_available", return_value=True)
    @patch("modelexpress.gds_loader.MxGdsLoader")
    @patch("modelexpress.load_strategy.base.start_metadata_publisher")
    @patch("modelexpress.load_strategy.base.is_nixl_available", return_value=False)
    def test_gds_success(self, _mock_nixl, _mock_pub, mock_gds_cls, _mock_avail):
        from modelexpress.load_strategy.gds_strategy import GdsStrategy

        mock_gds = MagicMock()
        mock_gds.load_iter.return_value = iter([("w", torch.zeros(1))])
        mock_gds_cls.return_value = mock_gds

        ctx = self._make_context()
        ctx.model_config.model = "test-model"

        strategy = GdsStrategy()
        model = MagicMock()
        result = strategy.load(model, ctx)

        assert result is True
        mock_gds.load_iter.assert_called_once()
        model.load_weights.assert_called_once()
        mock_gds.shutdown.assert_called_once()

    @patch("modelexpress.gds_transfer.is_gds_available", return_value=True)
    @patch("modelexpress.gds_loader.MxGdsLoader")
    def test_gds_failure_returns_false(self, mock_gds_cls, _mock_avail):
        from modelexpress.load_strategy.gds_strategy import GdsStrategy

        mock_gds = MagicMock()
        mock_gds.load_iter.side_effect = RuntimeError("GDS error")
        mock_gds_cls.return_value = mock_gds

        ctx = self._make_context()
        ctx.model_config.model = "test-model"

        strategy = GdsStrategy()
        result = strategy.load(MagicMock(), ctx)

        assert result is False
        mock_gds.shutdown.assert_called_once()

    @patch("modelexpress.gds_transfer.is_gds_available", return_value=False)
    def test_gds_not_available(self, _mock_avail):
        from modelexpress.load_strategy.gds_strategy import GdsStrategy

        ctx = self._make_context()
        strategy = GdsStrategy()
        assert strategy.is_available(ctx) is False
