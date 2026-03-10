# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM GMS launcher."""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("torch")

from modelexpress.gms.config import (
    MxConfig,
    GmsConfig,
    GmsMode,
    WeightSourceType,
)


@pytest.fixture
def source_config():
    return GmsConfig(model="test-model", mode=GmsMode.SOURCE)


@pytest.fixture
def target_config():
    return GmsConfig(model="test-model", mode=GmsMode.TARGET)


@pytest.fixture
def multi_gpu_source_config():
    return GmsConfig(
        model="test-model", mode=GmsMode.SOURCE, tp_size=2, ep_size=2
    )


class TestRun:
    """Tests for the run() dispatch function."""

    @patch("modelexpress.gms.launchers.vllm._run_source")
    def test_dispatches_source(self, mock_run_source, source_config):
        from modelexpress.gms.launchers.vllm import run

        run(source_config)
        mock_run_source.assert_called_once()

    @patch("modelexpress.gms.launchers.vllm._run_target")
    def test_dispatches_target(self, mock_run_target, target_config):
        from modelexpress.gms.launchers.vllm import run

        run(target_config)
        mock_run_target.assert_called_once()


class TestRunSource:
    """Tests for _run_source."""

    @patch("modelexpress.gms.launchers.vllm._source_worker")
    @patch("modelexpress.gms.launchers.vllm._find_free_port", return_value=12345)
    def test_single_worker_no_spawn(self, mock_port, mock_worker, source_config):
        from modelexpress.gms.launchers.vllm import _run_source

        mx_config = source_config.to_mx_config()
        _run_source(source_config, mx_config)

        # Single worker: called directly, not via mp.spawn
        mock_worker.assert_called_once_with(0, source_config, mx_config, 12345)

    @patch("modelexpress.gms.launchers.vllm.mp.spawn")
    @patch("modelexpress.gms.launchers.vllm._find_free_port", return_value=12345)
    def test_multi_worker_uses_spawn(self, mock_port, mock_spawn, multi_gpu_source_config):
        from modelexpress.gms.launchers.vllm import _run_source

        mx_config = multi_gpu_source_config.to_mx_config()
        _run_source(multi_gpu_source_config, mx_config)

        mock_spawn.assert_called_once()
        call_kwargs = mock_spawn.call_args
        assert call_kwargs.kwargs["nprocs"] == 4  # tp=2, ep=2


class TestRunTarget:
    """Tests for _run_target."""

    @patch("modelexpress.gms.launchers.vllm._target_worker")
    @patch("modelexpress.gms.launchers.vllm._find_free_port", return_value=12345)
    def test_single_worker_no_spawn(self, mock_port, mock_worker, target_config):
        from modelexpress.gms.launchers.vllm import _run_target

        mx_config = target_config.to_mx_config()
        _run_target(target_config, mx_config)

        mock_worker.assert_called_once_with(0, target_config, mx_config, 12345)

    @patch("modelexpress.gms.launchers.vllm.mp.spawn")
    @patch("modelexpress.gms.launchers.vllm._find_free_port", return_value=12345)
    def test_multi_worker_uses_spawn(self, mock_port, mock_spawn):
        from modelexpress.gms.launchers.vllm import _run_target

        config = GmsConfig(
            model="test-model", mode=GmsMode.TARGET, tp_size=4
        )
        mx_config = config.to_mx_config()
        _run_target(config, mx_config)

        mock_spawn.assert_called_once()
        call_kwargs = mock_spawn.call_args
        assert call_kwargs.kwargs["nprocs"] == 4


class TestSourceWorker:
    """Tests for _source_worker."""

    @patch("modelexpress.gms.launchers.vllm.source_finalize")
    @patch("modelexpress.gms.launchers.vllm.process_weights_after_loading")
    @patch("modelexpress.gms.launchers.vllm.source_post_load")
    @patch("modelexpress.gms.launchers.vllm._load_model")
    @patch("modelexpress.gms.launchers.vllm._get_weights_iterator")
    @patch("modelexpress.gms.launchers.vllm._build_vllm_configs")
    @patch("modelexpress.gms.launchers.vllm._init_vllm_distributed")
    @patch("torch.cuda.set_device")
    def test_source_worker_hook_order(
        self,
        mock_set_device,
        mock_init_dist,
        mock_build,
        mock_get_weights,
        mock_load,
        mock_post_load,
        mock_process,
        mock_finalize,
        source_config,
    ):
        from modelexpress.gms.launchers.vllm import _source_worker

        mock_vllm_config = MagicMock()
        mock_model_config = MagicMock()
        mock_load_config = MagicMock()
        mock_build.return_value = (mock_vllm_config, mock_model_config, mock_load_config)
        mock_get_weights.return_value = None
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        mx_config = source_config.to_mx_config()
        _source_worker(0, source_config, mx_config, 12345)

        # Verify hook order: load -> post_load -> process -> finalize
        call_order = []
        mock_load.side_effect = lambda *a, **k: call_order.append("load")
        mock_post_load.side_effect = lambda **k: call_order.append("post_load")
        mock_process.side_effect = lambda *a: call_order.append("process")
        mock_finalize.side_effect = lambda **k: call_order.append("finalize")

        # Verify all were called
        mock_set_device.assert_called_once_with(0)
        mock_init_dist.assert_called_once()
        mock_load.assert_called_once()
        mock_post_load.assert_called_once()
        mock_process.assert_called_once()
        mock_finalize.assert_called_once()


class TestTargetWorker:
    """Tests for _target_worker."""

    @patch("modelexpress.gms.launchers.vllm.target_commit")
    @patch("modelexpress.gms.launchers.vllm.process_weights_after_loading")
    @patch("modelexpress.gms.launchers.vllm.target_receive")
    @patch("modelexpress.gms.launchers.vllm.target_allocate")
    @patch("modelexpress.gms.launchers.vllm._create_dummy_model")
    @patch("modelexpress.gms.launchers.vllm._build_vllm_configs")
    @patch("modelexpress.gms.launchers.vllm._init_vllm_distributed")
    @patch("torch.cuda.set_device")
    def test_target_worker_hook_order(
        self,
        mock_set_device,
        mock_init_dist,
        mock_build,
        mock_dummy,
        mock_allocate,
        mock_receive,
        mock_process,
        mock_commit,
        target_config,
    ):
        from modelexpress.gms.launchers.vllm import _target_worker

        mock_vllm_config = MagicMock()
        mock_model_config = MagicMock()
        mock_load_config = MagicMock()
        mock_build.return_value = (mock_vllm_config, mock_model_config, mock_load_config)
        mock_model = MagicMock()
        mock_dummy.return_value = mock_model
        mock_gms = MagicMock()
        mock_nixl = MagicMock()
        mock_allocate.return_value = (mock_gms, mock_nixl)

        mx_config = target_config.to_mx_config()
        _target_worker(0, target_config, mx_config, 12345)

        mock_set_device.assert_called_once_with(0)
        mock_init_dist.assert_called_once()
        mock_dummy.assert_called_once()
        mock_allocate.assert_called_once()
        mock_receive.assert_called_once()
        mock_process.assert_called_once()
        mock_commit.assert_called_once()


class TestGetWeightsIterator:
    """Tests for _get_weights_iterator."""

    def test_disk_returns_none(self):
        from modelexpress.gms.launchers.vllm import _get_weights_iterator

        config = GmsConfig(model="test", weight_source=WeightSourceType.DISK)
        result = _get_weights_iterator(config, MagicMock())
        assert result is None

    def test_gds_raises(self):
        from modelexpress.gms.launchers.vllm import _get_weights_iterator

        config = GmsConfig(model="test", weight_source=WeightSourceType.GDS)
        with pytest.raises(NotImplementedError):
            _get_weights_iterator(config, MagicMock())

    def test_s3_raises(self):
        from modelexpress.gms.launchers.vllm import _get_weights_iterator

        config = GmsConfig(
            model="test",
            weight_source=WeightSourceType.S3,
            s3_bucket="bucket",
            s3_prefix="prefix",
        )
        with pytest.raises(NotImplementedError):
            _get_weights_iterator(config, MagicMock())


class TestFindFreePort:
    """Tests for _find_free_port."""

    def test_returns_valid_port(self):
        from modelexpress.gms.launchers.vllm import _find_free_port

        port = _find_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535
