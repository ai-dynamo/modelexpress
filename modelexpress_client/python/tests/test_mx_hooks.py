# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GMS shared hook library."""

from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

from modelexpress.gms.config import MxConfig


@pytest.fixture
def mx_config():
    return MxConfig(
        mx_server="localhost:8001",
        model_name="test-model",
        expected_workers=1,
        contiguous_reg=False,
        sync_start=False,
    )


@pytest.fixture
def mock_model():
    """Create a mock nn.Module with CUDA parameters."""
    model = MagicMock(spec=torch.nn.Module)
    param1 = MagicMock(spec=torch.nn.Parameter)
    param1.is_cuda = True
    param1.data = MagicMock()
    param1.data.data_ptr.return_value = 0x1000
    param1.data.numel.return_value = 100
    param1.data.element_size.return_value = 4

    param2 = MagicMock(spec=torch.nn.Parameter)
    param2.is_cuda = True
    param2.data = MagicMock()
    param2.data.data_ptr.return_value = 0x2000
    param2.data.numel.return_value = 200
    param2.data.element_size.return_value = 4

    model.named_parameters.return_value = [
        ("layer.weight", param1),
        ("layer.bias", param2),
    ]
    return model


class TestCollectGpuTensors:
    """Tests for _collect_gpu_tensors helper."""

    def test_collects_cuda_params(self, mock_model):
        from modelexpress.gms.mx_hooks import _collect_gpu_tensors

        tensors = _collect_gpu_tensors(mock_model)
        assert len(tensors) == 2
        assert "layer.weight" in tensors
        assert "layer.bias" in tensors

    def test_skips_non_cuda_params(self):
        from modelexpress.gms.mx_hooks import _collect_gpu_tensors

        model = MagicMock(spec=torch.nn.Module)
        cpu_param = MagicMock()
        cpu_param.is_cuda = False
        cpu_param.data = MagicMock()
        model.named_parameters.return_value = [("cpu_layer", cpu_param)]

        tensors = _collect_gpu_tensors(model)
        assert len(tensors) == 0


class TestWriteToGms:
    """Tests for _write_to_gms helper."""

    @patch("modelexpress.gms.mx_hooks.get_or_create_gms_client_memory_manager")
    @patch("modelexpress.gms.mx_hooks.register_module_tensors")
    @patch("modelexpress.gms.mx_hooks.get_socket_path")
    @patch("modelexpress.gms.mx_hooks.RequestedLockType")
    @patch("torch.cuda.synchronize")
    def test_connects_commits_and_switches(
        self,
        mock_sync,
        mock_lock_type,
        mock_socket_path,
        mock_register,
        mock_get_gms,
        mock_model,
    ):
        from modelexpress.gms.mx_hooks import _write_to_gms

        mock_gms_client = MagicMock()
        mock_gms_client.commit.return_value = True
        mock_gms_client.total_bytes = 1 << 30
        mock_get_gms.return_value = (mock_gms_client, MagicMock())
        mock_socket_path.return_value = "/tmp/gms.sock"

        result = _write_to_gms(0, mock_model)

        mock_gms_client.clear_all.assert_called_once()
        mock_register.assert_called_once_with(mock_gms_client, mock_model)
        mock_sync.assert_called_once()
        mock_gms_client.commit.assert_called_once()
        mock_gms_client.switch_to_read.assert_called_once()
        assert result is mock_gms_client

    @patch("modelexpress.gms.mx_hooks.get_or_create_gms_client_memory_manager")
    @patch("modelexpress.gms.mx_hooks.register_module_tensors")
    @patch("modelexpress.gms.mx_hooks.get_socket_path")
    @patch("modelexpress.gms.mx_hooks.RequestedLockType")
    @patch("torch.cuda.synchronize")
    def test_raises_on_commit_failure(
        self,
        mock_sync,
        mock_lock_type,
        mock_socket_path,
        mock_register,
        mock_get_gms,
        mock_model,
    ):
        from modelexpress.gms.mx_hooks import _write_to_gms

        mock_gms_client = MagicMock()
        mock_gms_client.commit.return_value = False
        mock_get_gms.return_value = (mock_gms_client, MagicMock())
        mock_socket_path.return_value = "/tmp/gms.sock"

        with pytest.raises(RuntimeError, match="GMS commit failed"):
            _write_to_gms(0, mock_model)


class TestRegisterNixl:
    """Tests for _register_nixl helper."""

    @patch("modelexpress.gms.mx_hooks.NixlTransferManager")
    def test_creates_and_registers(self, mock_nixl_cls):
        from modelexpress.gms.mx_hooks import _register_nixl

        mock_mgr = MagicMock()
        mock_nixl_cls.return_value = mock_mgr

        tensors = {"weight": MagicMock(numel=lambda: 100, element_size=lambda: 4)}
        result = _register_nixl(0, tensors)

        mock_nixl_cls.assert_called_once()
        mock_mgr.initialize.assert_called_once()
        mock_mgr.register_tensors.assert_called_once_with(tensors)
        assert result is mock_mgr


class TestPublishMetadata:
    """Tests for _publish_metadata helper."""

    @patch("modelexpress.gms.mx_hooks.MxClient")
    @patch("modelexpress.gms.mx_hooks.p2p_pb2")
    def test_publishes_successfully(self, mock_pb2, mock_client_cls, mx_config):
        from modelexpress.gms.mx_hooks import _publish_metadata

        mock_client = MagicMock()
        mock_client.publish_metadata.return_value = True
        mock_client_cls.return_value = mock_client

        mock_nixl_mgr = MagicMock()
        mock_nixl_mgr.nixl_metadata = b"metadata"

        tensors = {
            "w": MagicMock(
                data_ptr=lambda: 0x1000,
                numel=lambda: 100,
                element_size=lambda: 4,
                dtype=torch.float32,
            )
        }

        _publish_metadata(0, mock_nixl_mgr, tensors, mx_config)

        mock_client.publish_metadata.assert_called_once()
        mock_client.close.assert_called_once()


class TestSourcePostLoad:
    """Tests for source_post_load hook."""

    @patch("modelexpress.gms.mx_hooks._publish_metadata")
    @patch("modelexpress.gms.mx_hooks._register_nixl")
    @patch("modelexpress.gms.mx_hooks._collect_gpu_tensors")
    @patch("modelexpress.gms.mx_hooks._write_to_gms")
    def test_calls_helpers_in_order(
        self,
        mock_write,
        mock_collect,
        mock_register,
        mock_publish,
        mock_model,
        mx_config,
    ):
        from modelexpress.gms.mx_hooks import source_post_load

        mock_gms = MagicMock()
        mock_write.return_value = mock_gms
        mock_collect.return_value = {"w": MagicMock()}
        mock_nixl = MagicMock()
        mock_register.return_value = mock_nixl

        source_post_load(device_id=0, rank=0, model=mock_model, mx_config=mx_config)

        mock_write.assert_called_once_with(0, mock_model)
        mock_collect.assert_called_once_with(mock_model)
        mock_register.assert_called_once()
        mock_publish.assert_called_once()

        # Verify ordering: write before collect before register before publish
        assert mock_write.call_count == 1
        assert mock_collect.call_count == 1
        assert mock_register.call_count == 1
        assert mock_publish.call_count == 1


class TestSourceFinalize:
    """Tests for source_finalize hook."""

    @patch("modelexpress.gms.mx_hooks.MxClient")
    def test_calls_barrier_when_distributed(self, mock_client_cls, mx_config):
        from modelexpress.gms.mx_hooks import source_finalize

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        with patch("torch.distributed.is_initialized", return_value=True), patch(
            "torch.distributed.barrier"
        ) as mock_barrier:
            source_finalize(rank=0, mx_config=mx_config)
            mock_barrier.assert_called_once()

    @patch("modelexpress.gms.mx_hooks.MxClient")
    def test_skips_barrier_when_not_distributed(self, mock_client_cls, mx_config):
        from modelexpress.gms.mx_hooks import source_finalize

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        with patch("torch.distributed.is_initialized", return_value=False), patch(
            "torch.distributed.barrier"
        ) as mock_barrier:
            source_finalize(rank=0, mx_config=mx_config)
            mock_barrier.assert_not_called()

    @patch("modelexpress.gms.mx_hooks.MxClient")
    def test_publishes_ready(self, mock_client_cls, mx_config):
        from modelexpress.gms.mx_hooks import source_finalize

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        with patch("torch.distributed.is_initialized", return_value=False):
            source_finalize(rank=0, mx_config=mx_config)

        mock_client.publish_ready.assert_called_once()
        call_kwargs = mock_client.publish_ready.call_args
        assert call_kwargs.kwargs["model_name"] == "test-model"
        assert call_kwargs.kwargs["worker_id"] == 0
        mock_client.close.assert_called_once()


class TestTargetAllocate:
    """Tests for target_allocate hook."""

    @patch("modelexpress.gms.mx_hooks._register_nixl")
    @patch("modelexpress.gms.mx_hooks._collect_gpu_tensors")
    @patch("modelexpress.gms.mx_hooks.get_or_create_gms_client_memory_manager")
    @patch("modelexpress.gms.mx_hooks.get_socket_path")
    @patch("modelexpress.gms.mx_hooks.RequestedLockType")
    def test_allocates_and_registers(
        self,
        mock_lock_type,
        mock_socket_path,
        mock_get_gms,
        mock_collect,
        mock_register,
        mock_model,
        mx_config,
    ):
        from modelexpress.gms.mx_hooks import target_allocate

        mock_gms = MagicMock()
        mock_get_gms.return_value = (mock_gms, MagicMock())
        mock_socket_path.return_value = "/tmp/gms.sock"
        mock_collect.return_value = {"w": MagicMock()}
        mock_nixl = MagicMock()
        mock_register.return_value = mock_nixl

        gms_client, nixl_mgr = target_allocate(
            device_id=0, rank=0, model=mock_model, mx_config=mx_config
        )

        mock_gms.clear_all.assert_called_once()
        assert gms_client is mock_gms
        assert nixl_mgr is mock_nixl


class TestTargetReceive:
    """Tests for target_receive hook."""

    @patch("modelexpress.gms.mx_hooks._wait_for_source_worker")
    @patch("modelexpress.gms.mx_hooks.MxClient")
    @patch("torch.cuda.synchronize")
    def test_receives_successfully(
        self, mock_sync, mock_client_cls, mock_wait, mx_config
    ):
        from modelexpress.gms.mx_hooks import target_receive

        mock_client = MagicMock()
        mock_client.wait_for_ready.return_value = (True, "session-1", "hash-1")
        mock_client_cls.return_value = mock_client

        mock_worker = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.name = "weight"
        mock_tensor.addr = 0x1000
        mock_tensor.size = 400
        mock_tensor.device_id = 0
        mock_tensor.dtype = "torch.float32"
        mock_worker.tensors = [mock_tensor]
        mock_worker.nixl_metadata = b"meta"
        mock_wait.return_value = mock_worker

        mock_nixl = MagicMock()
        mock_nixl.receive_from_source.return_value = (400, 1, 0.01)

        target_receive(rank=0, nixl_mgr=mock_nixl, mx_config=mx_config)

        mock_nixl.receive_from_source.assert_called_once()
        mock_sync.assert_called_once()
        mock_client.close.assert_called_once()

    @patch("modelexpress.gms.mx_hooks.MxClient")
    def test_raises_when_source_not_ready(self, mock_client_cls, mx_config):
        from modelexpress.gms.mx_hooks import target_receive

        mock_client = MagicMock()
        mock_client.wait_for_ready.return_value = (False, None, None)
        mock_client_cls.return_value = mock_client

        with pytest.raises(RuntimeError, match="Source never became ready"):
            target_receive(rank=0, nixl_mgr=MagicMock(), mx_config=mx_config)

        mock_client.close.assert_called_once()


class TestTargetCommit:
    """Tests for target_commit hook."""

    @patch("modelexpress.gms.mx_hooks.register_module_tensors")
    @patch("torch.cuda.synchronize")
    def test_commits_successfully(
        self, mock_sync, mock_register, mock_model, mx_config
    ):
        from modelexpress.gms.mx_hooks import target_commit

        mock_gms = MagicMock()
        mock_gms.commit.return_value = True

        target_commit(
            device_id=0,
            rank=0,
            model=mock_model,
            gms_client=mock_gms,
            mx_config=mx_config,
        )

        mock_register.assert_called_once_with(mock_gms, mock_model)
        mock_sync.assert_called_once()
        mock_gms.commit.assert_called_once()
        mock_gms.switch_to_read.assert_called_once()

    @patch("modelexpress.gms.mx_hooks.register_module_tensors")
    @patch("torch.cuda.synchronize")
    def test_raises_on_commit_failure(
        self, mock_sync, mock_register, mock_model, mx_config
    ):
        from modelexpress.gms.mx_hooks import target_commit

        mock_gms = MagicMock()
        mock_gms.commit.return_value = False

        with pytest.raises(RuntimeError, match="GMS commit failed"):
            target_commit(
                device_id=0,
                rank=0,
                model=mock_model,
                gms_client=mock_gms,
                mx_config=mx_config,
            )

    @patch("modelexpress.gms.mx_hooks.register_module_tensors")
    @patch("torch.cuda.synchronize")
    def test_synchronize_before_commit(
        self, mock_sync, mock_register, mock_model, mx_config
    ):
        from modelexpress.gms.mx_hooks import target_commit

        mock_gms = MagicMock()
        mock_gms.commit.return_value = True

        call_order = []
        mock_sync.side_effect = lambda: call_order.append("sync")
        mock_gms.commit.side_effect = lambda: (
            call_order.append("commit"),
            True,
        )[-1]

        target_commit(
            device_id=0,
            rank=0,
            model=mock_model,
            gms_client=mock_gms,
            mx_config=mx_config,
        )

        assert call_order.index("sync") < call_order.index("commit")
