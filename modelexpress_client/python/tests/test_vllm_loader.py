# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared helpers and MxModelLoader detection logic."""

from unittest.mock import MagicMock, patch, call

import pytest
import torch
import torch.nn as nn

from modelexpress import p2p_pb2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loader():
    """Return an MxModelLoader with a fresh mock MxClient."""
    with patch("modelexpress.vllm_loader.DefaultModelLoader"), \
         patch("modelexpress.vllm_loader.DummyModelLoader"):
        load_config = MagicMock()
        load_config.load_format = "mx"
        load_config.device = None
        from modelexpress.vllm_loader import MxModelLoader
        loader = MxModelLoader(load_config)
    loader._mx_client = MagicMock()
    return loader


def _make_identity(model_name="test-model"):
    return p2p_pb2.SourceIdentity(model_name=model_name)


def _make_worker(rank=0, n_tensors=3):
    tensors = [
        p2p_pb2.TensorDescriptor(name=f"t{i}", addr=0x1000 + i, size=1024, device_id=rank, dtype="bfloat16")
        for i in range(n_tensors)
    ]
    return p2p_pb2.WorkerMetadata(worker_rank=rank, tensors=tensors)


def _make_instance_ref(mx_source_id="abc123def456abcd", instance_id="inst-1", model_name="test-model"):
    return p2p_pb2.SourceInstanceRef(
        mx_source_id=mx_source_id,
        instance_id=instance_id,
        model_name=model_name,
    )


def _make_metadata_resp(found=True, rank=0, mx_source_id="abc123def456abcd", instance_id="inst-1"):
    workers = [_make_worker(rank=rank)] if found else []
    return p2p_pb2.GetMetadataResponse(
        found=found,
        workers=workers,
        mx_source_id=mx_source_id,
        instance_id=instance_id,
    )


# ---------------------------------------------------------------------------
# _collect_cuda_tensors
# ---------------------------------------------------------------------------


class TestCollectCudaTensors:
    def test_empty_model(self):
        from modelexpress.vllm_loader import _collect_cuda_tensors
        assert _collect_cuda_tensors(nn.Module()) == {}

    def test_cpu_only_model(self):
        from modelexpress.vllm_loader import _collect_cuda_tensors
        assert _collect_cuda_tensors(nn.Linear(4, 2, bias=False)) == {}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_model(self):
        from modelexpress.vllm_loader import _collect_cuda_tensors
        model = nn.Linear(4, 2, bias=True).cuda()
        result = _collect_cuda_tensors(model)
        assert len(result) == 2
        assert all(t.is_cuda for t in result.values())


# ---------------------------------------------------------------------------
# Abstract method completeness
# ---------------------------------------------------------------------------


class TestAbstractMethodCompleteness:
    def test_instantiation_succeeds(self):
        assert _make_loader() is not None

    def test_no_remaining_abstract_methods(self):
        from modelexpress.vllm_loader import MxModelLoader
        remaining = getattr(MxModelLoader, "__abstractmethods__", frozenset())
        assert remaining == frozenset()

    def test_download_model_delegates_to_disk_loader(self):
        loader = _make_loader()
        cfg = MagicMock()
        loader.download_model(cfg)
        loader._disk_loader.download_model.assert_called_once_with(cfg)

    def test_load_weights_delegates_to_disk_loader(self):
        loader = _make_loader()
        model, cfg = MagicMock(), MagicMock()
        loader.load_weights(model, cfg)
        loader._disk_loader.load_weights.assert_called_once_with(model, cfg)


# ---------------------------------------------------------------------------
# _find_source_instances
# ---------------------------------------------------------------------------


class TestFindSourceInstances:
    def test_returns_empty_when_nixl_unavailable(self):
        loader = _make_loader()
        with patch("modelexpress.vllm_loader.is_nixl_available", return_value=False):
            result = loader._find_source_instances(_make_identity(), device_id=0)
        assert result == []
        loader._mx_client.list_sources.assert_not_called()

    def test_returns_empty_when_no_model_name(self):
        loader = _make_loader()
        with patch("modelexpress.vllm_loader.is_nixl_available", return_value=True):
            result = loader._find_source_instances(_make_identity(model_name=""), device_id=0)
        assert result == []

    def test_returns_empty_when_no_instances(self):
        loader = _make_loader()
        loader._mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(instances=[])
        with patch("modelexpress.vllm_loader.is_nixl_available", return_value=True):
            result = loader._find_source_instances(_make_identity(), device_id=0)
        assert result == []

    def test_returns_empty_when_list_sources_raises(self):
        loader = _make_loader()
        loader._mx_client.list_sources.side_effect = RuntimeError("server unreachable")
        with patch("modelexpress.vllm_loader.is_nixl_available", return_value=True):
            result = loader._find_source_instances(_make_identity(), device_id=0)
        assert result == []

    def test_calls_list_sources_with_ready_filter(self):
        loader = _make_loader()
        loader._mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(instances=[])
        identity = _make_identity()
        with patch("modelexpress.vllm_loader.is_nixl_available", return_value=True):
            loader._find_source_instances(identity, device_id=0)
        loader._mx_client.list_sources.assert_called_once_with(
            identity=identity,
            status_filter=p2p_pb2.SOURCE_STATUS_READY,
        )

    def test_returns_instance_with_matching_rank(self):
        loader = _make_loader()
        inst = _make_instance_ref()
        loader._mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(instances=[inst])
        loader._mx_client.get_metadata.return_value = _make_metadata_resp(rank=0)
        with patch("modelexpress.vllm_loader.is_nixl_available", return_value=True), \
             patch("modelexpress.vllm_loader.random.shuffle"):
            result = loader._find_source_instances(_make_identity(), device_id=0)
        assert len(result) == 1
        w, mx_id, iid = result[0]
        assert w.worker_rank == 0
        assert mx_id == inst.mx_source_id
        assert iid == inst.instance_id

    def test_skips_instance_when_metadata_not_found(self):
        loader = _make_loader()
        loader._mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
            instances=[_make_instance_ref()]
        )
        loader._mx_client.get_metadata.return_value = _make_metadata_resp(found=False)
        with patch("modelexpress.vllm_loader.is_nixl_available", return_value=True):
            result = loader._find_source_instances(_make_identity(), device_id=0)
        assert result == []

    def test_skips_instance_when_no_matching_rank(self):
        loader = _make_loader()
        loader._mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
            instances=[_make_instance_ref()]
        )
        # Instance has rank 1, we are rank 0
        loader._mx_client.get_metadata.return_value = _make_metadata_resp(rank=1)
        with patch("modelexpress.vllm_loader.is_nixl_available", return_value=True):
            result = loader._find_source_instances(_make_identity(), device_id=0)
        assert result == []

    def test_skips_instance_when_get_metadata_raises(self):
        loader = _make_loader()
        loader._mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
            instances=[_make_instance_ref()]
        )
        loader._mx_client.get_metadata.side_effect = RuntimeError("connection refused")
        with patch("modelexpress.vllm_loader.is_nixl_available", return_value=True):
            result = loader._find_source_instances(_make_identity(), device_id=0)
        assert result == []

    def test_returns_multiple_matching_instances(self):
        loader = _make_loader()
        insts = [_make_instance_ref(instance_id="inst-1"), _make_instance_ref(instance_id="inst-2")]
        loader._mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(instances=insts)
        loader._mx_client.get_metadata.side_effect = [
            _make_metadata_resp(rank=0, instance_id="inst-1"),
            _make_metadata_resp(rank=0, instance_id="inst-2"),
        ]
        with patch("modelexpress.vllm_loader.is_nixl_available", return_value=True), \
             patch("modelexpress.vllm_loader.random.shuffle"):
            result = loader._find_source_instances(_make_identity(), device_id=0)
        assert len(result) == 2

    def test_skips_instances_without_tensors(self):
        loader = _make_loader()
        loader._mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
            instances=[_make_instance_ref()]
        )
        # Worker has matching rank but no tensors
        empty_worker = p2p_pb2.WorkerMetadata(worker_rank=0, tensors=[])
        loader._mx_client.get_metadata.return_value = p2p_pb2.GetMetadataResponse(
            found=True, workers=[empty_worker]
        )
        with patch("modelexpress.vllm_loader.is_nixl_available", return_value=True):
            result = loader._find_source_instances(_make_identity(), device_id=0)
        assert result == []


# ---------------------------------------------------------------------------
# STALE marking on transfer failure
# ---------------------------------------------------------------------------


class TestStaleMarkingOnFailure:
    def test_marks_stale_on_failed_instance(self):
        """Failed instance is marked STALE before trying the next one."""
        loader = _make_loader()
        attempts = []

        def fake_load_as_target(_model, _cfg, _dev, _did, _ident, _worker, mx_source_id, instance_id):
            attempts.append(instance_id)
            if instance_id == "inst-1":
                raise RuntimeError("NIXL connect failed")

        loader._load_as_target = fake_load_as_target

        instances = [
            (_make_worker(rank=0), "src-id", "inst-1"),
            (_make_worker(rank=0), "src-id", "inst-2"),
        ]

        loaded_as_target = False
        for source_worker, mx_source_id, instance_id in instances:
            try:
                loader._load_as_target(None, None, None, 0, None, source_worker, mx_source_id, instance_id)
                loaded_as_target = True
                break
            except Exception:
                loader._mx_client.update_status(
                    mx_source_id=mx_source_id,
                    instance_id=instance_id,
                    worker_id=0,
                    status=p2p_pb2.SOURCE_STATUS_STALE,
                )

        assert loaded_as_target is True
        assert attempts == ["inst-1", "inst-2"]
        loader._mx_client.update_status.assert_called_once_with(
            mx_source_id="src-id",
            instance_id="inst-1",
            worker_id=0,
            status=p2p_pb2.SOURCE_STATUS_STALE,
        )

    def test_marks_all_stale_when_all_instances_fail(self):
        """Every failed instance is marked STALE."""
        loader = _make_loader()
        instances = [(_make_worker(rank=0), "src-id", f"inst-{i}") for i in range(3)]

        loaded_as_target = False
        for source_worker, mx_source_id, instance_id in instances:
            try:
                raise RuntimeError("transfer error")
            except Exception:
                loader._mx_client.update_status(
                    mx_source_id=mx_source_id,
                    instance_id=instance_id,
                    worker_id=0,
                    status=p2p_pb2.SOURCE_STATUS_STALE,
                )

        assert not loaded_as_target
        assert loader._mx_client.update_status.call_count == 3
        stale_instances = [
            c.kwargs["instance_id"] for c in loader._mx_client.update_status.call_args_list
        ]
        assert stale_instances == ["inst-0", "inst-1", "inst-2"]


# ---------------------------------------------------------------------------
# _publish_metadata_and_ready
# ---------------------------------------------------------------------------


class TestPublishMetadataAndReady:
    def test_calls_publish_then_update_status_ready(self):
        from modelexpress.vllm_loader import _publish_metadata_and_ready

        mx_client = MagicMock()
        mx_client.publish_metadata.return_value = "abc123def456abcd"
        mx_client.update_status.return_value = True

        nixl_manager = MagicMock()
        nixl_manager.nixl_metadata = b"nixl-data"

        tensors = {}
        for i in range(3):
            t = MagicMock(spec=torch.Tensor)
            t.data_ptr.return_value = 0x1000 + i * 1024
            t.numel.return_value = 256
            t.element_size.return_value = 2
            t.dtype = torch.bfloat16
            tensors[f"layer.{i}.weight"] = t

        identity = _make_identity("my-model")
        with patch.dict("os.environ", {"MX_CONTIGUOUS_REG": "0"}):
            _publish_metadata_and_ready(mx_client, nixl_manager, tensors, device_id=2, identity=identity, instance_id="inst-uuid")

        mx_client.publish_metadata.assert_called_once()
        call_args = mx_client.publish_metadata.call_args
        assert call_args.args[0] is identity
        assert call_args.args[2] == "inst-uuid"

        mx_client.update_status.assert_called_once_with(
            mx_source_id="abc123def456abcd",
            instance_id="inst-uuid",
            worker_id=2,
            status=p2p_pb2.SOURCE_STATUS_READY,
        )

    def test_update_status_failure_is_logged_not_raised(self):
        """update_status returning False should not propagate as an exception."""
        from modelexpress.vllm_loader import _publish_metadata_and_ready

        mx_client = MagicMock()
        mx_client.publish_metadata.return_value = "abc123def456abcd"
        mx_client.update_status.return_value = False  # server rejected

        nixl_manager = MagicMock()
        nixl_manager.nixl_metadata = b"data"

        identity = _make_identity()
        with patch.dict("os.environ", {"MX_CONTIGUOUS_REG": "0"}):
            # Must not raise
            _publish_metadata_and_ready(mx_client, nixl_manager, {}, device_id=0, identity=identity, instance_id="inst-1")

    def test_publish_failure_raises(self):
        """publish_metadata raising should propagate out."""
        from modelexpress.vllm_loader import _publish_metadata_and_ready

        mx_client = MagicMock()
        mx_client.publish_metadata.side_effect = RuntimeError("grpc error")

        nixl_manager = MagicMock()
        nixl_manager.nixl_metadata = b"data"

        identity = _make_identity()
        with patch.dict("os.environ", {"MX_CONTIGUOUS_REG": "0"}):
            with pytest.raises(RuntimeError, match="grpc error"):
                _publish_metadata_and_ready(mx_client, nixl_manager, {}, device_id=0, identity=identity, instance_id="inst-1")
