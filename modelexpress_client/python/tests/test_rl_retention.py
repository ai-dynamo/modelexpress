# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import modelexpress.rl_transfer as rl_transfer_module
from modelexpress import p2p_pb2
from modelexpress.rl_metadata import get_rl_source_metadata
from modelexpress.rl_transfer import RlNixlWeightTransfer, build_rl_base_identity


def _base_identity():
    return build_rl_base_identity(
        model_name="test-model",
        mx_version="0.3.0",
        backend_framework="vllm",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=0,
        dtype="bfloat16",
        quantization="",
        revision="",
    )


class _RetentionMxClient:
    def __init__(self):
        self.published = []
        self.status_updates = []

    def publish_metadata(self, identity, worker, worker_id):
        metadata = get_rl_source_metadata(identity)
        source_id = f"source-v{metadata.model_version}-{len(self.published)}"
        self.published.append((source_id, identity, worker, worker_id))
        return source_id

    def update_status(self, mx_source_id, worker_id, worker_rank, status):
        self.status_updates.append((mx_source_id, worker_id, worker_rank, status))
        return True


class _RetentionNixlManager:
    instances = []

    def __init__(self, *args, **kwargs):
        self.nixl_metadata = f"nixl-{len(type(self).instances)}".encode()
        self.tensor_descriptors = []
        self.registered_tensors = {}
        self.shutdown_called = False
        type(self).instances.append(self)

    def initialize(self):
        pass

    def register_tensors(self, tensors):
        self.registered_tensors = dict(tensors)
        self.tensor_descriptors = [
            p2p_pb2.TensorDescriptor(
                name=name,
                addr=1 + index,
                size=tensor.numel() * tensor.element_size(),
                device_id=tensor.device.index or 0,
                dtype=str(tensor.dtype),
            )
            for index, (name, tensor) in enumerate(tensors.items())
        ]

    def shutdown(self):
        self.shutdown_called = True


class _RetentionHeartbeat:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stop_calls = []
        type(self).instances.append(self)

    def start(self):
        self.started = True

    def stop(self, *, mark_stale=True):
        self.stop_calls.append(mark_stale)


def _patch_transfer_lifecycle(monkeypatch):
    _RetentionNixlManager.instances = []
    _RetentionHeartbeat.instances = []
    monkeypatch.setattr(
        rl_transfer_module,
        "NixlTransferManager",
        _RetentionNixlManager,
    )
    monkeypatch.setattr(rl_transfer_module, "HeartbeatThread", _RetentionHeartbeat)


def test_retained_publish_snapshots_tensors_and_keeps_sources_ready(monkeypatch):
    _patch_transfer_lifecycle(monkeypatch)
    mx_client = _RetentionMxClient()
    transfer = RlNixlWeightTransfer(
        mx_client=mx_client,
        base_identity=_base_identity(),
        worker_id="worker-local",
        retain_latest_k=2,
    )
    tensor = torch.tensor([1.0])

    source_v1 = transfer.publish_tensors({"w": tensor}, model_version=1)
    tensor.fill_(2.0)
    source_v2 = transfer.publish_tensors({"w": tensor}, model_version=2)

    assert source_v1 == "source-v1-0"
    assert source_v2 == "source-v2-1"
    assert [update[-1] for update in mx_client.status_updates] == [
        p2p_pb2.SOURCE_STATUS_READY,
        p2p_pb2.SOURCE_STATUS_READY,
    ]
    assert len(transfer._published_sources) == 2
    assert _RetentionNixlManager.instances[0].registered_tensors["w"] is not tensor
    assert _RetentionNixlManager.instances[0].registered_tensors["w"].item() == 1.0
    assert _RetentionNixlManager.instances[1].registered_tensors["w"].item() == 2.0


def test_retained_publish_prunes_versions_outside_window(monkeypatch):
    _patch_transfer_lifecycle(monkeypatch)
    mx_client = _RetentionMxClient()
    transfer = RlNixlWeightTransfer(
        mx_client=mx_client,
        base_identity=_base_identity(),
        worker_id="worker-local",
        retain_latest_k=2,
    )

    for version in (1, 2, 3):
        transfer.publish_tensors({"w": torch.tensor([float(version)])}, model_version=version)

    assert [source.mx_source_id for source in transfer._published_sources] == [
        "source-v2-1",
        "source-v3-2",
    ]
    assert mx_client.status_updates[-1] == (
        "source-v1-0",
        "worker-local",
        0,
        p2p_pb2.SOURCE_STATUS_STALE,
    )
    assert _RetentionNixlManager.instances[0].shutdown_called
    assert _RetentionHeartbeat.instances[0].stop_calls == [False]


def test_single_version_publish_replaces_previous_source(monkeypatch):
    _patch_transfer_lifecycle(monkeypatch)
    mx_client = _RetentionMxClient()
    transfer = RlNixlWeightTransfer(
        mx_client=mx_client,
        base_identity=_base_identity(),
        worker_id="worker-local",
        retain_latest_k=1,
    )

    transfer.publish_tensors({"w": torch.tensor([1.0])}, model_version=1)
    transfer.publish_tensors({"w": torch.tensor([2.0])}, model_version=2)

    assert [source.mx_source_id for source in transfer._published_sources] == [
        "source-v2-1",
    ]
    assert mx_client.status_updates == [
        ("source-v1-0", "worker-local", 0, p2p_pb2.SOURCE_STATUS_READY),
        ("source-v1-0", "worker-local", 0, p2p_pb2.SOURCE_STATUS_STALE),
        ("source-v2-1", "worker-local", 0, p2p_pb2.SOURCE_STATUS_READY),
    ]
    assert _RetentionNixlManager.instances[0].registered_tensors["w"] is not None
    assert _RetentionNixlManager.instances[0].shutdown_called


def test_republishing_same_version_replaces_duplicate_source(monkeypatch):
    _patch_transfer_lifecycle(monkeypatch)
    mx_client = _RetentionMxClient()
    transfer = RlNixlWeightTransfer(
        mx_client=mx_client,
        base_identity=_base_identity(),
        worker_id="worker-local",
        retain_latest_k=2,
    )

    transfer.publish_tensors({"w": torch.tensor([1.0])}, model_version=1)
    transfer.publish_tensors({"w": torch.tensor([1.5])}, model_version=1)

    assert [source.mx_source_id for source in transfer._published_sources] == [
        "source-v1-1",
    ]
    assert mx_client.status_updates == [
        ("source-v1-0", "worker-local", 0, p2p_pb2.SOURCE_STATUS_READY),
        ("source-v1-0", "worker-local", 0, p2p_pb2.SOURCE_STATUS_STALE),
        ("source-v1-1", "worker-local", 0, p2p_pb2.SOURCE_STATUS_READY),
    ]
    assert _RetentionNixlManager.instances[0].shutdown_called


def test_finalize_marks_all_retained_sources_stale(monkeypatch):
    _patch_transfer_lifecycle(monkeypatch)
    mx_client = _RetentionMxClient()
    transfer = RlNixlWeightTransfer(
        mx_client=mx_client,
        base_identity=_base_identity(),
        worker_id="worker-local",
        retain_latest_k=2,
    )
    transfer.publish_tensors({"w": torch.tensor([1.0])}, model_version=1)
    transfer.publish_tensors({"w": torch.tensor([2.0])}, model_version=2)

    transfer.finalize()

    assert transfer._published_sources == []
    assert mx_client.status_updates[-2:] == [
        ("source-v1-0", "worker-local", 0, p2p_pb2.SOURCE_STATUS_STALE),
        ("source-v2-1", "worker-local", 0, p2p_pb2.SOURCE_STATUS_STALE),
    ]
    assert all(manager.shutdown_called for manager in _RetentionNixlManager.instances)
    assert [heartbeat.stop_calls for heartbeat in _RetentionHeartbeat.instances] == [
        [False],
        [False],
    ]
