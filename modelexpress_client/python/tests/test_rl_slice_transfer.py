# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
import torch

import modelexpress.rl_transfer as rl_transfer_module
from modelexpress import p2p_pb2
from modelexpress.rl_metadata import RlSourceMetadata, RlSourceRole, with_rl_source_metadata
from modelexpress.rl_reshard import (
    TensorReceiveSpec,
    TensorShardSpec,
    TensorSlice,
    TransferPlan,
    TransferPlanEntry,
    plan_dense_reshard_transfers,
)
from modelexpress.rl_slice_transfer import (
    build_grouped_slice_transfer_manifest,
    build_slice_transfer_manifest,
)
from modelexpress.rl_transfer import RlNixlWeightTransfer, build_rl_base_identity
from modelexpress.types import TensorDescriptor


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


def _source_ref(
    shape_registry,
    *,
    mx_source_id="source-a",
    worker_id="worker-a",
    worker_rank=0,
    model_version=5,
    role=RlSourceRole.TRAINER,
    source_world_size=1,
):
    identity = with_rl_source_metadata(
        _base_identity(),
        RlSourceMetadata(
            model_version=model_version,
            role=role,
            world_size=source_world_size,
            shape_registry=shape_registry,
        ),
    )
    return p2p_pb2.SourceInstanceRef(
        mx_source_id=mx_source_id,
        worker_id=worker_id,
        model_name="test-model",
        worker_rank=worker_rank,
        identity=identity,
    )


def _source_spec(**kwargs):
    values = {
        "name": "w",
        "worker_rank": 0,
        "shape": (8,),
        "global_shape": (8,),
        "shard_offsets": (0,),
        "dtype": "torch.float32",
    }
    values.update(kwargs)
    return TensorShardSpec(**values)


def _target_spec(**kwargs):
    values = {
        "name": "w",
        "receiver_rank": 0,
        "shape": (2,),
        "global_shape": (8,),
        "shard_offsets": (4,),
        "dtype": "torch.float32",
    }
    values.update(kwargs)
    return TensorReceiveSpec(**values)


def test_slice_transfer_manifest_offsets_remote_descriptor_for_partial_target():
    source = _source_spec()
    target = _target_spec()
    plan = plan_dense_reshard_transfers([source], [target])
    target_tensor = torch.zeros((2,), dtype=torch.float32)

    manifest = build_slice_transfer_manifest(
        plan,
        source_descriptors=[
            TensorDescriptor(
                name="w",
                addr=100,
                size=32,
                device_id=0,
                dtype="torch.float32",
            )
        ],
        target_tensors={"w": target_tensor},
    )

    assert list(manifest.target_tensors) == ["w"]
    assert manifest.target_tensors["w"].data_ptr() == target_tensor.data_ptr()
    assert manifest.source_descriptors == [
        TensorDescriptor(
            name="w",
            addr=116,
            size=8,
            device_id=0,
            dtype="torch.float32",
        )
    ]
    assert manifest.output_tensors == [("w", target_tensor)]


def test_slice_transfer_manifest_rejects_noncontiguous_target_slice():
    source = TensorShardSpec(
        name="w",
        worker_rank=0,
        shape=(4, 4),
        global_shape=(4, 4),
        shard_offsets=(0, 0),
        dtype="torch.float32",
    )
    target = TensorReceiveSpec(
        name="w",
        receiver_rank=0,
        shape=(4, 2),
        global_shape=(4, 4),
        shard_offsets=(0, 1),
        dtype="torch.float32",
    )
    plan = TransferPlan(
        (
            TransferPlanEntry(
                tensor_name="w",
                source_worker_rank=0,
                receiver_rank=0,
                source=source,
                target=target,
                source_slice=TensorSlice((0, 1), (4, 2)),
                target_slice=TensorSlice((0, 0), (4, 2)),
            ),
        ),
        (),
    )

    with pytest.raises(RuntimeError, match="source slice.*not contiguous"):
        build_slice_transfer_manifest(
            plan,
            source_descriptors=[TensorDescriptor("w", 100, 64, 0, "torch.float32")],
            target_tensors={"w": torch.zeros((4, 2), dtype=torch.float32)},
        )


def test_slice_transfer_manifest_rejects_multi_source_tensor_plan():
    source_a = _source_spec(shape=(2,), global_shape=(4,), shard_offsets=(0,))
    source_b = _source_spec(worker_rank=1, shape=(2,), global_shape=(4,), shard_offsets=(2,))
    target = _target_spec(shape=(4,), global_shape=(4,), shard_offsets=(0,))
    plan = plan_dense_reshard_transfers([source_a, source_b], [target])

    with pytest.raises(RuntimeError, match="multi-source tensors are not supported"):
        build_slice_transfer_manifest(
            plan,
            source_descriptors=[TensorDescriptor("w", 100, 8, 0, "torch.float32")],
            target_tensors={"w": torch.zeros((4,), dtype=torch.float32)},
        )


def test_grouped_slice_transfer_manifest_materializes_multi_source_tensor_plan():
    source_a = _source_spec(shape=(2,), global_shape=(4,), shard_offsets=(0,))
    source_b = _source_spec(worker_rank=1, shape=(2,), global_shape=(4,), shard_offsets=(2,))
    target = _target_spec(shape=(4,), global_shape=(4,), shard_offsets=(0,))
    plan = plan_dense_reshard_transfers([source_a, source_b], [target])
    target_tensor = torch.zeros((4,), dtype=torch.float32)

    manifest = build_grouped_slice_transfer_manifest(
        plan,
        source_descriptors_by_rank={
            0: [TensorDescriptor("w", 100, 8, 0, "torch.float32")],
            1: [TensorDescriptor("w", 200, 8, 0, "torch.float32")],
        },
        target_tensors={"w": target_tensor},
    )

    assert list(manifest.target_tensors) == ["w.__mx_slice_0", "w.__mx_slice_1"]
    assert [transfer.source_worker_rank for transfer in manifest.source_transfers] == [0, 1]
    assert manifest.source_transfers[0].source_descriptors == [
        TensorDescriptor("w.__mx_slice_0", 100, 8, 0, "torch.float32"),
    ]
    assert manifest.source_transfers[1].source_descriptors == [
        TensorDescriptor("w.__mx_slice_1", 200, 8, 0, "torch.float32"),
    ]
    assert manifest.output_tensors == [("w", target_tensor)]


def test_receive_into_tensors_applies_dense_slice_plan(monkeypatch):
    class _FakeNixlManager:
        registered_tensor_names = []
        received_descriptors = []

        def __init__(self, *args, **kwargs):
            pass

        def initialize(self):
            pass

        def register_tensors(self, tensors):
            type(self).registered_tensor_names = list(tensors)

        def receive_from_source(self, source_metadata, source_tensors, timeout_seconds):
            del source_metadata
            del timeout_seconds
            type(self).received_descriptors = source_tensors
            return sum(descriptor.size for descriptor in source_tensors), len(source_tensors), 0.0

    class _MxClient:
        def list_sources(self, identity=None, status_filter=None):
            del identity
            del status_filter
            return p2p_pb2.ListSourcesResponse(
                instances=[
                    _source_ref(
                        {
                            "w": {
                                "shape": [8],
                                "global_shape": [8],
                                "shard_offsets": [0],
                                "dtype": "torch.float32",
                            }
                        }
                    )
                ],
            )

        def get_metadata(self, mx_source_id, worker_id):
            del mx_source_id
            del worker_id
            return p2p_pb2.GetMetadataResponse(
                found=True,
                worker=p2p_pb2.WorkerMetadata(
                    worker_rank=0,
                    nixl_metadata=b"source",
                    tensors=[
                        p2p_pb2.TensorDescriptor(
                            name="w",
                            addr=100,
                            size=32,
                            device_id=0,
                            dtype="torch.float32",
                        )
                    ],
                ),
            )

    monkeypatch.setattr(rl_transfer_module, "NixlTransferManager", _FakeNixlManager)
    transfer = RlNixlWeightTransfer(
        mx_client=_MxClient(),
        base_identity=_base_identity(),
        worker_id="worker-local",
    )
    target = torch.zeros((2,), dtype=torch.float32)

    tensors = asyncio.run(
        transfer.receive_into_tensors(
            {"w": target},
            model_version=5,
            receiver_rank=0,
            target_specs=[
                TensorReceiveSpec(
                    name="w",
                    receiver_rank=0,
                    shape=(2,),
                    global_shape=(8,),
                    shard_offsets=(4,),
                    dtype="torch.float32",
                )
            ],
        )
    )

    assert tensors == [("w", target)]
    assert _FakeNixlManager.registered_tensor_names == ["w"]
    assert _FakeNixlManager.received_descriptors == [
        TensorDescriptor("w", 116, 8, 0, "torch.float32"),
    ]


def test_receive_into_tensors_applies_multi_source_dense_fanin(monkeypatch):
    class _FakeNixlManager:
        registered_tensor_names = []
        receive_calls = []

        def __init__(self, *args, **kwargs):
            pass

        def initialize(self):
            pass

        def register_tensors(self, tensors):
            type(self).registered_tensor_names = list(tensors)

        def receive_from_source(self, source_metadata, source_tensors, timeout_seconds):
            del timeout_seconds
            type(self).receive_calls.append((source_metadata, list(source_tensors)))
            return sum(descriptor.size for descriptor in source_tensors), len(source_tensors), 0.0

        def shutdown(self):
            pass

    class _MxClient:
        def list_sources(self, identity=None, status_filter=None):
            del identity
            del status_filter
            return p2p_pb2.ListSourcesResponse(
                instances=[
                    _source_ref(
                        {
                            "w": {
                                "shape": [2],
                                "global_shape": [4],
                                "shard_offsets": [0],
                                "dtype": "torch.float32",
                            }
                        },
                        mx_source_id="source-r0",
                        worker_id="worker-r0",
                        worker_rank=0,
                        source_world_size=2,
                    ),
                    _source_ref(
                        {
                            "w": {
                                "shape": [2],
                                "global_shape": [4],
                                "shard_offsets": [2],
                                "dtype": "torch.float32",
                            }
                        },
                        mx_source_id="source-r1",
                        worker_id="worker-r1",
                        worker_rank=1,
                        source_world_size=2,
                    ),
                ],
            )

        def get_metadata(self, mx_source_id, worker_id):
            del mx_source_id
            rank_by_worker = {"worker-r0": 0, "worker-r1": 1}
            addr_by_worker = {"worker-r0": 100, "worker-r1": 200}
            rank = rank_by_worker[worker_id]
            return p2p_pb2.GetMetadataResponse(
                found=True,
                worker=p2p_pb2.WorkerMetadata(
                    worker_rank=rank,
                    nixl_metadata=f"source-r{rank}".encode(),
                    tensors=[
                        p2p_pb2.TensorDescriptor(
                            name="w",
                            addr=addr_by_worker[worker_id],
                            size=8,
                            device_id=0,
                            dtype="torch.float32",
                        )
                    ],
                ),
            )

    monkeypatch.setattr(rl_transfer_module, "NixlTransferManager", _FakeNixlManager)
    transfer = RlNixlWeightTransfer(
        mx_client=_MxClient(),
        base_identity=_base_identity(),
        worker_id="worker-local",
    )
    target = torch.zeros((4,), dtype=torch.float32)

    tensors = asyncio.run(
        transfer.receive_into_tensors(
            {"w": target},
            model_version=5,
            receiver_rank=0,
            target_specs=[
                TensorReceiveSpec(
                    name="w",
                    receiver_rank=0,
                    shape=(4,),
                    global_shape=(4,),
                    shard_offsets=(0,),
                    dtype="torch.float32",
                )
            ],
        )
    )

    assert tensors == [("w", target)]
    assert _FakeNixlManager.registered_tensor_names == [
        "w.__mx_slice_0",
        "w.__mx_slice_1",
    ]
    assert _FakeNixlManager.receive_calls == [
        (
            b"source-r0",
            [TensorDescriptor("w.__mx_slice_0", 100, 8, 0, "torch.float32")],
        ),
        (
            b"source-r1",
            [TensorDescriptor("w.__mx_slice_1", 200, 8, 0, "torch.float32")],
        ),
    ]
    assert transfer.last_receive_report is not None
    assert [attempt.success for attempt in transfer.last_receive_report.attempts] == [
        False,
        False,
        True,
        True,
    ]
    assert transfer.last_receive_report.retry_count == 2
    assert transfer.last_receive_report.source_worker_id == "worker-r0"
