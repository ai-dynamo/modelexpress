# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
import torch

import modelexpress.rl_transfer as rl_transfer_module
from modelexpress import p2p_pb2
from modelexpress.rl_fanin_transfer import preferred_dense_fanin_groups
from modelexpress.rl_metadata import (
    RlSourceMetadata,
    RlSourceRole,
    candidates_from_response,
    with_rl_source_metadata,
)
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


def test_slice_transfer_manifest_fragments_noncontiguous_source_slice():
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

    manifest = build_slice_transfer_manifest(
        plan,
        source_descriptors=[TensorDescriptor("w", 100, 64, 0, "torch.float32")],
        target_tensors={"w": torch.zeros((4, 2), dtype=torch.float32)},
    )

    assert list(manifest.target_tensors) == [
        "w.__mx_fragment_0",
        "w.__mx_fragment_1",
        "w.__mx_fragment_2",
        "w.__mx_fragment_3",
    ]
    assert manifest.source_descriptors == [
        TensorDescriptor("w.__mx_fragment_0", 104, 8, 0, "torch.float32"),
        TensorDescriptor("w.__mx_fragment_1", 120, 8, 0, "torch.float32"),
        TensorDescriptor("w.__mx_fragment_2", 136, 8, 0, "torch.float32"),
        TensorDescriptor("w.__mx_fragment_3", 152, 8, 0, "torch.float32"),
    ]
    assert all(tensor.is_contiguous() for tensor in manifest.target_tensors.values())


def test_slice_transfer_manifest_allows_same_source_moe_expert_slices():
    source = _source_spec(
        name="experts.w",
        shape=(4, 2),
        global_shape=(4, 2),
        shard_offsets=(0, 0),
        expert_ids=(0, 1, 2, 3),
        expert_axis=0,
    )
    target = _target_spec(
        name="experts.w",
        shape=(2, 2),
        global_shape=(4, 2),
        shard_offsets=(0, 0),
        expert_ids=(1, 3),
        expert_axis=0,
    )
    plan = plan_dense_reshard_transfers([source], [target])

    manifest = build_slice_transfer_manifest(
        plan,
        source_descriptors=[TensorDescriptor("experts.w", 100, 32, 0, "torch.float32")],
        target_tensors={"experts.w": torch.zeros((2, 2), dtype=torch.float32)},
    )

    assert list(manifest.target_tensors) == [
        "experts.w.__mx_slice_0",
        "experts.w.__mx_slice_1",
    ]
    assert manifest.source_descriptors == [
        TensorDescriptor("experts.w.__mx_slice_0", 108, 8, 0, "torch.float32"),
        TensorDescriptor("experts.w.__mx_slice_1", 124, 8, 0, "torch.float32"),
    ]


def test_slice_transfer_manifest_stages_noncontiguous_target_view():
    source = _source_spec(shape=(4, 2), global_shape=(4, 4), shard_offsets=(0, 1))
    target = _target_spec(shape=(4, 2), global_shape=(4, 4), shard_offsets=(0, 1))
    plan = plan_dense_reshard_transfers([source], [target])
    backing_tensor = torch.zeros((4, 4), dtype=torch.float32)
    target_view = backing_tensor[:, 1:3]

    manifest = build_slice_transfer_manifest(
        plan,
        source_descriptors=[TensorDescriptor("w", 100, 32, 0, "torch.float32")],
        target_tensors={"w": target_view},
    )
    transfer_target = manifest.target_tensors["w"]
    transfer_target.copy_(
        torch.arange(8, dtype=torch.float32).reshape(4, 2)
    )
    manifest.finalize()

    assert transfer_target.is_contiguous()
    assert transfer_target.data_ptr() != target_view.data_ptr()
    assert manifest.source_descriptors == [
        TensorDescriptor("w", 100, 32, 0, "torch.float32"),
    ]
    assert manifest.output_tensors == [("w", target_view)]
    assert torch.equal(
        backing_tensor[:, 1:3],
        torch.arange(8, dtype=torch.float32).reshape(4, 2),
    )
    assert torch.equal(backing_tensor[:, 0], torch.zeros(4))
    assert torch.equal(backing_tensor[:, 3], torch.zeros(4))


def test_slice_transfer_manifest_rejects_target_tensor_out_of_bounds():
    source = _source_spec(shape=(4, 2), global_shape=(4, 2), shard_offsets=(0, 0))
    target = _target_spec(shape=(4, 2), global_shape=(4, 2), shard_offsets=(0, 0))
    plan = plan_dense_reshard_transfers([source], [target])

    with pytest.raises(RuntimeError, match="target slice.*exceeds target tensor bounds"):
        build_slice_transfer_manifest(
            plan,
            source_descriptors=[TensorDescriptor("w", 100, 32, 0, "torch.float32")],
            target_tensors={"w": torch.zeros((4, 1), dtype=torch.float32)},
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


def test_grouped_slice_transfer_manifest_stages_noncontiguous_target_slices():
    source_a = _source_spec(
        shape=(4, 2),
        global_shape=(4, 4),
        shard_offsets=(0, 0),
    )
    source_b = _source_spec(
        worker_rank=1,
        shape=(4, 2),
        global_shape=(4, 4),
        shard_offsets=(0, 2),
    )
    target = _target_spec(shape=(4, 4), global_shape=(4, 4), shard_offsets=(0, 0))
    plan = plan_dense_reshard_transfers([source_a, source_b], [target])
    target_tensor = torch.zeros((4, 4), dtype=torch.float32)

    manifest = build_grouped_slice_transfer_manifest(
        plan,
        source_descriptors_by_rank={
            0: [TensorDescriptor("w", 100, 32, 0, "torch.float32")],
            1: [TensorDescriptor("w", 200, 32, 0, "torch.float32")],
        },
        target_tensors={"w": target_tensor},
    )
    manifest.target_tensors["w.__mx_slice_0"].fill_(1.0)
    manifest.target_tensors["w.__mx_slice_1"].fill_(2.0)
    manifest.finalize()

    assert all(tensor.is_contiguous() for tensor in manifest.target_tensors.values())
    assert manifest.source_transfers[0].source_descriptors == [
        TensorDescriptor("w.__mx_slice_0", 100, 32, 0, "torch.float32"),
    ]
    assert manifest.source_transfers[1].source_descriptors == [
        TensorDescriptor("w.__mx_slice_1", 200, 32, 0, "torch.float32"),
    ]
    assert torch.equal(target_tensor[:, :2], torch.ones((4, 2), dtype=torch.float32))
    assert torch.equal(target_tensor[:, 2:], torch.full((4, 2), 2.0))


def test_grouped_slice_transfer_manifest_fragments_noncontiguous_source_slices():
    source_a = _source_spec(
        shape=(4, 4),
        global_shape=(4, 4),
        shard_offsets=(0, 0),
    )
    source_b = _source_spec(
        worker_rank=1,
        shape=(4, 4),
        global_shape=(4, 4),
        shard_offsets=(0, 0),
    )
    target_a = _target_spec(
        name="w.left",
        shape=(4, 2),
        global_shape=(4, 4),
        shard_offsets=(0, 0),
    )
    target_b = _target_spec(
        name="w.right",
        shape=(4, 2),
        global_shape=(4, 4),
        shard_offsets=(0, 2),
    )
    plan = TransferPlan(
        (
            TransferPlanEntry(
                tensor_name="w.left",
                source_worker_rank=0,
                receiver_rank=0,
                source=source_a,
                target=target_a,
                source_slice=TensorSlice((0, 0), (4, 2)),
                target_slice=TensorSlice((0, 0), (4, 2)),
            ),
            TransferPlanEntry(
                tensor_name="w.right",
                source_worker_rank=1,
                receiver_rank=0,
                source=source_b,
                target=target_b,
                source_slice=TensorSlice((0, 2), (4, 2)),
                target_slice=TensorSlice((0, 0), (4, 2)),
            ),
        ),
        (),
    )

    manifest = build_grouped_slice_transfer_manifest(
        plan,
        source_descriptors_by_rank={
            0: [TensorDescriptor("w", 100, 64, 0, "torch.float32")],
            1: [TensorDescriptor("w", 200, 64, 0, "torch.float32")],
        },
        target_tensors={
            "w.left": torch.zeros((4, 2), dtype=torch.float32),
            "w.right": torch.zeros((4, 2), dtype=torch.float32),
        },
    )

    assert [transfer.source_worker_rank for transfer in manifest.source_transfers] == [0, 1]
    assert manifest.source_transfers[0].source_descriptors == [
        TensorDescriptor("w.left.__mx_fragment_0", 100, 8, 0, "torch.float32"),
        TensorDescriptor("w.left.__mx_fragment_1", 116, 8, 0, "torch.float32"),
        TensorDescriptor("w.left.__mx_fragment_2", 132, 8, 0, "torch.float32"),
        TensorDescriptor("w.left.__mx_fragment_3", 148, 8, 0, "torch.float32"),
    ]
    assert manifest.source_transfers[1].source_descriptors == [
        TensorDescriptor("w.right.__mx_fragment_0", 208, 8, 0, "torch.float32"),
        TensorDescriptor("w.right.__mx_fragment_1", 224, 8, 0, "torch.float32"),
        TensorDescriptor("w.right.__mx_fragment_2", 240, 8, 0, "torch.float32"),
        TensorDescriptor("w.right.__mx_fragment_3", 256, 8, 0, "torch.float32"),
    ]


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
        def __init__(self):
            self.metadata_calls = []

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
            self.metadata_calls.append((mx_source_id, worker_id))
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
    mx_client = _MxClient()
    transfer = RlNixlWeightTransfer(
        mx_client=mx_client,
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
        True,
        True,
    ]
    assert transfer.last_receive_report.retry_count == 0
    assert transfer.last_receive_report.source_worker_id == "worker-r0"
    assert mx_client.metadata_calls == [
        ("source-r0", "worker-r0"),
        ("source-r1", "worker-r1"),
    ]


def test_dense_fanin_preflight_preserves_complete_single_source_preference():
    target = {"w": torch.zeros((4,), dtype=torch.float32)}
    target_specs = [
        TensorReceiveSpec(
            name="w",
            receiver_rank=0,
            shape=(4,),
            global_shape=(4,),
            shard_offsets=(0,),
            dtype="torch.float32",
        )
    ]
    partial_refs = [
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
    ]

    partial_candidates = candidates_from_response(
        p2p_pb2.ListSourcesResponse(instances=partial_refs)
    )
    preferred_groups = preferred_dense_fanin_groups(
        partial_candidates,
        target_tensors=target,
        target_specs=target_specs,
        receiver_rank=0,
        same_rank_only=False,
    )

    assert [[candidate.worker_id for candidate in group] for group in preferred_groups] == [
        ["worker-r0", "worker-r1"]
    ]

    full_candidates = candidates_from_response(
        p2p_pb2.ListSourcesResponse(
            instances=[
                _source_ref(
                    {"w": {"shape": [4], "dtype": "torch.float32"}},
                    mx_source_id="source-full",
                    worker_id="worker-full",
                ),
                *partial_refs,
            ]
        )
    )

    assert preferred_dense_fanin_groups(
        full_candidates,
        target_tensors=target,
        target_specs=target_specs,
        receiver_rank=0,
        same_rank_only=False,
    ) == ()


def test_receive_into_tensors_scatters_into_noncontiguous_target_view(monkeypatch):
    class _FakeNixlManager:
        registered_tensors = {}
        received_descriptors = []

        def __init__(self, *args, **kwargs):
            pass

        def initialize(self):
            pass

        def register_tensors(self, tensors):
            type(self).registered_tensors = dict(tensors)

        def receive_from_source(self, source_metadata, source_tensors, timeout_seconds):
            del source_metadata
            del timeout_seconds
            type(self).received_descriptors = source_tensors
            type(self).registered_tensors["w"].copy_(
                torch.arange(8, dtype=torch.float32).reshape(4, 2)
            )
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
                                "shape": [4, 2],
                                "global_shape": [4, 4],
                                "shard_offsets": [0, 1],
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
    backing_tensor = torch.zeros((4, 4), dtype=torch.float32)
    target_view = backing_tensor[:, 1:3]

    tensors = asyncio.run(
        transfer.receive_into_tensors(
            {"w": target_view},
            model_version=5,
            receiver_rank=0,
            target_specs=[
                TensorReceiveSpec(
                    name="w",
                    receiver_rank=0,
                    shape=(4, 2),
                    global_shape=(4, 4),
                    shard_offsets=(0, 1),
                    dtype="torch.float32",
                )
            ],
        )
    )

    assert tensors == [("w", target_view)]
    assert _FakeNixlManager.registered_tensors["w"].is_contiguous()
    assert _FakeNixlManager.registered_tensors["w"].data_ptr() != target_view.data_ptr()
    assert _FakeNixlManager.received_descriptors == [
        TensorDescriptor("w", 100, 32, 0, "torch.float32"),
    ]
    assert torch.equal(
        backing_tensor[:, 1:3],
        torch.arange(8, dtype=torch.float32).reshape(4, 2),
    )
    assert torch.equal(backing_tensor[:, 0], torch.zeros(4))
    assert torch.equal(backing_tensor[:, 3], torch.zeros(4))


def test_receive_into_tensors_gathers_noncontiguous_source_fragments(monkeypatch):
    class _FakeNixlManager:
        registered_tensors = {}
        received_descriptors = []

        def __init__(self, *args, **kwargs):
            pass

        def initialize(self):
            pass

        def register_tensors(self, tensors):
            type(self).registered_tensors = dict(tensors)

        def receive_from_source(self, source_metadata, source_tensors, timeout_seconds):
            del source_metadata
            del timeout_seconds
            type(self).received_descriptors = list(source_tensors)
            for row, descriptor in enumerate(source_tensors):
                type(self).registered_tensors[descriptor.name].fill_(float(row + 1))
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
                                "shape": [4, 4],
                                "global_shape": [4, 4],
                                "shard_offsets": [0, 0],
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
                            size=64,
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
    target = torch.zeros((4, 2), dtype=torch.float32)

    tensors = asyncio.run(
        transfer.receive_into_tensors(
            {"w": target},
            model_version=5,
            receiver_rank=0,
            target_specs=[
                TensorReceiveSpec(
                    name="w",
                    receiver_rank=0,
                    shape=(4, 2),
                    global_shape=(4, 4),
                    shard_offsets=(0, 1),
                    dtype="torch.float32",
                )
            ],
        )
    )

    assert tensors == [("w", target)]
    assert _FakeNixlManager.received_descriptors == [
        TensorDescriptor("w.__mx_fragment_0", 104, 8, 0, "torch.float32"),
        TensorDescriptor("w.__mx_fragment_1", 120, 8, 0, "torch.float32"),
        TensorDescriptor("w.__mx_fragment_2", 136, 8, 0, "torch.float32"),
        TensorDescriptor("w.__mx_fragment_3", 152, 8, 0, "torch.float32"),
    ]
    assert torch.equal(
        target,
        torch.tensor(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
            ],
            dtype=torch.float32,
        ),
    )
