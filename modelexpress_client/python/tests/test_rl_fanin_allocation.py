# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import torch

import modelexpress.rl_fanin_transfer as rl_fanin_transfer_module
from modelexpress import p2p_pb2
from modelexpress.rl_fanin_transfer import (
    infer_dense_fanin_receive_specs,
    prepare_dense_fanin_receive,
    preferred_dense_fanin_groups,
)
from modelexpress.rl_metadata import (
    RlSourceMetadata,
    RlSourceRole,
    candidates_from_response,
    with_rl_source_metadata,
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
    mx_source_id,
    worker_id,
    worker_rank,
):
    identity = with_rl_source_metadata(
        _base_identity(),
        RlSourceMetadata(
            model_version=5,
            role=RlSourceRole.TRAINER,
            world_size=2,
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


def _partial_source_response():
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
            ),
        ],
    )


class _MxClient:
    def list_sources(self, identity=None, status_filter=None):
        del identity
        del status_filter
        return _partial_source_response()

    def get_metadata(self, mx_source_id, worker_id):
        del mx_source_id
        rank = {"worker-r0": 0, "worker-r1": 1}[worker_id]
        addr = {0: 100, 1: 200}[rank]
        return p2p_pb2.GetMetadataResponse(
            found=True,
            worker=p2p_pb2.WorkerMetadata(
                worker_rank=rank,
                nixl_metadata=f"source-r{rank}".encode(),
                tensors=[
                    p2p_pb2.TensorDescriptor(
                        name="w",
                        addr=addr,
                        size=8,
                        device_id=0,
                        dtype="torch.float32",
                    )
                ],
            ),
        )


def test_preferred_dense_fanin_groups_infers_allocated_receive_specs():
    candidates = candidates_from_response(_partial_source_response())

    specs = infer_dense_fanin_receive_specs(candidates, receiver_rank=0)
    preferred_groups = preferred_dense_fanin_groups(
        candidates,
        target_tensors=None,
        target_specs=None,
        receiver_rank=0,
        same_rank_only=False,
    )

    assert [(spec.name, spec.shape, spec.global_shape, spec.shard_offsets) for spec in specs] == [
        ("w", (4,), (4,), (0,))
    ]
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
                    worker_rank=0,
                ),
                *_partial_source_response().instances,
            ]
        )
    )
    assert (
        preferred_dense_fanin_groups(
            full_candidates,
            target_tensors=None,
            target_specs=None,
            receiver_rank=0,
            same_rank_only=False,
        )
        == ()
    )


def test_prepare_dense_fanin_receive_allocates_missing_target_tensors(monkeypatch):
    allocated = {}

    def fake_allocate(specs, *, device):
        allocated["device"] = device
        allocated["specs"] = tuple(specs)
        return {"w": torch.zeros((4,), dtype=torch.float32)}

    monkeypatch.setattr(
        rl_fanin_transfer_module,
        "allocate_tensors_from_receive_specs",
        fake_allocate,
    )
    candidates = candidates_from_response(_partial_source_response())

    plan = prepare_dense_fanin_receive(
        mx_client=_MxClient(),
        candidates=candidates,
        target_tensors=None,
        target_specs=None,
        receiver_rank=0,
        same_rank_only=False,
        target_device="cuda:0",
    )

    assert allocated["device"] == "cuda:0"
    assert [(spec.name, spec.shape) for spec in allocated["specs"]] == [("w", (4,))]
    assert plan.manifest.output_tensors[0][0] == "w"
    assert sorted(plan.manifest.target_tensors) == ["w.__mx_slice_0", "w.__mx_slice_1"]
    assert [
        (source.candidate.worker_id, source.source_descriptors)
        for source in plan.source_transfers
    ] == [
        ("worker-r0", [TensorDescriptor("w.__mx_slice_0", 100, 8, 0, "torch.float32")]),
        ("worker-r1", [TensorDescriptor("w.__mx_slice_1", 200, 8, 0, "torch.float32")]),
    ]


def test_receive_tensors_uses_allocated_dense_fanin(monkeypatch):
    class _FakeNixlManager:
        registered_tensor_names = []
        receive_calls = []

        def __init__(self, *args, **kwargs):
            pass

        def initialize(self):
            pass

        def register_tensors(self, tensors):
            type(self).registered_tensor_names = sorted(tensors)

        def receive_from_source(self, source_metadata, source_tensors, timeout_seconds):
            del timeout_seconds
            type(self).receive_calls.append((source_metadata, list(source_tensors)))
            return sum(descriptor.size for descriptor in source_tensors), len(source_tensors), 0.0

        def shutdown(self):
            pass

    def fake_allocate(specs, *, device):
        assert device == "cuda:0"
        assert [(spec.name, spec.shape) for spec in specs] == [("w", (4,))]
        return {"w": torch.zeros((4,), dtype=torch.float32)}

    monkeypatch.setattr(rl_fanin_transfer_module, "allocate_tensors_from_receive_specs", fake_allocate)
    monkeypatch.setattr("modelexpress.rl_transfer.NixlTransferManager", _FakeNixlManager)
    transfer = RlNixlWeightTransfer(
        mx_client=_MxClient(),
        base_identity=_base_identity(),
        worker_id="worker-local",
        device_id=0,
    )

    result = asyncio.run(transfer.receive_tensors(model_version=5, receiver_rank=0))

    assert result[0][0] == "w"
    assert _FakeNixlManager.registered_tensor_names == ["w.__mx_slice_0", "w.__mx_slice_1"]
    assert _FakeNixlManager.receive_calls == [
        (b"source-r0", [TensorDescriptor("w.__mx_slice_0", 100, 8, 0, "torch.float32")]),
        (b"source-r1", [TensorDescriptor("w.__mx_slice_1", 200, 8, 0, "torch.float32")]),
    ]
    assert transfer.last_receive_report is not None
    assert transfer.last_receive_report.retry_count == 0
