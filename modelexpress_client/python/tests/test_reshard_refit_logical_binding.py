# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical contract tests for logical planning and per-replica binding."""

from modelexpress.refit.reshard.rendezvous import structural_manifest_digest
import ctypes

import pytest
import torch

from modelexpress.refit.reshard.slice_plan import Shard
from modelexpress.refit.reshard.transfer_plan import (
    SourceInfo,
    bind_transfer,
    plan_transfer,
    plan_transfer_groups,
)
from modelexpress.refit.reshard.types import CaptureResult, RecordedCopy
from tests.test_refit_bounded_descriptors import harness


def _source(tensor, session="physical-0"):
    tensor = tensor.contiguous()
    shard = Shard(
        shard_offset=(0,) * tensor.ndim,
        shape=tuple(tensor.shape),
        session=session,
        addr=tensor.data_ptr(),
        elsize=tensor.element_size(),
    )
    return SourceInfo(
        global_shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        elsize=tensor.element_size(),
        shards=[shard],
    )


def _apply(descriptors):
    for d in descriptors:
        if d.nbytes:
            ctypes.memmove(d.dst_addr, d.src_addr, d.nbytes)


@pytest.mark.parametrize("shared_storage", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_binding_keeps_multiple_sources_attached_to_their_logical_slots(shared_storage, grouped):
    if shared_storage:
        backing = torch.tensor([2, 4, 6, 8, 11, 13, 17, 19], dtype=torch.float32)
        left, right = backing[:4].reshape(2, 2), backing[4:].reshape(2, 2)
        left_session = right_session = "physical-shared"
        left_slot = right_slot = "logical-slot-shared"
    else:
        backing = None
        left = torch.tensor([[2, 4], [6, 8]], dtype=torch.float32)
        right = torch.tensor([[11, 13], [17, 19]], dtype=torch.float32)
        left_session, right_session = "physical-rank-1", "physical-rank-0"
        left_slot, right_slot = "logical-slot-a", "logical-slot-b"
    capture = CaptureResult(copies=[
        RecordedCopy("left", (), "left_param", 0, (2, 2), (2, 1), torch.float32),
        RecordedCopy("right", (), "right_param", 0, (2, 2), (2, 1), torch.float32),
    ])
    published = {"left": _source(left, left_session),
                 "right": _source(right, right_session)}
    logical = plan_transfer(
        capture, published,
        source_slots={left_session: left_slot, right_session: right_slot},
    )
    plans = tuple(plan_transfer_groups(
        (CaptureResult(copies=[copy]) for copy in capture.copies), published, logical,
    )) if grouped else (logical,)
    for offset in (0, 100):
        if shared_storage:
            storage = backing + offset
            replicas = {"left": storage[:4].view(2, 2), "right": storage[4:].view(2, 2)}
        else:
            replicas = {"left": left + offset, "right": right + offset}
        sources = {
            (left_slot, "left"): (f"replica-{offset}-{left_session}", replicas["left"].data_ptr()),
            (right_slot, "right"): (f"replica-{offset}-{right_session}", replicas["right"].data_ptr()),
        }
        outputs = {name + "_param": torch.zeros_like(value) for name, value in replicas.items()}
        source_calls, destination_calls = [], []

        def resolve_source(buffer):
            key = buffer.source_slot_id, buffer.src_name
            source_calls.append(key)
            return sources[key]

        def resolve_destination(kind, name):
            destination_calls.append((kind, name))
            return outputs[name].data_ptr()

        sessions = set()
        for plan in plans:
            physical = bind_transfer(plan, resolve_source, resolve_destination)
            sessions.update(descriptor.session for descriptor in physical.descriptors)
            _apply(physical.descriptors)
        assert sessions == {session for session, _ in sources.values()}
        assert sorted(source_calls) == sorted(sources)
        assert sorted(destination_calls) == [("recv", "left_param"), ("recv", "right_param")]
        for name, value in replicas.items():
            assert torch.equal(outputs[name + "_param"], value)


def test_binding_failure_does_not_return_a_partial_descriptor_plan():
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    capture = CaptureResult(copies=[RecordedCopy(
        "weight", (), "param", 0, (2, 3), (3, 1), torch.float32
    )])
    logical = plan_transfer(
        capture, {"weight": _source(source)},
        source_slots={"physical-0": "slot-0"},
    )
    calls = []

    def resolve_destination(kind, name):
        calls.append((kind, name))
        raise RuntimeError("destination is unavailable")

    with pytest.raises(RuntimeError, match="destination is unavailable"):
        bind_transfer(
            logical,
            lambda _buffer: ("session", source.data_ptr()),
            resolve_destination,
        )
    assert calls == [("recv", "param")]


@pytest.mark.parametrize("keyword", [False, True])
def test_recorded_operation_arguments_are_snapshotted_before_binding(keyword):
    from modelexpress.refit.reshard.geometry import capture_geometry

    dims = [1, 0]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.empty(3, 2, device="meta"))
            self.param.weight_loader = self.load

        def load(self, param, loaded):
            view = loaded.permute(dims=dims) if keyword else loaded.permute(dims)
            dims[:] = [0, 1]
            param.data.copy_(view)

        def load_weights(self, weights):
            for _name, loaded in weights:
                self.param.weight_loader(self.param, loaded)

    capture = capture_geometry(Model(), [("weight", torch.float32, (2, 3))])
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    logical = plan_transfer(
        capture, {"weight": _source(source)}, source_slots={"physical-0": "slot-0"},
        max_segments_per_copy=1,
    )

    destination = torch.zeros(3, 2, dtype=torch.float32)
    staging = torch.empty_like(source)
    physical = bind_transfer(
        logical,
        lambda _buffer: ("session", source.data_ptr()),
        lambda kind, _name: staging.data_ptr()
        if kind == "full" else destination.data_ptr(),
    )
    _apply(physical.full_pulls[0])
    frozen_copy = logical.full_pulls[0].copies[0]
    replayed = staging
    for op, args, kwargs in frozen_copy.op_chain:
        replayed = getattr(replayed, op)(*args, **dict(kwargs))
    destination.copy_(replayed)
    assert torch.equal(destination, source.permute(1, 0))


@pytest.mark.parametrize("grouped", [False, True])
def test_full_pull_replay_preserves_source_order_after_capture_reordering(grouped):
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    b = torch.tensor([[11, 12], [13, 14]], dtype=torch.float32)

    def copy(source, op_chain):
        return RecordedCopy(
            src_name=source, op_chain=op_chain, param_name="param", dest_offset=0,
            dest_shape=(2, 2), dest_stride=(2, 1), dest_dtype=torch.float32,
        )

    capture = CaptureResult(copies=[
        copy("b", (("transpose", (0, 1), ()),)),
        copy("a", ()),
        copy("a", (("transpose", (0, 1), ()),)),
    ])
    sources = {"a": _source(a), "b": _source(b, "physical-1")}
    logical = plan_transfer(
        capture, sources,
        source_slots={"physical-0": "slot-a", "physical-1": "slot-b"},
        max_segments_per_copy=1,
        verify_full_sources=True,
    )
    if grouped:
        logical = next(plan_transfer_groups(
            (capture,), sources, logical, max_segments_per_copy=1,
        ))

    staging = {"a": torch.empty_like(a), "b": torch.empty_like(b)}
    destination = torch.zeros_like(a)
    physical = bind_transfer(
        logical,
        lambda buffer: (
            sources[buffer.src_name].shards[0].session,
            sources[buffer.src_name].shards[0].addr,
        ),
        lambda kind, name: staging[name].data_ptr()
        if kind == "full" else destination.data_ptr(),
    )
    for full, descriptors in zip(logical.full_pulls, physical.full_pulls, strict=True):
        _apply(descriptors)
        for recorded in full.copies:
            replayed = staging[full.src_name]
            for op, args, kwargs in recorded.op_chain:
                replayed = getattr(replayed, op)(*args, **dict(kwargs))
            destination.as_strided(
                recorded.dest_shape, recorded.dest_stride,
                destination.storage_offset() + recorded.dest_offset,
            ).copy_(replayed)
    assert torch.equal(destination, b.transpose(0, 1))


def test_staged_prepare_binds_split_sources_from_reordered_manifests(harness):
    from modelexpress.refit.reshard.rendezvous import (
        PublishedShard, PublishedTensor, structural_manifest_digest,
        wrap_rendezvous_blob,
    )

    first = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    second = torch.arange(100, 108, dtype=torch.float32).reshape(2, 4)
    manifests = []
    for agent, tensor, offset in (
        ("agent-z", first, (0, 0)),
        ("agent-a", second, (2, 0)),
    ):
        shard = PublishedShard(
            agent_name=agent, device_id=0, addr=tensor.data_ptr(),
            shard_offset=offset, shape=tuple(tensor.shape),
        )
        published = PublishedTensor(
            name="weight", dtype="torch.float32", elsize=4,
            full_shape=(4, 4), shards=[shard],
        )
        manifests.append(wrap_rendezvous_blob(
            agent.encode(), agent, f"{agent}:19000", [published]
        ))
    capture = CaptureResult(copies=[RecordedCopy(
        "weight", (), "weight", 0, (4, 4), (4, 1), torch.float32
    )])
    layout = {"weight": ((4, 4), torch.float32)}

    capture_calls = []

    def capture_layout(_schema):
        capture_calls.append(_schema)
        return capture, layout

    from modelexpress_rl.inference.adapter import TrainerSourceShard
    from modelexpress_rl.inference.plan import TrainerSourceSnapshot

    snapshot = TrainerSourceSnapshot(
        "mesh", 1, tuple(
            TrainerSourceShard(
                slot, f"worker-{index}", structural_manifest_digest(blob),
                f"{agent}:19000", blob, structural_manifest_digest(blob),
            )
            for index, (slot, agent, blob) in enumerate((
                ("slot-z", "agent-z", manifests[0]),
                ("slot-a", "agent-a", manifests[1]),
            ))
        ),
    )
    harness.transfer._capture_layout = capture_layout
    request = dict(
        manifests=manifests,
        trainer_snapshot=snapshot,
    )
    prepared = harness.transfer.prepare(**request)
    metrics, installed = harness.collect(prepared)
    expected = torch.cat((first, second))
    assert torch.equal(installed["weight"], expected)
    assert metrics["bytes_received"] == expected.numel() * expected.element_size()

    first.add_(1000)
    second.add_(1000)
    request["trainer_snapshot"] = harness.transfer.cached_trainer_source()
    prepared = harness.transfer.prepare(**request)
    _second_metrics, second_install = harness.collect(prepared)
    assert torch.equal(second_install["weight"], torch.cat((first, second)))
    assert len(capture_calls) == 1
