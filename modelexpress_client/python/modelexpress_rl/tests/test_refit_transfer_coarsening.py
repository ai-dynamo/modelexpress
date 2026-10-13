# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes

import pytest
import torch
from modelexpress_rl.inference.nixl_staged_transfer import _replay_ops
from modelexpress_rl.refit.reshard.transfer_plan import plan_transfer as _plan_staged_transfer

from modelexpress_rl.refit.reshard.slice_plan import Shard
from modelexpress_rl.refit.reshard.transfer_plan import SourceInfo
from modelexpress_rl.refit.reshard.types import CaptureResult, RecordedCopy


@pytest.fixture(autouse=True)
def planning_options(monkeypatch):
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    monkeypatch.setenv("MX_RESHARD_MAX_SEGMENTS_PER_COPY", "64")


def _inputs(source, ops, expected, shards=None):
    capture = CaptureResult(
        copies=[
            RecordedCopy(
                "source",
                ops,
                "weight",
                0,
                tuple(expected.shape),
                tuple(expected.contiguous().stride()),
                expected.dtype,
            )
        ],
        source_replay=True,
        generation=7,
    )
    table = {
        "source": SourceInfo(
            tuple(source.shape),
            source.dtype,
            source.element_size(),
            shards
            if shards is not None
            else [
                Shard(
                    (0,) * source.ndim,
                    tuple(source.shape),
                    "peer",
                    source.data_ptr(),
                    source.element_size(),
                )
            ],
        )
    }
    return capture, table


def _replay(plan, table):
    outputs = []
    buffers = {
        (name, tuple(shard.shard_offset), tuple(shard.shape)): shard
        for name, source in table.items() for shard in source.shards
    }
    for pull in plan.full_pulls:
        staging = torch.zeros(pull.global_shape, dtype=pull.dtype)
        for segment in reversed(pull.segments):
            buffer = plan.source_buffers[segment.source_buffer]
            shard = buffers[(buffer.src_name, buffer.shard_offset, buffer.shape)]
            ctypes.memmove(
                staging.data_ptr() + segment.dst_byte,
                shard.addr + segment.src_byte,
                segment.nbytes,
            )
        outputs.append(_replay_ops(staging, pull.copies[0].op_chain))
    return outputs


@pytest.mark.parametrize("operation", ["transpose", "columns"])
def test_source_replay_coarsens_excessive_descriptors(operation):
    source = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)
    if operation == "transpose":
        ops, expected, exact_count = (("transpose", (0, 1), ()),), source.t(), 16384
    else:
        ops, expected, exact_count = (("narrow", (1, 0, 64), ()),), source[:, :64], 128
    capture, table = _inputs(source, ops, expected)
    plan = _plan_staged_transfer(capture, table)

    assert plan.descriptor_count() == 1
    assert plan.bytes_planned() == source.numel() * 4
    assert plan.exact_descriptor_count == exact_count
    assert plan.exact_bytes == expected.numel() * 4
    assert plan.descriptor_savings() == exact_count - 1
    assert plan.extra_wire_bytes() == (source.numel() - expected.numel()) * 4
    assert torch.equal(_replay(plan, table)[0], expected)


def test_coarsening_keeps_small_contiguous_prefix_and_cast_suffix(monkeypatch):
    monkeypatch.setenv("MX_RESHARD_MAX_SEGMENTS_PER_COPY", "8")
    source = torch.linspace(0.1, 2.0, 128 * 128).reshape(128, 128)
    ops = (
        ("narrow", (0, 17, 16), ()),
        ("narrow", (1, 3, 64), ()),
        ("bfloat16", (), ()),
        ("float", (), ()),
    )
    expected = source[17:33, 3:67].bfloat16().float()
    capture, table = _inputs(source, ops, expected)
    plan = _plan_staged_transfer(capture, table)

    assert plan.descriptor_count() == 1
    assert plan.bytes_planned() == 16 * 128 * 4
    assert torch.equal(_replay(plan, table)[0], expected)


@pytest.mark.parametrize("limit, descriptors", [(127, 1), (128, 128)])
def test_descriptor_threshold_preserves_efficient_exact_slices(
    monkeypatch, limit, descriptors
):
    monkeypatch.setenv("MX_RESHARD_MAX_SEGMENTS_PER_COPY", str(limit))
    source = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)
    capture, table = _inputs(source, (("narrow", (1, 0, 64), ()),), source[:, :64])
    plan = _plan_staged_transfer(capture, table)
    assert plan.descriptor_count() == descriptors
    assert torch.equal(_replay(plan, table)[0], source[:, :64])


def test_later_contiguous_prefix_avoids_unnecessary_overread():
    source = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)
    ops = (("t", (), ()), ("t", (), ()), ("narrow", (0, 17, 2), ()))
    capture, table = _inputs(source, ops, source[17:19])
    plan = _plan_staged_transfer(capture, table)
    assert plan.descriptor_count() == 1
    assert plan.bytes_planned() == 2 * 128 * 4
    assert torch.equal(_replay(plan, table)[0], source[17:19])


@pytest.mark.parametrize("budget, descriptors", [(65536, 1), (65535, 128), (32768, 128)])
def test_bounded_coarsening_respects_arena_capacity(budget, descriptors):
    source = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)
    expected = source[:, :64]
    capture, table = _inputs(source, (("narrow", (1, 0, 64), ()),), expected)
    plan = _plan_staged_transfer(capture, table, max_staging_bytes=budget)
    assert plan.descriptor_count() == descriptors
    assert torch.equal(_replay(plan, table)[0], expected)


def test_coarsening_keeps_tuple_selection_and_shard_offsets():
    source = torch.arange(3 * 128 * 128, dtype=torch.float32).reshape(3, 128, 128)
    ops = (("unbind", (0,), ()), ("__getitem__", (1,), ()), ("t", (), ()))
    capture, table = _inputs(source, ops, source[1].t())
    plan = _plan_staged_transfer(capture, table)
    assert plan.descriptor_count() == 1
    assert plan.bytes_planned() == 128 * 128 * 4
    assert torch.equal(_replay(plan, table)[0], source[1].t())


def test_coarsening_does_not_add_bytes_without_reducing_descriptors(monkeypatch):
    monkeypatch.setenv("MX_RESHARD_MAX_SEGMENTS_PER_COPY", "4")
    source = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16)
    shards = [
        Shard((row, 0), (1, 16), str(row), source[row].data_ptr(), 4)
        for row in range(8)
    ]
    capture, table = _inputs(
        source, (("narrow", (1, 0, 8), ()),), source[:, :8], shards
    )
    plan = _plan_staged_transfer(capture, table)
    assert plan.descriptor_count() == 8
    assert plan.bytes_planned() == 8 * 8 * 4
    assert torch.equal(_replay(plan, table)[0], source[:, :8])


def test_coarsening_reads_contiguous_runs_from_each_source_shard():
    source = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)
    shards = [
        Shard((row, 0), (64, 128), str(row), source[row].data_ptr(), 4)
        for row in (0, 64)
    ]
    expected = source[:, 7:71].t()
    ops = (("narrow", (1, 7, 64), ()), ("t", (), ()))
    capture, table = _inputs(source, ops, expected, shards)
    plan = _plan_staged_transfer(capture, table)
    assert plan.descriptor_count() == 2
    assert torch.equal(_replay(plan, table)[0], expected)


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_source_replay_rejects_invalid_descriptor_threshold(monkeypatch, limit):
    monkeypatch.setenv("MX_RESHARD_MAX_SEGMENTS_PER_COPY", limit)
    source = torch.ones(4)
    capture, table = _inputs(source, (), source)
    with pytest.raises(ValueError, match="max_segments_per_copy must be at least 1"):
        _plan_staged_transfer(capture, table)




def test_fragmented_shards_get_best_reduction_when_threshold_cannot_be_met():
    source = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16)
    shards = [
        Shard((row, 0), (1, 16), str(row), source[row].data_ptr(), 4)
        for row in range(8)
    ]
    capture, table = _inputs(source, (("t", (), ()),), source.t(), shards)
    plan = _plan_staged_transfer(capture, table, max_segments_per_copy=1)
    assert plan.descriptor_count() == 8
    assert plan.exact_descriptor_count == 128
    assert plan.extra_wire_bytes() == 0
    torch.testing.assert_close(_replay(plan, table)[0], source.t())


def test_whole_source_verification_preserves_all_bytes_under_coarsening():
    source = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)
    capture, table = _inputs(source, (("narrow", (1, 0, 64), ()),), source[:, :64])
    plan = _plan_staged_transfer(capture, table, verify_full_sources=True,
                                max_staging_bytes=32768)
    assert plan.verify_full_sources
    assert plan.bytes_planned() == 65536
    torch.testing.assert_close(_replay(plan, table)[0], source[:, :64])
