# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import pytest

import torch
from modelexpress_rl.refit.reshard.slice_plan import Shard
from modelexpress_rl.refit.reshard.transfer_plan import SourceInfo, plan_transfer
from modelexpress_rl.refit.reshard.types import CaptureResult, RecordedCopy, UnsupportedReshard
from tests.helpers.refit_replay import replay_ops as _replay_ops


def test_digest_mode_does_not_replace_partial_replay_reads_with_full_sources():
    source = torch.arange(3 * 64, dtype=torch.float32).reshape(3, 64)
    sources = {"experts.w13": SourceInfo(
        tuple(source.shape), source.dtype, 4,
        [Shard((0, 0), tuple(source.shape), "worker", source.data_ptr(), 4)],
    )}
    capture = CaptureResult(copies=[
        RecordedCopy(
            "experts.w13", (("select", (0, i), ()),), f"expert{i}.weight",
            0, (64,), (1,), source.dtype,
        ) for i in range(3)
    ], source_replay=True)

    normal = plan_transfer(capture, sources, max_staging_bytes=256)
    assert normal.bytes_planned() == source.numel() * source.element_size()
    assert [pull.global_shape for pull in normal.full_pulls] == [(64,)] * 3
    with pytest.raises(UnsupportedReshard, match="whole-shard digests cannot verify"):
        plan_transfer(capture, sources, max_staging_bytes=256, verify_full_sources=True)


def test_digest_mode_preserves_the_normal_replay_plan_and_interleaved_write_order():
    source = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    sources = {name: SourceInfo(
        tuple(source.shape), source.dtype, 4,
        [Shard((0, 0), tuple(source.shape), name, source.data_ptr(), 4)],
    ) for name in ("a", "b")}
    capture = CaptureResult(copies=[
        RecordedCopy(
            name, (("transpose", (0, 1), ()),), f"owner{i}.weight",
            0, (4, 4), (4, 1), source.dtype,
        ) for i, name in enumerate(("a", "b", "a"))
    ], source_replay=True)

    normal = plan_transfer(capture, sources, max_segments_per_copy=1)
    verified = plan_transfer(
        capture, sources, max_segments_per_copy=1, verify_full_sources=True
    )
    assert verified.full_pulls == normal.full_pulls
    assert verified.source_buffers == normal.source_buffers
    assert verified.bytes_planned() == normal.bytes_planned()
    assert [p.copy_indices for p in verified.full_pulls] == [(0,), (1,), (2,)]
    assert [p.copies[0].op_chain for p in verified.full_pulls] == [
        c.op_chain for c in capture.copies
    ]


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("max_segments", [64, 512])
@pytest.mark.parametrize("arena_bytes", [None, 32768])
def test_replay_bounds_column_reads_without_exceeding_arena(rank, max_segments, arena_bytes):
    source = torch.arange(8 * 256 * 64, dtype=torch.float32).reshape(8, 256, 64)
    shards = [
        Shard((expert, 0, 0), (1, 256, 64), f"s{expert}", source[expert].data_ptr(), 4)
        for expert in range(8)
    ]
    copy = RecordedCopy(
        "x", (("select", (0, 5), ()), ("narrow", (1, rank * 32, 32), ())),
        "a.weight", 0, (256, 32), (32, 1), source.dtype,
    )
    options = {} if arena_bytes is None else {"max_staging_bytes": arena_bytes}
    plan = plan_transfer(
        CaptureResult(copies=[copy], source_replay=True),
        {"x": SourceInfo(tuple(source.shape), source.dtype, 4, shards)},
        max_segments_per_copy=max_segments, **options,
    )
    pull = plan.full_pulls[0]
    coarsened = max_segments < 256 and arena_bytes is None
    assert len(pull.segments) == (1 if coarsened else 256)
    assert pull.copy_indices == (0,)
    assert plan.exact_descriptor_count == 256
    assert plan.exact_bytes == 32768
    received = torch.empty(pull.global_shape, dtype=pull.dtype)
    if arena_bytes is not None:
        assert received.numel() * received.element_size() <= arena_bytes
    for read in pull.segments:
        ctypes.memmove(
            received.data_ptr() + read.dst_byte,
            shards[read.source_buffer].addr + read.src_byte,
            read.nbytes,
        )
    actual = _replay_ops(received, pull.copies[0].op_chain)
    torch.testing.assert_close(actual, source[5, :, rank * 32:(rank + 1) * 32], rtol=0, atol=0)


def test_replay_materializes_transposed_source_before_flattening():
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    copy = RecordedCopy(
        'x', (('t', (), ()), ('contiguous', (), ()), ('view', (6,), ())),
        'a.weight', 0, (6,), (1,), torch.float32,
    )
    sources = {'x': SourceInfo(
        (2, 3), torch.float32, 4,
        [Shard((0, 0), (2, 3), 's', source.data_ptr(), 4)],
    )}
    plan = plan_transfer(CaptureResult(copies=[copy], source_replay=True), sources)
    pull = plan.full_pulls[0]
    received = torch.empty(pull.global_shape, dtype=pull.dtype)
    for read in pull.segments:
        ctypes.memmove(
            received.data_ptr() + read.dst_byte,
            source.data_ptr() + read.src_byte,
            read.nbytes,
        )
    actual = _replay_ops(received, pull.copies[0].op_chain)
    torch.testing.assert_close(actual, source.t().contiguous().view(6))


@pytest.mark.parametrize("index", [
    [2, 0, 2], torch.tensor([2, 0, 2]),
    (slice(None), [3, 1]), ((2, 0),),
    (torch.tensor([2, 0]), torch.tensor([3, 1])),
    [], [-1, 0],
])
def test_integer_index_capture_transfer_and_replay(index):
    from modelexpress_rl.refit.reshard.geometry import build_lazy_weights, capture_weights

    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    expected = source[index].clone()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.empty_like(expected, device="meta"))

        def load_weights(self, weights):
            for _, value in weights:
                self.weight.copy_(value[index])

    captured = capture_weights(
        Model(), build_lazy_weights([("x", source.dtype, tuple(source.shape))]),
        source_replay=True,
    )
    assert not captured.unsupported
    assert len(captured.copies) == 1
    # Capture owns the indices: later caller changes must not change replay.
    if isinstance(index, list):
        index[:] = [1]
    elif isinstance(index, torch.Tensor):
        index.fill_(1)
    sources = {"x": SourceInfo(
        tuple(source.shape), source.dtype, source.element_size(),
        [Shard((0, 0), tuple(source.shape), "s", source.data_ptr(), 4)],
    )}
    plan = plan_transfer(captured, sources)
    pull = plan.full_pulls[0]
    received = torch.empty(pull.global_shape, dtype=pull.dtype)
    for read in pull.segments:
        ctypes.memmove(received.data_ptr() + read.dst_byte,
                       source.data_ptr() + read.src_byte, read.nbytes)
    torch.testing.assert_close(_replay_ops(received, pull.copies[0].op_chain), expected)


@pytest.mark.parametrize("nodes", [("a", "b", "c", "d", "solo"), ("solo", "d", "c", "b", "a")])
def test_owner_groups_preserve_order_across_transitive_and_cyclic_links(nodes):
    from modelexpress_rl.refit.reshard.types import _connected_groups

    links = (("b", "c"), ("a", "b"), ("d", "c"), ("foreign", "solo"), ("a", "c"))
    groups = _connected_groups(nodes, links)
    connected = tuple(node for node in nodes if node != "solo")
    expected = (("solo",), connected) if nodes[0] == "solo" else (connected, ("solo",))
    assert groups == expected
