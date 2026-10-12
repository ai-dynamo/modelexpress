# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import pytest

import torch
from modelexpress_rl.refit.reshard.slice_plan import Shard
from modelexpress_rl.refit.reshard.transfer_plan import SourceInfo, plan_transfer
from modelexpress_rl.refit.reshard.types import CaptureResult, RecordedCopy
from modelexpress_rl.inference.nixl_staged_transfer import _replay_ops


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
