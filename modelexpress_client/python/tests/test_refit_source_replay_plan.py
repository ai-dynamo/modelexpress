# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes

import torch
from modelexpress.refit.reshard.slice_plan import Shard
from modelexpress.refit.reshard.transfer_plan import SourceInfo, plan_transfer
from modelexpress.refit.reshard.types import CaptureResult, RecordedCopy
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
