# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext

import pytest
import torch

import modelexpress_rl.inference.nixl_staged_transfer as module
from modelexpress_rl.inference.plan import StreamingSettings
from tests.test_refit_bounded_descriptors import harness


@pytest.mark.parametrize("harness", [None, StreamingSettings(512, "cpu", 2)], indirect=True)
def test_native_conversion_records_generator_device_for_local_replay(harness, monkeypatch):
    h = harness
    monkeypatch.setattr(module, "classic_cuda_alloc", nullcontext)
    h.transfer._device = torch.device("cpu")
    h.capture.source_replay = True
    h.transfer._convert_native_to_hf = lambda weights: {
        name: value.to(device=value.device, dtype=torch.float32)
        for name, value in weights.items()
    }
    prepared = h.prepare()
    batches = (
        (h.transfer.stage(prepared).tensors,)
        if h.streaming is None else h.transfer.iter_bounded(prepared, {})
    )
    for batch in batches:
        for _, copy, tensor in batch.writes:
            replayed = module._replay_ops(tensor, copy.op_chain)
            assert replayed.device == torch.device("cpu")
            expected = h.sources[copy.src_name]
            if copy.src_name == "full":
                expected = expected.t()
            torch.testing.assert_close(replayed, expected)
