# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch

from modelexpress.refit.reshard.types import IncompleteRefit
import modelexpress_rl.inference.nixl_staged_transfer as module
from modelexpress_rl.inference.nixl_staged_transfer import _replay_ops
from modelexpress_rl.inference.plan import StreamingSettings
from tests.test_refit_bounded_descriptors import harness


@pytest.mark.parametrize("harness", [None, StreamingSettings(512, "cpu", 2)], indirect=True)
@pytest.mark.parametrize("verify", [False, True])
def test_ordered_sources_survive_warm_reuse(harness, monkeypatch, verify):
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", str(int(verify)))
    monkeypatch.setattr(module, "classic_cuda_alloc", nullcontext)
    h = harness
    h.transfer._device = torch.device("cpu")
    h.capture.source_replay = True
    # The final write revisits the first owner, crossing two receive arenas.
    h.capture.copies[-1] = replace(h.capture.copies[-1], param_name="a.weight")
    del h.layout["d.weight"]
    for iteration in range(2):
        for source in h.sources.values():
            source.add_(1)
        prepared = h.prepare()
        if h.streaming is None:
            batches = (h.transfer.stage(prepared).tensors,)
        else:
            batches = h.transfer.iter_bounded(prepared, {})
        seen = []
        for batch in batches:
            for index, copy, source in batch.writes:
                expected = h.sources[copy.src_name]
                if copy.src_name == "full":
                    expected = expected.t()
                assert torch.equal(_replay_ops(source, copy.op_chain), expected)
                seen.append((index, copy.param_name))
        assert seen == [(0, "a.weight"), (1, "b.weight"), (2, "c.weight"), (3, "a.weight")]
        assert prepared.metrics["plan_cache_hits"] == iteration
    assert len(h.captures) == 1


def test_replay_rejects_oversized_write_before_native_setup(harness):
    h = harness
    h.capture.source_replay = True
    h.transfer._buffer_budget = 255
    before = list(h.events)
    with pytest.raises(IncompleteRefit, match="recorded source write requires"):
        h.prepare()
    assert h.events == before


def test_replay_rejects_destination_bounds_before_native_setup(harness):
    h = harness
    h.capture.source_replay = True
    h.capture.copies[0] = replace(h.capture.copies[0], dest_offset=10)
    before = list(h.events)
    with pytest.raises(IncompleteRefit, match="physical owner bounds"):
        h.prepare()
    assert h.events == before


@pytest.mark.parametrize("harness", [StreamingSettings(512, "cpu", 2)], indirect=True)
def test_replay_abandonment_drains_prefetched_read(harness):
    h = harness
    h.capture.source_replay = True
    prepared = h.prepare()
    iterator = h.transfer.iter_bounded(prepared, {})
    assert next(iterator).writes
    iterator.close()
    transport = h.transports[-1]
    assert transport.posted == transport.awaited
