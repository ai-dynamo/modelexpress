# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch

from modelexpress_rl.refit.reshard.types import IncompleteRefit, UnsupportedReshard
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


@pytest.mark.parametrize("harness", [None, StreamingSettings(512, "cpu", 2)], indirect=True)
@pytest.mark.parametrize("missing", [False, True])
def test_replay_digest_verification_rejects_missing_or_stale_digests(
    harness, monkeypatch, missing
):
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1")
    monkeypatch.setattr(module, "classic_cuda_alloc", nullcontext)
    h = harness
    h.transfer._device = torch.device("cpu")
    h.capture.source_replay = True
    if missing:
        monkeypatch.setattr(
            "tests.test_refit_bounded_descriptors.tensor_digest", lambda _tensor: ""
        )
    prepared = h.prepare()
    if not missing:
        h.sources["exact"].add_(1)

    message = "did not publish a verification digest" if missing else "digest mismatch"
    with pytest.raises(RuntimeError, match=message):
        if h.streaming is None:
            h.transfer.stage(prepared)
        else:
            next(h.transfer.iter_bounded(prepared, {}))


@pytest.mark.parametrize("harness", [None, StreamingSettings(512, "cpu", 2)], indirect=True)
def test_partial_replay_digest_rejection_precedes_native_setup(harness, monkeypatch):
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1")
    h = harness
    h.capture.source_replay = True
    h.capture.copies[0] = replace(
        h.capture.copies[0], op_chain=(("narrow", (0, 0, 2), ()),),
        dest_shape=(2, 4),
    )
    before = list(h.events)

    with pytest.raises(UnsupportedReshard, match="whole-shard digests cannot verify"):
        h.prepare()

    assert h.events == before


@pytest.mark.parametrize("harness", [StreamingSettings(512, "cpu", 2)], indirect=True)
def test_replay_scratch_analysis_is_reused_until_generation_changes(harness, monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setattr(module, "classic_cuda_alloc", nullcontext)
    h = harness
    h.capture.source_replay = True
    analyze = Mock(wraps=module.replay_scratch_bytes)
    monkeypatch.setattr(module, "replay_scratch_bytes", analyze)
    for update in range(2):
        prepared = h.prepare()
        list(h.transfer.iter_bounded(prepared, {}))
        if update == 0:
            cold_calls = analyze.call_count
            assert cold_calls == len(h.capture.copies)
        else:
            assert analyze.call_count == cold_calls
    list(h.transfer.iter_bounded(h.prepare(generation=2), {}))
    assert analyze.call_count == cold_calls * 2


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


@pytest.mark.parametrize("harness", [StreamingSettings(2048, "cpu", 2)], indirect=True)
@pytest.mark.parametrize("replay", [False, True])
def test_bounded_receive_clears_used_arena_once_per_batch(harness, monkeypatch, replay):
    monkeypatch.setattr(module, "classic_cuda_alloc", nullcontext)
    h = harness
    h.capture.source_replay = replay
    prepared = h.prepare()
    zero = torch.Tensor.zero_
    cleared = []

    def clear(tensor):
        cleared.append(tensor.numel() * tensor.element_size())
        return zero(tensor)

    monkeypatch.setattr(torch.Tensor, "zero_", clear)
    for batch in h.transfer.iter_bounded(prepared, {}):
        if replay:
            for _, copy, source in batch.writes:
                expected = h.sources[copy.src_name]
                if copy.src_name == "full":
                    expected = expected.t()
                assert torch.equal(_replay_ops(source, copy.op_chain), expected)
        else:
            # Copy destinations leave two leading and two trailing values unwritten.
            for tensor in batch.values():
                assert torch.equal(tensor[:2], torch.zeros(2, dtype=tensor.dtype))
                assert torch.equal(tensor[-2:], torch.zeros(2, dtype=tensor.dtype))
    assert cleared == [batch.nbytes for batch in prepared.batches]


@pytest.mark.parametrize("harness", [StreamingSettings(512, "cpu", 2)], indirect=True)
def test_coarsening_preserves_warm_replay_and_reports_wire_tradeoff(harness):
    h = harness
    h.capture.source_replay = True
    h.capture.copies[1] = replace(
        h.capture.copies[1],
        op_chain=(("transpose", (0, 1), ()), ("narrow", (0, 0, 2), ())),
        dest_shape=(2, 4),
    )
    for iteration in range(2):
        h.sources["full"].add_(1)
        prepared = h.prepare()
        assert prepared.metrics["descriptor_savings"] > 0
        assert prepared.metrics["extra_wire_bytes"] == 32
        assert prepared.metrics["exact_descriptors"] > prepared.metrics["planned_descriptors"]
        assert prepared.metrics["plan_cache_hits"] == iteration
        for batch in h.transfer.iter_bounded(prepared, {}):
            for index, copy, source in batch.writes:
                if index == 1:
                    torch.testing.assert_close(
                        _replay_ops(source, copy.op_chain), h.sources["full"].t()[:2]
                    )
