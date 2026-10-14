# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Timing for the RL generator refit path.

The recorder and its stage vocabulary are covered by ``test_refit_timing``.
What these tests cover is the part that was missing: something on this path
opening a cycle, and the layers underneath it contributing the durations they
already measure. Without it a framework could see the total refit and roughly
18% of its breakdown, with the wire transfer, the reconstruction, post-load
processing and the copy into kernel storage all charged as one span.
"""

import logging
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from modelexpress.refit import (
    RefitTimingRecorder,
    current_refit_timing,
    set_refit_cold,
    use_refit_timing,
)
from modelexpress_rl import timing
from modelexpress_rl.inference.methods.load_time_tensor import (
    LoadTimeTensorNixlUpdateMethod,
    _attribute_transfer,
)
from modelexpress_rl.inference.plan import ResolvedTrainerSource, TrainerSourceSnapshot
from modelexpress_rl.train import WeightPayloadFormat

STAGED_METRICS = {
    "bytes_received": 4_000_000_000,
    "segments": 12,
    "wire_s": 1.5,
    "reconstruct_s": 0.25,
}


def test_a_cycle_is_opened_by_default(monkeypatch):
    """On by default, because the durations are measured either way and a refit
    nobody can break down is the state this replaced."""
    monkeypatch.delenv("MX_REFIT_TIMING", raising=False)

    assert timing.start_cycle(version_id="run.a1:7", rank=0) is not None


def test_a_cycle_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("MX_REFIT_TIMING", "0")

    assert timing.start_cycle(version_id="run.a1:7", rank=0) is None


def test_a_caller_driving_its_own_cycle_is_not_displaced(monkeypatch):
    """Nesting would split one refit across two records and leave both looking
    partial, so an outer cycle wins and this path contributes to it instead."""
    monkeypatch.delenv("MX_REFIT_TIMING", raising=False)
    outer = RefitTimingRecorder(backend="caller", version=1)

    with use_refit_timing(outer):
        assert timing.start_cycle(version_id="run.a1:7", rank=0) is None
        _attribute_transfer(STAGED_METRICS)

    assert outer.has_measurements("wire_transfer")


def test_the_wire_and_the_work_after_it_are_charged_apart(monkeypatch):
    """The distinction the split exists for: time on the fabric against time
    replaying the copy chain and converting dtypes. Together they cannot
    separate a slow fabric from an expensive layout."""
    monkeypatch.delenv("MX_REFIT_TIMING", raising=False)
    recorder = timing.start_cycle(version_id="run.a1:7", rank=0)

    with timing.active(recorder):
        _attribute_transfer(STAGED_METRICS)

    stages = recorder.as_dict()["stages"]
    assert stages["wire_transfer"]["duration_ms"] == 1500.0
    assert stages["receive_sync"]["duration_ms"] == 250.0


def test_the_payload_size_travels_with_the_timing(monkeypatch):
    """A duration without its bytes cannot be turned into a rate, and the rate
    is what says whether a transfer was near line speed."""
    monkeypatch.delenv("MX_REFIT_TIMING", raising=False)
    recorder = timing.start_cycle(version_id="run.a1:7", rank=0)

    with timing.active(recorder):
        _attribute_transfer(STAGED_METRICS)

    assert recorder.as_dict()["bytes"] == 4_000_000_000


def test_a_peer_pull_reports_a_wire_time_and_no_reconstruction(monkeypatch):
    """Pulling an identical-rank peer's canonical buffers needs no replay, and
    reports no ``reconstruct_s``. The stage has to stay unmeasured rather than
    be recorded as zero, so a mean over cycles is not dragged down by cycles
    that never had the work."""
    monkeypatch.delenv("MX_REFIT_TIMING", raising=False)
    recorder = timing.start_cycle(version_id="run.a1:7", rank=0)

    with timing.active(recorder):
        _attribute_transfer({"bytes_received": 1, "wire_s": 0.5, "peer_s": 0.6})

    assert recorder.has_measurements("wire_transfer")
    assert not recorder.has_measurements("receive_sync")


def test_reusing_a_transfer_plan_is_marked_warm(monkeypatch):
    """Planning dominates a cold cycle and is close to free on a warm one, so a
    record that did not say which is a number describing neither."""
    monkeypatch.delenv("MX_REFIT_TIMING", raising=False)
    recorder = timing.start_cycle(version_id="run.a1:7", rank=0)

    with timing.active(recorder):
        set_refit_cold(False)

    assert recorder.as_dict()["cold_warm"] == "warm"


@pytest.mark.parametrize(
    "bounded, outcome",
    [(False, "complete"), (True, "complete"), (True, "error"), (True, "close")],
)
def test_transfer_timing_reports_cache_reuse_and_consumed_bytes(bounded, outcome) -> None:
    class Transfer:
        bounded_staging = bounded

        def __init__(self) -> None:
            self.hits = iter((0, 1))

        def prepare(self, **_kwargs) -> SimpleNamespace:
            return SimpleNamespace(
                metrics={"plan_cache_hits": next(self.hits)},
                batches=[SimpleNamespace(layouts=({"weight": None},))],
            )

        def stage(self, _prepared) -> SimpleNamespace:
            return SimpleNamespace(metrics=dict(STAGED_METRICS))

        def iter_bounded(self, _prepared, metrics):
            metrics.update(STAGED_METRICS)
            yield {"weight": 7}
            if outcome == "error":
                raise RuntimeError("transfer failed")

    method = LoadTimeTensorNixlUpdateMethod(transfer=Transfer())
    source = ResolvedTrainerSource(TrainerSourceSnapshot("mesh", 1, ()))
    version = SimpleNamespace(
        base_version_id=None,
        layout_signature="layout",
        payload_format=WeightPayloadFormat.FULL_TENSOR,
    )
    for label, expected in (("v1", "cold"), ("v2", "warm")):
        recorder = RefitTimingRecorder(backend="rl_generator", version=label, rank=0)
        with use_refit_timing(recorder):
            prepared = method.prepare(version=version, source=source)
            expected_error = (
                pytest.raises(RuntimeError, match="transfer failed")
                if outcome == "error" else nullcontext()
            )
            with expected_error, method.installation_context(prepared):
                if bounded:
                    assert recorder.as_dict()["bytes"] == 0
                    batches = prepared.batches()
                    assert next(batches) == {"weight": 7}
                    if outcome == "error":
                        next(batches)
                    elif outcome == "complete":
                        assert list(batches) == []
                    batches.close()
            method.release(prepared)
        record = recorder.as_dict()
        assert record["cold_warm"] == expected
        assert record["stages"]["transfer_planning"]["count"] == 1
        assert record["bytes"] == STAGED_METRICS["bytes_received"]
        assert record["stages"]["wire_transfer"]["duration_ms"] == 1500.0
        assert record["stages"]["receive_sync"]["duration_ms"] == 250.0


def test_the_record_is_emitted_once(monkeypatch, caplog):
    """The client reports from both the apply and the release path, so that a
    version staged and dropped without ever installing is still reported. Only
    one of them may actually produce a record."""
    monkeypatch.delenv("MX_REFIT_TIMING", raising=False)
    recorder = timing.start_cycle(version_id="run.a1:7", rank=0)
    logger = logging.getLogger("test_rl_generator_timing")

    with caplog.at_level(logging.INFO, logger=logger.name):
        timing.emit(recorder, logger)
        timing.emit(recorder, logger)

    assert len([r for r in caplog.records if "MX_REFIT_TIMING" in r.getMessage()]) == 1


def test_nothing_is_emitted_when_there_is_no_cycle():
    """Every helper here is reachable from a caller that never started a cycle,
    so absence has to be ordinary rather than an error."""
    logger = logging.getLogger("test_rl_generator_timing")

    timing.emit(None, logger)
    with timing.active(None):
        _attribute_transfer(STAGED_METRICS)
        set_refit_cold(True)

    assert current_refit_timing() is None
