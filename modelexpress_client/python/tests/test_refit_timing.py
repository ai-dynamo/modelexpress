# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging

import pytest
from modelexpress.refit import (
    MX_REFIT_TIMING_PREFIX,
    REFIT_TIMING_STAGES,
    RefitTimingRecorder,
    add_refit_duration,
    add_refit_metadata,
    current_refit_timing,
    refit_span,
    use_refit_timing,
)
from modelexpress.refit_timing import (
    RefitTimingRecorder as CompatibilityRefitTimingRecorder,
)


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_legacy_refit_timing_import_is_compatible():
    assert CompatibilityRefitTimingRecorder is RefitTimingRecorder


def test_stage_order_repeated_spans_and_unattributed_sum():
    clock = _Clock()
    timing = RefitTimingRecorder(
        backend="vllm",
        version=7,
        rank=2,
        tp_rank=1,
        tp_size=4,
        ep_rank=0,
        ep_size=2,
        cold=True,
        clock=clock,
    )

    with timing.span("wire_transfer"):
        clock.advance(2.0)
    with timing.span("wire_transfer"):
        clock.advance(3.0)
    timing.mark_not_applicable("post_install", combined_with="installation")
    clock.advance(5.0)
    timing.finish()

    payload = timing.as_dict()
    assert tuple(payload["stages"]) == REFIT_TIMING_STAGES
    assert payload["stages"]["wire_transfer"] == {
        "status": "ok",
        "duration_ms": 5000.0,
        "count": 2,
    }
    assert payload["stages"]["post_install"]["status"] == "combined"
    assert payload["e2e_ms"] == 10000.0
    assert payload["unattributed_ms"] == 5000.0
    assert payload["cold_warm"] == "cold"
    assert timing.has_measurements("wire_transfer")
    assert not timing.has_measurements("post_install")


def test_error_span_is_recorded_and_reraised():
    clock = _Clock()
    timing = RefitTimingRecorder(backend="test", version="bad", clock=clock)

    with pytest.raises(RuntimeError, match="boom"), timing.span("transformation"):
        clock.advance(0.25)
        raise RuntimeError("boom")

    stage = timing.as_dict()["stages"]["transformation"]
    assert stage["status"] == "error"
    assert stage["duration_ms"] == 250.0


def test_context_helpers_and_emit_once(caplog):
    logger = logging.getLogger("modelexpress.test.refit_timing")
    timing = RefitTimingRecorder(backend="test", version=1)

    with caplog.at_level(logging.INFO, logger=logger.name):
        with use_refit_timing(timing):
            assert current_refit_timing() is timing
            with refit_span("control_discovery"):
                pass
        assert current_refit_timing() is None
        first = timing.emit(logger)
        timing.add_bytes(7)
        second = timing.emit(logger)

    lines = [
        record.message
        for record in caplog.records
        if record.message.startswith(MX_REFIT_TIMING_PREFIX)
    ]
    assert len(lines) == 1
    assert json.loads(lines[0].split(" ", 1)[1]) == first == second
    assert second["bytes"] == 0


def test_unknown_stage_and_negative_bytes_rejected():
    timing = RefitTimingRecorder(backend="test", version=1)
    with pytest.raises(ValueError, match="unknown refit timing stage"):
        timing.add_duration("download", 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        timing.add_bytes(-1)


def test_a_span_can_name_its_duration_and_report_what_it_discovered():
    """Several spans usually share one normalized stage, so the stage total
    cannot say which of them was expensive. Naming them can, and a counter only
    knowable once the work has run has somewhere to go."""
    clock = _Clock()
    recorder = RefitTimingRecorder(backend="test", version="v1", clock=clock)

    with use_refit_timing(recorder):
        with refit_span(
            "source_preparation",
            accumulate_metadata=True,
            duration_key="manifest_fetch_s",
        ) as discovered:
            clock.advance(0.25)
            discovered["manifest_bytes"] = 4096
        with refit_span(
            "source_preparation",
            accumulate_metadata=True,
            duration_key="manifest_hash_s",
        ) as discovered:
            clock.advance(0.5)
            discovered["manifest_bytes"] = 1024

    stage = recorder.as_dict()["stages"]["source_preparation"]
    assert stage["count"] == 2
    assert stage["duration_ms"] == pytest.approx(750.0)
    assert stage["metadata"]["manifest_fetch_s"] == pytest.approx(0.25)
    assert stage["metadata"]["manifest_hash_s"] == pytest.approx(0.5)
    assert stage["metadata"]["manifest_bytes"] == 5120


def test_a_failed_span_still_names_the_duration_it_spent():
    clock = _Clock()
    recorder = RefitTimingRecorder(backend="test", version="v1", clock=clock)

    with (
        use_refit_timing(recorder),
        pytest.raises(RuntimeError),
        refit_span("source_preparation", duration_key="manifest_fetch_s"),
    ):
        clock.advance(0.125)
        raise RuntimeError("manifest fetch failed")

    stage = recorder.as_dict()["stages"]["source_preparation"]
    assert stage["status"] == "error"
    assert stage["metadata"]["manifest_fetch_s"] == pytest.approx(0.125)


def test_metadata_can_be_attached_without_inventing_a_span():
    """Sizes and cache outcomes describe a stage but have no duration of their
    own. Filing them through a zero-second span would leave the stage looking
    like it ran more times than it did."""
    recorder = RefitTimingRecorder(backend="test", version="v1")

    with use_refit_timing(recorder):
        add_refit_metadata(
            "source_preparation", {"manifest_bytes": 32}, accumulate=True
        )
        add_refit_metadata("source_preparation", {"manifest_bytes": 8}, accumulate=True)
        add_refit_metadata("source_preparation", {"manifest_cache_hit": True})

    stage = recorder.as_dict()["stages"]["source_preparation"]
    assert stage["count"] == 0
    assert stage["metadata"] == {"manifest_bytes": 40, "manifest_cache_hit": True}


def test_a_duration_from_another_layer_is_clamped_not_rejected():
    """These arrive from a clock this process did not read, so a small negative
    is a clock artifact rather than a caller bug worth raising on."""
    recorder = RefitTimingRecorder(backend="test", version="v1")

    with use_refit_timing(recorder):
        add_refit_duration("wire_transfer", -0.01)

    assert recorder.as_dict()["stages"]["wire_transfer"]["duration_ms"] == 0.0
