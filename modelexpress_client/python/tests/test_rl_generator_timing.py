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
from types import SimpleNamespace

from modelexpress.refit import (
    RefitTimingRecorder,
    current_refit_timing,
    use_refit_timing,
)
from modelexpress_rl import timing
from modelexpress_rl.inference.adapter import (
    GeneratorSource,
    GeneratorTransferInputs,
    NixlGeneratorSource,
)
from modelexpress_rl.inference.methods.full_tensor import (
    FullTensorNixlUpdateMethod,
    _attribute_transfer,
)
from modelexpress_rl.inference.plan import TrainerUpdateSource
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
        timing.record_cold(False)

    assert recorder.as_dict()["cold_warm"] == "warm"


def test_version_digest_refreshes_verification_without_replanning():
    class Transfer:
        def __init__(self):
            self.prepare_calls = 0
            self.refresh_calls = 0

        def unpublish_peer(self):
            pass

        def prepare(self, **_kwargs):
            self.prepare_calls += 1
            return object()

        def refresh_sources(self, _prepared, _manifests):
            self.refresh_calls += 1

        def stage(self, _prepared):
            return SimpleNamespace(metrics={"bytes_received": 0})

    transfer = Transfer()
    method = FullTensorNixlUpdateMethod(
        transfer=transfer,
        capture_layout=lambda _manifest: None,
        parameter_layout=dict,
        build_identity=lambda _version: None,
        worker_rank=0,
        worker_id="worker",
        accelerator="cuda",
        p2p_client=object(),
    )

    def source(version, digest):
        return TrainerUpdateSource(
            inputs=GeneratorTransferInputs(
                version_id=version,
                base_version_id=None,
                layout_signature="layout",
                payload_format=WeightPayloadFormat.FULL_TENSOR,
                sources=(
                    GeneratorSource(
                        source_slot_id="rank:0",
                        worker_id="trainer-0",
                        manifest_digest=digest,
                        transport=NixlGeneratorSource(
                            manifest_endpoint="trainer:9000",
                            manifest=version.encode(),
                            structural_digest="stable-structure",
                        ),
                    ),
                ),
            )
        )

    first = method.prepare(version=None, source=source("v1", "digest-1"))
    method.release(first)
    second = method.prepare(version=None, source=source("v2", "digest-2"))
    method.release(second)

    assert transfer.prepare_calls == 1
    assert transfer.refresh_calls == 1


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
        timing.record_cold(True)

    assert current_refit_timing() is None
