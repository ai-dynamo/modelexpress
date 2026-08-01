# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""The per-refit stage record emitted by ReshardReceiver.update_weights.

A refit is one number to the caller and six or more stages underneath: discovery,
the peer handshake, geometry capture, planning, allocation and registration on the
first step, then wire, re-slice, dtype cast and install on every step. Without a
breakdown a regression can only be described as "refit got slower".

These tests pin the properties a consumer of the record depends on:

  * it is one JSON object per refit at WARNING, so a harness capturing it does not
    have to turn on INFO across every dependency;
  * ``accounted_s`` equals the sum of the stage durations, which is what lets a
    caller compute the unattributed remainder and decide whether to trust the
    breakdown at all;
  * one-time setup costs are attributed to the step that paid them rather than
    silently dropped, so a cold first step accounts for itself;
  * the byte economics travel with the timings, so useful bytes are read off the
    record instead of reconstructed from an assumed sharding afterwards.

The last one is the reason this record exists in this form: a derived
useful-bytes figure once made a half-complete refit look like the faster one.

Run: pytest tests/test_reshard_refit_stage_record.py
"""

import json
import logging

import pytest
import torch

from tests.test_reshard_refit_fused_wire import _build, _RecordingTransport


def _run(monkeypatch, caplog, *, fused=True, enabled=True):
    monkeypatch.setenv("MX_RESHARD_FUSED_WIRE", "1" if fused else "0")
    monkeypatch.setenv("MX_REFIT_STAGE_RECORD", "1" if enabled else "0")
    monkeypatch.delenv("MX_RESHARD_MAX_GBPS", raising=False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    transport = _RecordingTransport()
    harness, keepalive = _build(transport)
    with caplog.at_level(logging.WARNING):
        metrics = harness.update_weights(step=7)
    return harness, metrics, keepalive


def _record(caplog):
    line = next(m for m in caplog.messages if "MX_REFIT_STAGE" in m)
    return json.loads(line.split("MX_REFIT_STAGE ", 1)[1])


def test_one_record_per_refit_at_warning(monkeypatch, caplog):
    _h, _m, _k = _run(monkeypatch, caplog)
    lines = [m for m in caplog.messages if "MX_REFIT_STAGE " in m]
    assert len(lines) == 1
    record = _record(caplog)
    assert record["schema"] == "refit-stage-v2"
    assert record["step"] == 7


def test_the_record_is_off_when_disabled(monkeypatch, caplog):
    """A caller that does not want the line must be able to silence it, since it is
    emitted at WARNING and would otherwise be unavoidable noise."""
    _h, _m, _k = _run(monkeypatch, caplog, enabled=False)
    assert not [m for m in caplog.messages if "MX_REFIT_STAGE " in m]


def test_accounted_s_is_the_sum_of_the_stages(monkeypatch, caplog):
    """The whole point of the field: a caller subtracts it from its own end-to-end
    figure to get the unattributed remainder."""
    _h, _m, _k = _run(monkeypatch, caplog)
    record = _record(caplog)
    stage_total = sum(
        value
        for key, value in record.items()
        if key.endswith("_s") and key != "accounted_s"
    )
    assert record["accounted_s"] == pytest.approx(stage_total, abs=1e-4)


def test_the_wire_and_install_stages_are_present_and_positive(monkeypatch, caplog):
    _h, _m, _k = _run(monkeypatch, caplog)
    record = _record(caplog)
    assert "wire_fused_s" in record
    for stage in ("wire_fused_s", "reslice_s", "convert_s", "install_s"):
        assert record[stage] >= 0.0, stage


def test_phased_mode_reports_its_three_wire_stages_separately(monkeypatch, caplog):
    """Per-phase wire attribution is only recoverable with the fused wire off, which
    is the reason the phased path is kept."""
    _h, _m, _k = _run(monkeypatch, caplog, fused=False)
    record = _record(caplog)
    assert "wire_fused_s" not in record
    for stage in ("wire_exact_s", "wire_full_s", "wire_convert_s"):
        assert stage in record, stage
    assert record["fused_wire"] is False


def test_the_byte_economics_travel_with_the_timings(monkeypatch, caplog):
    """Useful bytes must be readable from the record. Deriving them analysis-side
    from an assumed sharding is what let an incomplete refit look faster."""
    _h, metrics, _k = _run(monkeypatch, caplog)
    record = _record(caplog)
    for field in (
        "bytes",
        "segments",
        "extra_wire_bytes",
        "descriptor_savings",
        "exact_descriptors",
        "full_pull_sources",
        "unbounded_sources",
        "converts",
        "fallback",
    ):
        assert field in record, field
    assert record["bytes"] == metrics["bytes_received"]
    assert record["segments"] == metrics["segments"]
    # Useful bytes are wire minus the redundancy the planner knowingly accepted.
    assert record["bytes"] - record["extra_wire_bytes"] >= 0


def test_the_stages_also_reach_the_returned_metrics(monkeypatch, caplog):
    """A caller holding the metrics dict should not have to parse the log line."""
    _h, metrics, _k = _run(monkeypatch, caplog)
    assert "install_s" in metrics
    assert metrics["install_s"] >= 0.0


def test_setup_costs_land_on_the_step_that_paid_them(monkeypatch, caplog):
    """A cold first refit runs discovery, the handshake, capture, planning and
    registration. Those belong in that step's record, or its unattributed remainder
    is the entire setup cost and the breakdown is useless exactly when it matters
    most."""
    monkeypatch.setenv("MX_REFIT_STAGE_RECORD", "1")
    monkeypatch.setenv("MX_RESHARD_FUSED_WIRE", "1")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    transport = _RecordingTransport()
    harness, keepalive = _build(transport)

    cold_plan = harness._plan
    harness._plan = None

    def _fake_prepare(timeout):
        harness._plan = cold_plan
        harness._prepare_stages = {
            "prepare_discover_s": 1.5,
            "prepare_handshake_s": 0.25,
            "prepare_capture_s": 0.5,
            "prepare_plan_s": 0.125,
            "prepare_alloc_s": 2.0,
            "prepare_register_s": 3.0,
        }

    harness._prepare = _fake_prepare
    with caplog.at_level(logging.WARNING):
        harness.update_weights(step=1)

    record = _record(caplog)
    for stage, expected in (
        ("prepare_discover_s", 1.5),
        ("prepare_handshake_s", 0.25),
        ("prepare_capture_s", 0.5),
        ("prepare_plan_s", 0.125),
        ("prepare_alloc_s", 2.0),
        ("prepare_register_s", 3.0),
    ):
        assert record[stage] == pytest.approx(expected), stage
    assert record["accounted_s"] >= 7.375
    assert all(t.data_ptr() for t in keepalive)


def test_a_warm_step_carries_no_setup_stages(monkeypatch, caplog):
    """The counterpart: setup is reported once, so a steady-state step is not
    inflated by costs it did not pay."""
    _h, _m, _k = _run(monkeypatch, caplog)
    record = _record(caplog)
    assert not [key for key in record if key.startswith("prepare_")]


def test_a_skipped_stage_is_absent_rather_than_zero(monkeypatch, caplog):
    """Zero would claim the stage ran and took no time. Absent says it did not run,
    which is what a plan with no converts means."""
    monkeypatch.setenv("MX_RESHARD_FUSED_WIRE", "1")
    monkeypatch.setenv("MX_REFIT_STAGE_RECORD", "1")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    transport = _RecordingTransport()
    harness, keepalive = _build(transport)
    harness._plan.converts = []
    harness._plan.full_pulls = []

    with caplog.at_level(logging.WARNING):
        harness.update_weights(step=1)

    record = _record(caplog)
    assert "convert_s" not in record
    assert "reslice_s" not in record
    assert "install_s" in record
    assert all(t.data_ptr() for t in keepalive)  # keep the source addresses valid
