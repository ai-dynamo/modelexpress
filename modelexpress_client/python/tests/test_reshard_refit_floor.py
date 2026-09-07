# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Throughput-floor guard: report a wire rate far below what the fabric can do.

The ceiling guard catches transfers that did not happen. Nothing caught a
transfer that happened 20x too slowly, and this is the incident these tests are
anchored to: four concurrent receivers on one node collapsed from ~26 GB/s each
to ~1.5 GB/s each, and every signal available said the refit was healthy. Byte
counts exact, descriptor counts exact, ``fallback`` 0, ``converts`` 0, coverage
100%, no error and no warning anywhere. It was found by a human dividing bytes by
seconds by hand.

A rate is only interpretable against a bound, and we only ever had the upper one.

The numbers below are that record verbatim, alongside the healthy single-reader
measurement from the same pods on the same day, because a guard that fires on
real good runs gets switched off and then protects nothing.

Run: pytest tests/test_reshard_refit_floor.py
"""

import json
import logging

import pytest
import torch

from tests.test_reshard_refit_fused_wire import _build, _RecordingTransport

# The observed incident, verbatim, from MX_STAGE_COST.
SLOW_BYTES = 15_889_231_872
SLOW_WIRE_S = 10.213842
SLOW_GBPS = 12.4  # bytes * 8 / s / 1e9

# The healthy measurement from the same pods: 7212 descriptors, 16.64 GB, 0.632 s.
FAST_BYTES = 16_637_117_592
FAST_WIRE_S = 0.631954

# A floor a fabric with 400 Gb/s rails should clear with room to spare, and which
# sits well above the incident and well below the healthy run.
FLOOR_GBPS = 50.0


def _receiver(monkeypatch, floor):
    """The receiver module with the floor configured.

    No module reload, for the same reason as the ceiling tests: the value is read
    at call time so a harness can set it per run.
    """
    if floor is None:
        monkeypatch.delenv("MX_RESHARD_MIN_GBPS", raising=False)
    else:
        monkeypatch.setenv("MX_RESHARD_MIN_GBPS", str(floor))
    from modelexpress.refit.reshard import receiver

    return receiver


class _Rig:
    _global_rank = 3


def _check(mod, wire_bytes, stages, step=1):
    return mod.ReshardReceiver._check_throughput_floor(_Rig(), step, wire_bytes, stages)


def _record(caplog):
    line = next(m for m in caplog.messages if "MX_REFIT_SLOW_THROUGHPUT" in m)
    return json.loads(line.split("MX_REFIT_SLOW_THROUGHPUT ", 1)[1])


def test_the_incident_is_reported(monkeypatch, caplog):
    """12.4 Gbps against a 50 Gbps floor: the run that produced no signal at all."""
    mod = _receiver(monkeypatch, FLOOR_GBPS)
    with caplog.at_level(logging.WARNING):
        _check(mod, SLOW_BYTES, {"wire_fused_s": SLOW_WIRE_S})

    payload = _record(caplog)
    assert payload["schema"] == "refit-slow-throughput-v1"
    assert payload["wire_bytes"] == SLOW_BYTES
    assert payload["floor_gbps"] == FLOOR_GBPS
    assert payload["implied_gbps"] == pytest.approx(SLOW_GBPS, abs=0.2)


def test_it_does_not_raise(monkeypatch, caplog):
    """A slow refit is still a correct refit.

    The ceiling aborts because an impossible rate means the buffers hold whatever
    was there before. Nothing is wrong with these bytes -- they just took too
    long -- and killing a training run over throughput would be worse than the
    throughput. Enforcement belongs in a CI gate reading the record.
    """
    mod = _receiver(monkeypatch, FLOOR_GBPS)
    with caplog.at_level(logging.WARNING):
        assert _check(mod, SLOW_BYTES, {"wire_fused_s": SLOW_WIRE_S}) is None


def test_the_healthy_run_is_silent(monkeypatch, caplog):
    """The other half of the guard. 16.64 GB in 0.632 s is 210 Gbps, and a floor
    that fires on that is a floor someone will turn off."""
    mod = _receiver(monkeypatch, FLOOR_GBPS)
    with caplog.at_level(logging.WARNING):
        _check(mod, FAST_BYTES, {"wire_fused_s": FAST_WIRE_S})

    assert not [m for m in caplog.messages if "MX_REFIT_SLOW_THROUGHPUT" in m]


def test_the_shortfall_is_quantified(monkeypatch, caplog):
    """ "Slightly under a conservative floor" and "an order of magnitude under" are
    different incidents, and a gate may reasonably treat them differently."""
    mod = _receiver(monkeypatch, FLOOR_GBPS)
    with caplog.at_level(logging.WARNING):
        _check(mod, SLOW_BYTES, {"wire_fused_s": SLOW_WIRE_S})

    assert _record(caplog)["shortfall_x"] == pytest.approx(
        FLOOR_GBPS / SLOW_GBPS, abs=0.1
    )


def test_a_rate_exactly_at_the_floor_is_allowed(monkeypatch, caplog):
    """The floor is meant to be attainable, so the comparison must not be strict."""
    mod = _receiver(monkeypatch, FLOOR_GBPS)
    exact_s = SLOW_BYTES * 8 / (FLOOR_GBPS * 1e9)
    with caplog.at_level(logging.WARNING):
        _check(mod, SLOW_BYTES, {"wire_fused_s": exact_s})

    assert not [m for m in caplog.messages if "MX_REFIT_SLOW_THROUGHPUT" in m]


def test_disabled_by_default(monkeypatch, caplog):
    """Only the operator knows what their fabric should deliver. Absent config the
    guard stays out of the way rather than guessing, exactly as the ceiling does."""
    mod = _receiver(monkeypatch, None)
    assert mod._min_gbps() == 0
    with caplog.at_level(logging.WARNING):
        _check(mod, SLOW_BYTES, {"wire_fused_s": SLOW_WIRE_S})

    assert not [m for m in caplog.messages if "MX_REFIT_SLOW_THROUGHPUT" in m]


@pytest.mark.parametrize("value", ["nan", "NaN", "inf", "-inf"])
def test_a_non_finite_floor_disables_rather_than_warning_on_every_refit(
    monkeypatch, caplog, value
):
    """Every comparison against NaN is False, so a fat-fingered floor would pass
    the "is it configured" test and then warn on every single refit -- turning the
    guard into noise, which is how guards get ignored."""
    mod = _receiver(monkeypatch, value)
    with caplog.at_level(logging.WARNING):
        _check(mod, SLOW_BYTES, {"wire_fused_s": SLOW_WIRE_S})

    assert not [m for m in caplog.messages if "MX_REFIT_SLOW_THROUGHPUT" in m]


@pytest.mark.parametrize("value", ["-50", "-0.5"])
def test_a_negative_floor_disables_but_says_so(monkeypatch, caplog, value):
    """A stray minus sign must not retire the gate in silence.

    A negative floor is finite, so it clears the check above, and then
    ``warn_if_below_floor`` treats anything at or below zero as off. The gate
    stops reporting while the operator who typed -50 believes one is in force,
    and in CI that reads as a green run rather than a missing signal. ``0`` is
    the documented off switch, so any other disabling value has to announce
    itself.
    """
    mod = _receiver(monkeypatch, value)
    with caplog.at_level(logging.WARNING):
        assert mod._min_gbps() == 0

    assert [m for m in caplog.messages if "is negative" in m], (
        f"a negative floor must warn, got {caplog.messages}"
    )


def test_an_unparseable_floor_disables_rather_than_crashes(monkeypatch):
    """A typo in a harness env var must not take down every refit."""
    mod = _receiver(monkeypatch, "not-a-number")
    assert mod._min_gbps() == 0
    assert _check(mod, SLOW_BYTES, {"wire_fused_s": SLOW_WIRE_S}) is None


def test_phased_mode_sums_its_three_wire_stages(monkeypatch, caplog):
    """With the fused wire off there is no wire_fused_s. Reading one phase alone
    would understate elapsed time and so manufacture a rate that clears the floor,
    which is the failure direction that matters for a lower bound."""
    mod = _receiver(monkeypatch, FLOOR_GBPS)
    phased = {"wire_exact_s": 4.0, "wire_full_s": 4.0, "wire_convert_s": 2.213842}
    with caplog.at_level(logging.WARNING):
        _check(mod, SLOW_BYTES, phased)

    assert _record(caplog)["wire_s"] == pytest.approx(SLOW_WIRE_S, abs=1e-5)


def test_a_missing_or_zero_wire_time_is_not_a_zero_rate(monkeypatch, caplog):
    """A refit with nothing planned must not divide by zero or report itself slow."""
    mod = _receiver(monkeypatch, FLOOR_GBPS)
    with caplog.at_level(logging.WARNING):
        assert _check(mod, SLOW_BYTES, {}) is None
        assert _check(mod, SLOW_BYTES, {"wire_fused_s": 0.0}) is None
        assert _check(mod, 0, {"wire_fused_s": 0.5}) is None

    assert not [m for m in caplog.messages if "MX_REFIT_SLOW_THROUGHPUT" in m]


def test_the_report_names_the_rank_that_saw_it(monkeypatch, caplog):
    """The incident had one fast rank and three slow ones in the same run, so a
    record that cannot be attributed to a rank cannot describe it."""
    mod = _receiver(monkeypatch, FLOOR_GBPS)
    with caplog.at_level(logging.WARNING):
        _check(mod, SLOW_BYTES, {"wire_fused_s": SLOW_WIRE_S})

    assert _record(caplog)["rank"] == 3


# ------------------------------------------- where the guard sits in the refit
def _refit(monkeypatch, floor, stage_record):
    """A whole refit through the real update_weights, with the floor configured."""
    monkeypatch.setenv("MX_RESHARD_FUSED_WIRE", "1")
    monkeypatch.setenv("MX_RESHARD_MIN_GBPS", str(floor))
    monkeypatch.delenv("MX_RESHARD_MAX_GBPS", raising=False)
    monkeypatch.setenv("MX_REFIT_STAGE_RECORD", stage_record)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    return _build(_RecordingTransport())


def test_a_breached_floor_still_installs(monkeypatch, caplog):
    """The defining difference from the ceiling, exercised through the real path.

    A breached ceiling must block the install, because an impossible rate means
    the receive buffers were never filled. A breached floor must not, because the
    bytes are correct and only slow. A guard that halted training over throughput
    would cost more than the throughput it was reporting.
    """
    # High enough that any real rate is below it, so the test does not depend on
    # how fast the in-memory transport happens to be.
    harness, keepalive = _refit(monkeypatch, 1e9, "1")

    with caplog.at_level(logging.WARNING):
        harness.update_weights(step=1)

    assert harness._install_order == ["install"], (
        "a slow refit is still a correct refit and must reach live parameters"
    )
    assert [m for m in caplog.messages if "MX_REFIT_SLOW_THROUGHPUT" in m]
    assert keepalive is not None


def test_the_floor_does_not_depend_on_the_stage_record(monkeypatch, caplog):
    """``MX_REFIT_STAGE_RECORD=0`` must not silence the floor.

    The floor is emitted next to the stage record, which is a natural place to
    nest it inside the same flag by accident. Doing so would mean anyone who
    turned the per-refit timings off - a reasonable thing to do in production -
    would also lose the only signal that reports a collapsed transfer, and would
    never know it had gone.
    """
    harness, keepalive = _refit(monkeypatch, 1e9, "0")

    with caplog.at_level(logging.WARNING):
        harness.update_weights(step=1)

    assert not [m for m in caplog.messages if "MX_REFIT_STAGE " in m], (
        "precondition: the stage record is off for this run"
    )
    assert [m for m in caplog.messages if "MX_REFIT_SLOW_THROUGHPUT" in m], (
        "the floor must report independently of the stage record"
    )
    assert keepalive is not None


def test_a_healthy_refit_through_the_real_path_is_silent(monkeypatch, caplog):
    """The other half: a floor low enough to clear must produce nothing."""
    harness, keepalive = _refit(monkeypatch, 0.000001, "1")

    with caplog.at_level(logging.WARNING):
        harness.update_weights(step=1)

    assert harness._install_order == ["install"]
    assert not [m for m in caplog.messages if "MX_REFIT_SLOW_THROUGHPUT" in m]
    assert keepalive is not None


def test_floor_and_ceiling_can_both_be_configured(monkeypatch, caplog):
    """They bound the same quantity from opposite sides and must not interfere. A
    rate between them is the healthy case and should produce neither record."""
    monkeypatch.setenv("MX_RESHARD_MIN_GBPS", str(FLOOR_GBPS))
    monkeypatch.setenv("MX_RESHARD_MAX_GBPS", "400")
    from modelexpress.refit.reshard import receiver

    with caplog.at_level(logging.WARNING):
        receiver.ReshardReceiver._check_throughput_floor(
            _Rig(), 1, FAST_BYTES, {"wire_fused_s": FAST_WIRE_S}
        )
        receiver.ReshardReceiver._check_throughput_ceiling(
            _Rig(), 1, FAST_BYTES, {"wire_fused_s": FAST_WIRE_S}
        )

    assert not [
        m
        for m in caplog.messages
        if "MX_REFIT_SLOW_THROUGHPUT" in m or "MX_REFIT_IMPOSSIBLE_THROUGHPUT" in m
    ]
