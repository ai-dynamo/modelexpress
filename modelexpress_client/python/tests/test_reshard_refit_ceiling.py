# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Throughput-ceiling guard: refuse a wire rate the fabric cannot produce.

Bug 10 delivered zero bytes and reported the fastest refit we had recorded - 40.61 GB
in 0.84 s, 387 Gbps, on a pod holding two EFAs worth ~191 Gbps. Every other signal
called it healthy: coverage 100%, fallback 0, addresses and digests stable. Only the
parameter-equality gate dissented, and timing runs switch that off by design.

These tests pin the arithmetic and the on/off behaviour, using the real Bug 10
numbers so the regression is anchored to something that actually happened.

Run: pytest tests/test_reshard_refit_ceiling.py
"""

import importlib

import pytest


# The observed Bug 10 record, verbatim.
BUG10_BYTES = 40_611_246_080
BUG10_WIRE_S = 0.839641
BUG10_IMPLIED_GBPS = 386.9  # bytes * 8 / s / 1e9
FOUR_EFA_CEILING = 381.6
TWO_EFA_CEILING = 190.8


def _receiver(monkeypatch, ceiling):
    """Reimport the module so the module-level ceiling is re-read."""
    if ceiling is None:
        monkeypatch.delenv("MX_RESHARD_MAX_GBPS", raising=False)
    else:
        monkeypatch.setenv("MX_RESHARD_MAX_GBPS", str(ceiling))
    import modelexpress.refit.reshard.receiver as receiver

    return importlib.reload(receiver)


class _Rig:
    """Just enough object to call the guard as an unbound method."""


def _check(mod, ceiling, wire_bytes, stages, step=1):
    return mod.ReshardReceiver._check_throughput_ceiling(
        _Rig(), step, wire_bytes, stages
    )


def test_the_bug_10_run_is_rejected(monkeypatch):
    """387 Gbps on a 2-EFA pod must abort, not become the best row in the matrix."""
    mod = _receiver(monkeypatch, TWO_EFA_CEILING)
    with pytest.raises(RuntimeError) as excinfo:
        _check(mod, TWO_EFA_CEILING, BUG10_BYTES, {"wire_fused_s": BUG10_WIRE_S})
    message = str(excinfo.value)
    assert "386.9" in message or "387" in message
    assert "did not transfer" in message or "without delivering" in message


def test_bug_10_would_have_been_caught_even_at_the_four_efa_ceiling(monkeypatch):
    """The rate was impossible even for a fully-provisioned pod, which is what made
    it diagnosable from the record alone."""
    mod = _receiver(monkeypatch, FOUR_EFA_CEILING)
    with pytest.raises(RuntimeError):
        _check(mod, FOUR_EFA_CEILING, BUG10_BYTES, {"wire_fused_s": BUG10_WIRE_S})


def test_the_real_topology_b_run_passes(monkeypatch):
    """Same byte count, 1.273 s, 255.3 Gbps on 4 EFAs: a genuine measurement that
    must not be rejected. Guards that fail good runs get switched off."""
    mod = _receiver(monkeypatch, FOUR_EFA_CEILING)
    assert _check(mod, FOUR_EFA_CEILING, BUG10_BYTES, {"wire_fused_s": 1.273}) is None


def test_a_rate_exactly_at_the_ceiling_is_allowed(monkeypatch):
    """The ceiling is attainable in principle, so the comparison must not be strict."""
    mod = _receiver(monkeypatch, FOUR_EFA_CEILING)
    exact_s = BUG10_BYTES * 8 / (FOUR_EFA_CEILING * 1e9)
    assert _check(mod, FOUR_EFA_CEILING, BUG10_BYTES, {"wire_fused_s": exact_s}) is None


def test_disabled_by_default(monkeypatch):
    """Nobody knows the fabric ceiling but the operator, so absent config the guard
    must stay out of the way."""
    mod = _receiver(monkeypatch, None)
    assert mod._MAX_GBPS == 0
    assert _check(mod, 0, BUG10_BYTES, {"wire_fused_s": BUG10_WIRE_S}) is None


def test_phased_mode_sums_its_three_wire_stages(monkeypatch):
    """Without _FUSED_WIRE there is no wire_fused_s, and reading one phase alone
    would understate the elapsed time and manufacture an impossible rate."""
    mod = _receiver(monkeypatch, FOUR_EFA_CEILING)
    phased = {"wire_exact_s": 0.5, "wire_full_s": 0.5, "wire_convert_s": 0.273}
    assert _check(mod, FOUR_EFA_CEILING, BUG10_BYTES, phased) is None

    too_fast = {"wire_exact_s": 0.3, "wire_full_s": 0.3, "wire_convert_s": 0.2}
    with pytest.raises(RuntimeError):
        _check(mod, FOUR_EFA_CEILING, BUG10_BYTES, too_fast)


def test_a_missing_or_zero_wire_time_is_not_an_infinite_rate(monkeypatch):
    """A refit with nothing planned must not divide by zero or trip the guard."""
    mod = _receiver(monkeypatch, FOUR_EFA_CEILING)
    assert _check(mod, FOUR_EFA_CEILING, BUG10_BYTES, {}) is None
    assert _check(mod, FOUR_EFA_CEILING, BUG10_BYTES, {"wire_fused_s": 0.0}) is None
    assert _check(mod, FOUR_EFA_CEILING, 0, {"wire_fused_s": 0.5}) is None


def test_the_failure_is_machine_readable_before_it_raises(monkeypatch, caplog):
    """The record has to survive the abort, or the guard destroys the evidence that
    explains it."""
    mod = _receiver(monkeypatch, TWO_EFA_CEILING)
    import json
    import logging

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError):
            _check(mod, TWO_EFA_CEILING, BUG10_BYTES, {"wire_fused_s": BUG10_WIRE_S})

    line = next(
        m for m in caplog.messages if "MX_REFIT_IMPOSSIBLE_THROUGHPUT" in m
    )
    payload = json.loads(line.split("MX_REFIT_IMPOSSIBLE_THROUGHPUT ", 1)[1])
    assert payload["schema"] == "refit-impossible-throughput-v1"
    assert payload["wire_bytes"] == BUG10_BYTES
    assert payload["ceiling_gbps"] == TWO_EFA_CEILING
    assert payload["implied_gbps"] == pytest.approx(BUG10_IMPLIED_GBPS, abs=0.2)
