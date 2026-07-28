# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Throughput-ceiling guard: refuse a wire rate the fabric cannot produce.

The guard exists because a transport can report completions it never earned, and a
refit that "finishes" implausibly fast is evidence of that rather than a good result.

Calibration is the whole difficulty, and we got it wrong first time. The ceiling was
set to 381.6 Gbps - one 400 Gb/s EFA adapter derated to 95.4% - and on 2026-07-27 it
aborted a healthy refit measured at 386.5 Gbps, because the rank had two of its node's
four adapters available to it. The hardware is a p6e-gb200.36xlarge: four adapters at
400 Gb/s, 1600 Gbps per node, consistent with the 14,400 Gbps quoted for a 36-GPU
UltraServer. So the ceiling must be a bound a rank cannot exceed however the adapters
are shared, not an expected rate.

That correction also retires the guard's original claim. It was written believing it
would have caught Bug 10 - 40.61 GB in 0.84 s, 386.9 Gbps, delivering nothing - but at
a correct ceiling that rate is legal. Silent no-op reads are caught by the
parameter-equality gate. This guard catches only a transport that beats the whole node,
and these tests are written to keep both facts pinned.

Run: pytest tests/test_reshard_refit_ceiling.py
"""

import importlib

import pytest


# The observed Bug 10 record, verbatim.
BUG10_BYTES = 40_611_246_080
BUG10_WIRE_S = 0.839641
BUG10_IMPLIED_GBPS = 386.9  # bytes * 8 / s / 1e9

# p6e-gb200.36xlarge: 4 EFA adapters at 400 Gb/s.
ADAPTER_GBPS = 400.0
NODE_CEILING = 4 * ADAPTER_GBPS  # 1600

# The miscalibration that cost a run, kept so the regression stays anchored.
BAD_CEILING = 381.6
V40_HEALTHY_WIRE_S = 0.840649  # 386.5 Gbps, verified correct by the gate


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


def test_a_rate_above_the_configured_ceiling_aborts(monkeypatch):
    """The mechanism: whatever bound the operator sets is enforced, and the message
    names the rate so the abort explains itself."""
    mod = _receiver(monkeypatch, 200.0)
    with pytest.raises(RuntimeError) as excinfo:
        _check(mod, 200.0, BUG10_BYTES, {"wire_fused_s": BUG10_WIRE_S})
    message = str(excinfo.value)
    assert "386.9" in message or "387" in message
    assert "did not transfer" in message or "without delivering" in message


def test_the_healthy_386_gbps_refit_must_not_be_rejected(monkeypatch):
    """The v40 false positive, pinned.

    386.5 Gbps was a real transfer - the parameter-equality gate checked 6192 sources
    and found zero mismatches, coverage was 100%, and the sentinel fill showed every
    buffer overwritten. A guard that aborts this is worse than no guard, because the
    run it kills looks exactly like the one it is meant to catch.
    """
    mod = _receiver(monkeypatch, NODE_CEILING)
    assert (
        _check(mod, NODE_CEILING, BUG10_BYTES, {"wire_fused_s": V40_HEALTHY_WIRE_S})
        is None
    )


def test_the_old_ceiling_is_what_made_that_a_false_positive(monkeypatch):
    """Kept as the counterexample: at 381.6 the same healthy run aborts. This is why
    the ceiling has to be a hard bound rather than a derated single adapter."""
    mod = _receiver(monkeypatch, BAD_CEILING)
    with pytest.raises(RuntimeError):
        _check(mod, BAD_CEILING, BUG10_BYTES, {"wire_fused_s": V40_HEALTHY_WIRE_S})


def test_bug_10_would_not_have_been_caught_at_the_true_ceiling(monkeypatch):
    """The guard's original justification, disproven.

    Bug 10 delivered nothing at 386.9 Gbps, which is legal across two 400 Gb/s
    adapters. Arithmetic cannot separate that run from a good one; only the gate can.
    Asserting this keeps anyone from citing the guard as a defence against silent
    no-op reads.
    """
    mod = _receiver(monkeypatch, NODE_CEILING)
    assert (
        _check(mod, NODE_CEILING, BUG10_BYTES, {"wire_fused_s": BUG10_WIRE_S}) is None
    )


def test_the_real_topology_b_run_passes(monkeypatch):
    """Same byte count, 1.273 s, 255.3 Gbps: a genuine measurement that must not be
    rejected. Guards that fail good runs get switched off."""
    mod = _receiver(monkeypatch, NODE_CEILING)
    assert _check(mod, NODE_CEILING, BUG10_BYTES, {"wire_fused_s": 1.273}) is None


def test_a_rate_exactly_at_the_ceiling_is_allowed(monkeypatch):
    """The ceiling is attainable in principle, so the comparison must not be strict."""
    mod = _receiver(monkeypatch, NODE_CEILING)
    exact_s = BUG10_BYTES * 8 / (NODE_CEILING * 1e9)
    assert _check(mod, NODE_CEILING, BUG10_BYTES, {"wire_fused_s": exact_s}) is None


def test_a_rate_beyond_the_whole_node_still_aborts(monkeypatch):
    """What the guard is actually for: a completion path so fast it beats every
    adapter on the box at once. 0.05 s for 40.61 GB is 6498 Gbps."""
    mod = _receiver(monkeypatch, NODE_CEILING)
    with pytest.raises(RuntimeError):
        _check(mod, NODE_CEILING, BUG10_BYTES, {"wire_fused_s": 0.05})


def test_disabled_by_default(monkeypatch):
    """Nobody knows the fabric ceiling but the operator, so absent config the guard
    must stay out of the way."""
    mod = _receiver(monkeypatch, None)
    assert mod._MAX_GBPS == 0
    assert _check(mod, 0, BUG10_BYTES, {"wire_fused_s": BUG10_WIRE_S}) is None


def test_phased_mode_sums_its_three_wire_stages(monkeypatch):
    """Without _FUSED_WIRE there is no wire_fused_s, and reading one phase alone
    would understate the elapsed time and manufacture an impossible rate."""
    mod = _receiver(monkeypatch, NODE_CEILING)
    phased = {"wire_exact_s": 0.5, "wire_full_s": 0.5, "wire_convert_s": 0.273}
    assert _check(mod, NODE_CEILING, BUG10_BYTES, phased) is None

    too_fast = {"wire_exact_s": 0.06, "wire_full_s": 0.06, "wire_convert_s": 0.06}
    with pytest.raises(RuntimeError):
        _check(mod, NODE_CEILING, BUG10_BYTES, too_fast)


def test_a_missing_or_zero_wire_time_is_not_an_infinite_rate(monkeypatch):
    """A refit with nothing planned must not divide by zero or trip the guard."""
    mod = _receiver(monkeypatch, NODE_CEILING)
    assert _check(mod, NODE_CEILING, BUG10_BYTES, {}) is None
    assert _check(mod, NODE_CEILING, BUG10_BYTES, {"wire_fused_s": 0.0}) is None
    assert _check(mod, NODE_CEILING, 0, {"wire_fused_s": 0.5}) is None


def test_the_failure_is_machine_readable_before_it_raises(monkeypatch, caplog):
    """The record has to survive the abort, or the guard destroys the evidence that
    explains it."""
    mod = _receiver(monkeypatch, 200.0)
    import json
    import logging

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError):
            _check(mod, 200.0, BUG10_BYTES, {"wire_fused_s": BUG10_WIRE_S})

    line = next(
        m for m in caplog.messages if "MX_REFIT_IMPOSSIBLE_THROUGHPUT" in m
    )
    payload = json.loads(line.split("MX_REFIT_IMPOSSIBLE_THROUGHPUT ", 1)[1])
    assert payload["schema"] == "refit-impossible-throughput-v1"
    assert payload["wire_bytes"] == BUG10_BYTES
    assert payload["ceiling_gbps"] == 200.0
    assert payload["implied_gbps"] == pytest.approx(BUG10_IMPLIED_GBPS, abs=0.2)
