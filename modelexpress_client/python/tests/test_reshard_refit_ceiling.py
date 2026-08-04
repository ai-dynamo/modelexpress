# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Throughput-ceiling guard: refuse a wire rate the fabric cannot produce.

The incident these tests are anchored to: a receiver holding two of its node's
four network adapters reported 40.61 GB delivered in 0.84 s, an implied 387 Gbps
against a two-adapter ceiling of about 191 Gbps, and delivered nothing. Every
other signal called the run healthy - coverage 100%, fallback 0, addresses and
digests stable - and only the parameter-equality gate dissented, which timing
runs switch off by design.

The numbers below are that record verbatim, so the regression is pinned to
something that happened rather than to a constructed example. The genuine
Topology B run is pinned alongside it, because a guard that rejects real
measurements gets switched off and then protects nothing.

Run: pytest tests/test_reshard_refit_ceiling.py
"""

import json
import logging

import pytest
import torch

from tests.test_reshard_refit_fused_wire import _build, _RecordingTransport

# The observed record, verbatim.
OBSERVED_BYTES = 40_611_246_080
OBSERVED_WIRE_S = 0.839641
OBSERVED_IMPLIED_GBPS = 386.9  # bytes * 8 / s / 1e9
FOUR_ADAPTER_CEILING = 381.6
TWO_ADAPTER_CEILING = 190.8


def _receiver(monkeypatch, ceiling):
    """The receiver module with the ceiling configured.

    No module reload: the ceiling is read at call time precisely so a harness can
    set it per run, and a test that needed a reload would be testing something the
    product does not do.
    """
    if ceiling is None:
        monkeypatch.delenv("MX_RESHARD_MAX_GBPS", raising=False)
    else:
        monkeypatch.setenv("MX_RESHARD_MAX_GBPS", str(ceiling))
    from modelexpress.refit.reshard import receiver

    return receiver


class _Rig:
    """Just enough object to call the guard as an unbound method."""


def _check(mod, wire_bytes, stages, step=1):
    return mod.ReshardReceiver._check_throughput_ceiling(
        _Rig(), step, wire_bytes, stages
    )


def test_the_impossible_run_is_rejected(monkeypatch):
    """387 Gbps on a two-adapter pod must abort, not become the best row."""
    mod = _receiver(monkeypatch, TWO_ADAPTER_CEILING)
    with pytest.raises(RuntimeError) as excinfo:
        _check(mod, OBSERVED_BYTES, {"wire_fused_s": OBSERVED_WIRE_S})
    message = str(excinfo.value)
    assert "386.9" in message
    assert "without moving the payload" in message
    assert "failed, not fast" in message


def test_it_would_have_been_caught_even_at_the_four_adapter_ceiling(monkeypatch):
    """The rate was impossible even for a fully-provisioned pod, which is what makes
    it diagnosable from the stage record alone."""
    mod = _receiver(monkeypatch, FOUR_ADAPTER_CEILING)
    with pytest.raises(RuntimeError):
        _check(mod, OBSERVED_BYTES, {"wire_fused_s": OBSERVED_WIRE_S})


def test_the_real_topology_b_run_passes(monkeypatch):
    """Same byte count, 1.273 s, 255.3 Gbps on four adapters: a genuine measurement
    that must not be rejected."""
    mod = _receiver(monkeypatch, FOUR_ADAPTER_CEILING)
    assert _check(mod, OBSERVED_BYTES, {"wire_fused_s": 1.273}) is None


def test_a_rate_exactly_at_the_ceiling_is_allowed(monkeypatch):
    """The ceiling is attainable in principle, so the comparison must not be strict."""
    mod = _receiver(monkeypatch, FOUR_ADAPTER_CEILING)
    exact_s = OBSERVED_BYTES * 8 / (FOUR_ADAPTER_CEILING * 1e9)
    assert _check(mod, OBSERVED_BYTES, {"wire_fused_s": exact_s}) is None


def test_disabled_by_default(monkeypatch):
    """Only the operator knows their fabric ceiling, so absent config the guard must
    stay out of the way rather than guess one."""
    mod = _receiver(monkeypatch, None)
    assert mod._max_gbps() == 0
    assert _check(mod, OBSERVED_BYTES, {"wire_fused_s": OBSERVED_WIRE_S}) is None


def test_an_unparseable_ceiling_disables_rather_than_crashes(monkeypatch):
    """A typo in a harness env var must not take down every refit."""
    mod = _receiver(monkeypatch, "not-a-number")
    assert mod._max_gbps() == 0
    assert _check(mod, OBSERVED_BYTES, {"wire_fused_s": OBSERVED_WIRE_S}) is None


def test_phased_mode_sums_its_three_wire_stages(monkeypatch):
    """With the fused wire off there is no wire_fused_s, and reading one phase alone
    would understate elapsed time and manufacture an impossible rate."""
    mod = _receiver(monkeypatch, FOUR_ADAPTER_CEILING)
    phased = {"wire_exact_s": 0.5, "wire_full_s": 0.5, "wire_convert_s": 0.273}
    assert _check(mod, OBSERVED_BYTES, phased) is None

    too_fast = {"wire_exact_s": 0.3, "wire_full_s": 0.3, "wire_convert_s": 0.2}
    with pytest.raises(RuntimeError):
        _check(mod, OBSERVED_BYTES, too_fast)


def test_a_missing_or_zero_wire_time_is_not_an_infinite_rate(monkeypatch):
    """A refit with nothing planned must not divide by zero or trip the guard."""
    mod = _receiver(monkeypatch, FOUR_ADAPTER_CEILING)
    assert _check(mod, OBSERVED_BYTES, {}) is None
    assert _check(mod, OBSERVED_BYTES, {"wire_fused_s": 0.0}) is None
    assert _check(mod, 0, {"wire_fused_s": 0.5}) is None


def test_the_failure_is_machine_readable_before_it_raises(monkeypatch, caplog):
    """The record has to survive the abort, or the guard destroys the evidence that
    explains it."""
    mod = _receiver(monkeypatch, TWO_ADAPTER_CEILING)
    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError):
        _check(mod, OBSERVED_BYTES, {"wire_fused_s": OBSERVED_WIRE_S})

    line = next(m for m in caplog.messages if "MX_REFIT_IMPOSSIBLE_THROUGHPUT" in m)
    payload = json.loads(line.split("MX_REFIT_IMPOSSIBLE_THROUGHPUT ", 1)[1])
    assert payload["schema"] == "refit-impossible-throughput-v1"
    assert payload["wire_bytes"] == OBSERVED_BYTES
    assert payload["ceiling_gbps"] == TWO_ADAPTER_CEILING
    assert payload["implied_gbps"] == pytest.approx(OBSERVED_IMPLIED_GBPS, abs=0.2)


# ------------------------------------------- where the guard sits in the refit
def _refit(monkeypatch, ceiling):
    """A whole refit through the real update_weights, with the ceiling configured."""
    monkeypatch.setenv("MX_RESHARD_FUSED_WIRE", "1")
    monkeypatch.setenv("MX_RESHARD_MAX_GBPS", str(ceiling))
    monkeypatch.delenv("MX_REFIT_STAGE_RECORD", raising=False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    return _build(_RecordingTransport())


def test_a_breached_ceiling_blocks_the_install(monkeypatch):
    """The point of the guard, and the thing calling it after ``_install`` silently
    gave up.

    An impossible rate means the transport reported completions it did not earn, so
    the receive buffers hold whatever was there before. Raising after the install
    documents that the live parameters were overwritten with untrustworthy bytes;
    raising before it is what actually prevents them from being.
    """
    # Low enough that any real rate breaches it, so the test does not depend on
    # how fast the in-memory transport happens to be.
    harness, keepalive = _refit(monkeypatch, 0.000001)

    with pytest.raises(RuntimeError, match="failed, not fast"):
        harness.update_weights(step=1)

    assert harness._install_order == [], (
        "the ceiling breached, so nothing should have reached live parameters"
    )
    assert keepalive is not None


def test_a_rate_under_the_ceiling_still_installs(monkeypatch):
    """The other half: a guard that blocks legitimate refits gets switched off, and
    then protects nothing."""
    harness, keepalive = _refit(monkeypatch, 1e9)

    harness.update_weights(step=1)

    assert harness._install_order == ["install"]
    assert keepalive is not None
