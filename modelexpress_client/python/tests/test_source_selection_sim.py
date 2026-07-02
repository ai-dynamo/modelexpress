# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the load-aware source-selection simulation.

Guards the headline the benchmark exists to demonstrate: under a high-fan-out
burst, load_aware spreads targets more evenly and finishes sooner than the
load-blind baselines, and it is a no-op (identical to rendezvous_hash) when
there is no concurrency to react to.
"""

from __future__ import annotations

from benchmarks.source_selection_sim import run_policies, simulate


def test_load_aware_beats_rendezvous_under_high_fanout():
    # Small arrival gap vs service time -> heavy overlap -> contention on any
    # source that a load-blind policy over-picks. rendezvous_hash and
    # load_aware are deterministic, so single runs give exact assertions.
    results = run_policies(
        m_sources=4, n_targets=40, arrival_dt=0.5, service_time=10.0, trials=1
    )
    la = results["load_aware"]
    rh = results["rendezvous_hash"]

    # Balance: load_aware never concentrates more than the deterministic baseline.
    assert la.max_source_share <= rh.max_source_share
    assert la.cov <= rh.cov
    # Makespan: steering off busy sources shortens the fan-out (no regression).
    assert la.makespan <= rh.makespan

    # random shuffles with an unseeded local RNG, so any single trial is
    # nondeterministic; compare against its mean balance over several trials,
    # which concentrates well above load_aware's near-ideal spread.
    rnd_shares = [
        simulate(
            "random", m_sources=4, n_targets=40, arrival_dt=0.5, service_time=10.0
        ).max_source_share
        for _ in range(10)
    ]
    assert la.max_source_share <= sum(rnd_shares) / len(rnd_shares)


def test_load_aware_matches_rendezvous_without_concurrency():
    # Arrivals spaced far wider than the service time: every transfer finishes
    # before the next target arrives, so observed load is always 0 and load_aware
    # must reduce exactly to rendezvous_hash.
    la = simulate(
        "load_aware", m_sources=4, n_targets=12, arrival_dt=1000.0, service_time=1.0
    )
    rh = simulate(
        "rendezvous_hash",
        m_sources=4,
        n_targets=12,
        arrival_dt=1000.0,
        service_time=1.0,
    )
    assert la.selections == rh.selections


def test_simulate_conserves_targets():
    r = simulate(
        "load_aware", m_sources=4, n_targets=25, arrival_dt=0.5, service_time=5.0
    )
    assert r.n == 25
    assert len(r.durations) == 25
    assert r.makespan > 0
