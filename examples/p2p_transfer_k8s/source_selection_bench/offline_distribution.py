#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline source-selection distribution benchmark.

Measures, directly from the real selector code in
``modelexpress.source_selection``, the properties that distinguish the Phase 1
policies. These signals are deterministic in the policy and the
(target, source) identities and do not depend on GPU hardware or RDMA, so they
reproduce the on-cluster ``mx_p2p_source_selections_total`` distribution
without standing up real transfers. The on-cluster benchmark adds the secondary
signals (makespan, transfer-time tails) that this cannot.

Three things are reported per (M sources, N targets) config:

  1. Balance        -- max-source share and CoV of first-choice load, averaged
                       fairly (random over seeds, rendezvous over source-set
                       relabelings). Lower = more even.
  2. Determinism    -- fraction of targets that pick a DIFFERENT source on a
                       second, independent invocation with the SAME source set.
  3. Disruption     -- fraction of targets whose pick changes when ONE source is
                       removed from the set.

Run:  python offline_distribution.py
"""

from __future__ import annotations

import statistics
from collections import Counter

from modelexpress import p2p_pb2
from modelexpress.source_selection import (
    RandomSelector,
    RendezvousHashSelector,
    SourceSelectionContext,
)

TRIALS = 32


def _sources(m: int, salt: int = 0) -> list[p2p_pb2.SourceInstanceRef]:
    return [
        p2p_pb2.SourceInstanceRef(
            mx_source_id=f"{(i + 1) * (salt * 1009 + 1):016x}"[:16],
            worker_id=f"mx-source-{salt}-w{i}",
            worker_rank=0,
            model_name="bench-model",
        )
        for i in range(m)
    ]


def _ctx(t: int, seed: int | None = None) -> SourceSelectionContext:
    return SourceSelectionContext(
        worker_rank=0, global_rank=t, worker_id=f"mx-target-{t}",
        model_name="bench-model", selector_seed=seed,
    )


def _first_choice(selector, sources, n, seed_base=None):
    out = []
    for t in range(n):
        seed = None if seed_base is None else seed_base * 100_003 + t
        out.append(selector.order(sources, _ctx(t, seed))[0].worker_id)
    return out


def _balance(picks, m):
    n = len(picks)
    counts = Counter(picks)
    per_source = [counts.get(k, 0) for k in {p for p in picks}] + [0] * (m - len(counts))
    mean = n / m
    return max(per_source) / n, (statistics.pstdev(per_source) / mean if mean else 0)


def run_config(m, n):
    rnd = RandomSelector()
    rdv = RendezvousHashSelector()

    # 1. Balance, averaged fairly.
    rnd_bal = [_balance(_first_choice(rnd, _sources(m), n, seed_base=k), m) for k in range(TRIALS)]
    rdv_bal = [_balance(_first_choice(rdv, _sources(m, salt=k), n), m) for k in range(TRIALS)]
    rnd_share = statistics.mean(b[0] for b in rnd_bal)
    rnd_cov = statistics.mean(b[1] for b in rnd_bal)
    rdv_share = statistics.mean(b[0] for b in rdv_bal)
    rdv_cov = statistics.mean(b[1] for b in rdv_bal)

    # 2. Determinism: second independent invocation, same source set.
    src = _sources(m)
    rnd_a = _first_choice(rnd, src, n, seed_base=1)
    rnd_b = _first_choice(rnd, src, n, seed_base=2)
    rnd_churn = sum(a != b for a, b in zip(rnd_a, rnd_b)) / n
    rdv_a = _first_choice(rdv, src, n)
    rdv_b = _first_choice(rdv, src, n)
    rdv_churn = sum(a != b for a, b in zip(rdv_a, rdv_b)) / n

    # 3. Disruption: remove one source, fraction of picks that change.
    src_minus = src[:-1]
    removed = src[-1].worker_id
    rdv_before = _first_choice(rdv, src, n)
    rdv_after = _first_choice(rdv, src_minus, n)
    # targets that were NOT on the removed source and still changed = pure disruption
    rdv_disrupt = sum(
        b != a for a, b in zip(rdv_before, rdv_after) if a != removed
    ) / max(1, sum(1 for a in rdv_before if a != removed))

    return {
        "rnd_share": rnd_share, "rnd_cov": rnd_cov,
        "rdv_share": rdv_share, "rdv_cov": rdv_cov,
        "ideal_share": (n // m + (1 if n % m else 0)) / n,
        "rnd_churn": rnd_churn, "rdv_churn": rdv_churn,
        "rdv_disrupt": rdv_disrupt,
    }


def main():
    print("Offline source-selection benchmark (real selector code)\n")
    hdr = f"{'M,N':>7} {'fan':>4} | {'max-share (ideal)':>22} | {'CoV':>13} | {'re-pick churn':>20} | {'1-src disrupt':>13}"
    print(hdr)
    print(f"{'':>7} {'':>4} | {'rand   rdv':>22} | {'rand   rdv':>13} | {'rand    rdv':>20} | {'rdv':>13}")
    print("-" * len(hdr))
    for m, n in [(4, 20), (4, 24), (4, 40), (8, 32)]:
        r = run_config(m, n)
        print(
            f"{f'{m},{n}':>7} {n / m:>4.0f} | "
            f"{r['rnd_share']:.3f}  {r['rdv_share']:.3f} ({r['ideal_share']:.3f})   | "
            f"{r['rnd_cov']:.3f} {r['rdv_cov']:.3f} | "
            f"{r['rnd_churn'] * 100:5.0f}%  {r['rdv_churn'] * 100:4.0f}%        | "
            f"{r['rdv_disrupt'] * 100:5.1f}%"
        )
    print("\nReading: balance is ~equal (both uniform-hash); rendezvous_hash's win is")
    print("determinism (0% re-pick churn vs random's ~(M-1)/M) and minimal disruption")
    print("when the source set changes (only ~1/M of unaffected targets move).")


if __name__ == "__main__":
    main()
