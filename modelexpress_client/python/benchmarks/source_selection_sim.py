# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load-aware source-selection fan-out simulation.

Drives the *real* selector code (``modelexpress.source_selection``) over a
processor-sharing model of many targets pulling weights from a fixed pool of
pre-warmed sources, and compares ``random`` / ``rendezvous_hash`` /
``load_aware`` on source-utilization balance and fan-out makespan.

Why a timed simulation rather than the identity-only offline sim used for the
Phase-1 policies (PR #461): ``random`` and ``rendezvous_hash`` depend only on
identity, so their distribution can be reproduced without standing up
transfers. ``load_aware`` instead reacts to *live* per-source load, so showing
its benefit requires modeling transfers that occupy sources over time and a
target that observes the load left by earlier arrivals.

Model
-----
- ``M`` sources are all present for the whole run (pre-warmed pool). A source
  serving ``k`` concurrent transfers splits its egress bandwidth evenly, so each
  in-flight transfer progresses at rate ``1/k`` -- a processor-sharing queue.
  This is what makes piling many targets onto one source slow: contention
  stretches every transfer that source is serving.
- ``N`` targets arrive one every ``arrival_dt`` seconds (a staggered fan-out /
  RL-style repeated weight sync). On arrival a target ranks the sources with the
  configured selector, given each source's *current* in-flight count as
  ``active_transfers``, and starts a transfer on the winner.
- The signal a target sees is exactly what the server-side load tracker
  estimates on a real cluster: how many transfers a source is currently serving.

Reported per policy: source-utilization balance (max-source share and the
coefficient of variation across sources) and fan-out makespan (wall-clock for
all N to finish), plus per-target transfer-time tails.
"""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass
from types import SimpleNamespace

from modelexpress import p2p_pb2
from modelexpress.source_selection import get_selector

_MODEL = "bench-model"
_EPS = 1e-9


def _ctx(target_idx: int) -> SimpleNamespace:
    """Duck-typed LoadContext: selectors read only these fields."""
    return SimpleNamespace(
        worker_rank=0,
        global_rank=target_idx,
        worker_id=f"target-{target_idx}",
        identity=SimpleNamespace(model_name=_MODEL),
    )


def _candidates(
    source_ids: list[str], inflight: dict[str, int]
) -> list[p2p_pb2.SourceInstanceRef]:
    """Build the candidate list a target would receive from ListSources, with
    each source's current in-flight count surfaced as active_transfers."""
    return [
        p2p_pb2.SourceInstanceRef(
            mx_source_id=sid,
            worker_id=sid,
            model_name=_MODEL,
            worker_rank=0,
            active_transfers=inflight[sid],
        )
        for sid in source_ids
    ]


@dataclass
class SimResult:
    policy: str
    selections: dict[str, int]  # source_id -> targets served
    makespan: float
    durations: list[float]  # per-target transfer wall-clock

    @property
    def n(self) -> int:
        return sum(self.selections.values())

    @property
    def max_source_share(self) -> float:
        """Fraction of targets that landed on the single busiest source."""
        return max(self.selections.values()) / self.n if self.n else 0.0

    @property
    def cov(self) -> float:
        """Coefficient of variation of per-source selection counts (0 = even)."""
        counts = list(self.selections.values())
        mean = statistics.fmean(counts)
        if mean == 0:
            return 0.0
        return statistics.pstdev(counts) / mean

    def duration_pct(self, pct: float) -> float:
        if not self.durations:
            return 0.0
        ordered = sorted(self.durations)
        rank = min(len(ordered) - 1, int(math.ceil(pct / 100.0 * len(ordered)) - 1))
        return ordered[max(0, rank)]


def simulate(
    policy: str,
    m_sources: int,
    n_targets: int,
    arrival_dt: float,
    service_time: float,
) -> SimResult:
    """Run one fan-out under ``policy`` and return utilization + timing stats.

    ``service_time`` is the transfer duration at full (uncontended) bandwidth;
    ``arrival_dt`` is the gap between successive target arrivals. A small
    ``arrival_dt`` relative to ``service_time`` means high concurrency (many
    overlapping transfers), which is where source contention -- and the value of
    steering away from busy sources -- shows up.
    """
    source_ids = [f"src-{i}" for i in range(m_sources)]
    selector = get_selector(policy)
    # Per source: list of [remaining_work, target_idx, start_time]; remaining
    # work is measured in full-bandwidth seconds and drains at rate 1/k.
    active: dict[str, list[list[float]]] = {s: [] for s in source_ids}
    selections = {s: 0 for s in source_ids}
    durations: list[float] = []

    def next_completion_dt() -> float:
        best = math.inf
        for s in source_ids:
            k = len(active[s])
            if k == 0:
                continue
            min_rem = min(x[0] for x in active[s])
            best = min(best, min_rem * k)  # drains at 1/k
        return best

    def drain(dt: float) -> None:
        if dt <= 0:
            return
        for s in source_ids:
            k = len(active[s])
            if k == 0:
                continue
            share = dt / k
            for x in active[s]:
                x[0] -= share

    t = 0.0
    next_arrival = 0
    while next_arrival < n_targets or any(active[s] for s in source_ids):
        t_arrival = next_arrival * arrival_dt if next_arrival < n_targets else math.inf
        t_completion = t + next_completion_dt()
        if t_arrival <= t_completion:
            drain(t_arrival - t)
            t = t_arrival
            inflight = {s: len(active[s]) for s in source_ids}
            ordered = selector.order(
                _candidates(source_ids, inflight), _ctx(next_arrival)
            )
            chosen = ordered[0].mx_source_id
            selections[chosen] += 1
            active[chosen].append([service_time, next_arrival, t])
            next_arrival += 1
        else:
            drain(t_completion - t)
            t = t_completion
            for s in source_ids:
                finished = [x for x in active[s] if x[0] <= _EPS]
                for x in finished:
                    durations.append(t - x[2])
                if finished:
                    active[s] = [x for x in active[s] if x[0] > _EPS]

    return SimResult(policy, selections, makespan=t, durations=durations)


def run_policies(
    m_sources: int,
    n_targets: int,
    arrival_dt: float,
    service_time: float,
    trials: int,
    policies: tuple[str, ...] = ("random", "rendezvous_hash", "load_aware"),
) -> dict[str, SimResult]:
    """Run each policy ``trials`` times and keep the median-makespan run.

    ``rendezvous_hash`` / ``load_aware`` are deterministic (identity + load), so
    trials coincide; ``random`` varies, so multiple trials give a fair picture.
    """
    out: dict[str, SimResult] = {}
    for policy in policies:
        runs = [
            simulate(policy, m_sources, n_targets, arrival_dt, service_time)
            for _ in range(trials)
        ]
        runs.sort(key=lambda r: r.makespan)
        out[policy] = runs[len(runs) // 2]
    return out


def _format_table(
    label: str,
    results: dict[str, SimResult],
    m_sources: int,
    n_targets: int,
) -> str:
    ideal = math.ceil(n_targets / m_sources) / n_targets
    lines = [
        f"\n{label}  (M={m_sources} sources, N={n_targets} targets, "
        f"ideal max-share={ideal:.2f})",
        f"  {'policy':<16}{'max-share':>11}{'cov':>8}{'makespan':>11}"
        f"{'p50':>9}{'p90':>9}{'p99':>9}",
    ]
    for policy, r in results.items():
        lines.append(
            f"  {policy:<16}{r.max_source_share:>11.3f}{r.cov:>8.3f}"
            f"{r.makespan:>11.1f}{r.duration_pct(50):>9.1f}"
            f"{r.duration_pct(90):>9.1f}{r.duration_pct(99):>9.1f}"
        )
    return "\n".join(lines)


# Sweep mirrors #461's configs (M x N) plus a high-fan-out sticky-source case.
_DEFAULT_SWEEP = (
    # (label, M, N, arrival_dt, service_time)
    ("4x20 moderate fan-out", 4, 20, 1.0, 10.0),
    ("4x40 high fan-out", 4, 40, 0.5, 10.0),
    ("8x32 wide pool", 8, 32, 0.5, 10.0),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=int, help="M: number of pre-warmed sources")
    parser.add_argument(
        "--targets", type=int, help="N: number of targets in the fan-out"
    )
    parser.add_argument(
        "--arrival-dt", type=float, default=0.5, help="seconds between arrivals"
    )
    parser.add_argument(
        "--service-time", type=float, default=10.0, help="uncontended transfer seconds"
    )
    parser.add_argument(
        "--trials", type=int, default=5, help="trials per policy (median kept)"
    )
    args = parser.parse_args()

    print(
        "Load-aware source-selection simulation "
        "(processor-sharing fan-out; real selector code)"
    )
    if args.sources and args.targets:
        sweep = [
            ("custom", args.sources, args.targets, args.arrival_dt, args.service_time)
        ]
    else:
        sweep = list(_DEFAULT_SWEEP)

    for label, m, n, dt, svc in sweep:
        results = run_policies(m, n, dt, svc, args.trials)
        print(_format_table(label, results, m, n))
    print(
        "\nLower max-share and makespan are better. load_aware steers new targets "
        "away from busy sources,\nflattening utilization and shortening the tail "
        "where random/rendezvous_hash pile onto one source."
    )


if __name__ == "__main__":
    main()
