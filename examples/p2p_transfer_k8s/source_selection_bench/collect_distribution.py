#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize on-cluster source-utilization balance from target worker logs.

Parses the structured selection logs emitted by RdmaStrategy
(``source_selector=...`` and the per-attempt ``source_worker_id=...`` of the
source a target actually transferred from) and reports the same balance signal
the offline benchmark computes, plus the fan-out makespan passed in by
run_bench.sh.

Usage: collect_distribution.py <logdir> <policy> <M> <N> <makespan_seconds>
"""

from __future__ import annotations

import glob
import os
import re
import statistics
import sys
from collections import Counter

# attempt_index=0 is the selector's top-ranked source -- the selection signal
# this benchmark measures (which source the policy chose first).
ATTEMPT_RE = re.compile(r"source_attempt_index=0 source_worker_id=(\S+)")


def selected_source(log_text: str) -> str | None:
    m = ATTEMPT_RE.search(log_text)
    return m.group(1) if m else None


def gini(counts: list[int]) -> float:
    n = len(counts)
    if n == 0 or sum(counts) == 0:
        return 0.0
    xs = sorted(counts)
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * sum(xs)) - (n + 1) / n


def main() -> None:
    logdir, policy, m, n, makespan = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    picks = []
    for path in sorted(glob.glob(os.path.join(logdir, "target-*.log"))):
        with open(path, encoding="utf-8", errors="replace") as f:
            src = selected_source(f.read())
        if src:
            picks.append(src)

    counts = Counter(picks)
    per_source = list(counts.values()) + [0] * max(0, m - len(counts))
    total = sum(per_source)
    mean = (total / m) if m else 0
    print(f"\n=== on-cluster source distribution: policy={policy} M={m} N={n} ===")
    print(f"targets with a recorded source: {total}/{n}")
    print(f"fan-out makespan: {makespan}s")
    for src, c in counts.most_common():
        print(f"  {src}: {c}")
    if total:
        print(f"max-source share: {max(per_source) / total:.3f}  (ideal {1 / m:.3f})")
        if mean:
            print(f"CoV: {statistics.pstdev(per_source) / mean:.3f}")
        print(f"Gini: {gini(per_source):.3f}")
        print(f"empty sources: {sum(1 for c in per_source if c == 0)}/{m}")


if __name__ == "__main__":
    main()
