# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate one bench_collective_refit run into a table.

A cohort is only as fast as its slowest rank, so every figure here is the
maximum across ranks rather than a mean: a refit is not finished until the
last generator has its bytes.
"""

from __future__ import annotations

import json
import os
import sys

GIB = 1024**3


def main(directory: str) -> int:
    results = []
    for name in sorted(os.listdir(directory)):
        if name.startswith("rank") and name.endswith(".json"):
            with open(os.path.join(directory, name)) as handle:
                results.append(json.load(handle))
    if not results:
        print(f"no rank results in {directory} - read the .log files")
        return 1

    geometry = results[0]["geometry"]
    expected = geometry["trainers"] + geometry["generators"]
    volume = results[0]["global_bulk_bytes"]
    verdicts = [r["verified"] for r in results if r["verified"] is not None]

    print()
    print(f"geometry      {geometry['trainers']} trainers -> {geometry['generators']} generators, "
          f"{geometry['partitions']} source partition(s)")
    print(f"payload       {geometry['params']} params of {geometry['rows']}x{geometry['cols']} "
          f"{geometry['dtype']} = {volume / GIB:.2f} GiB global")
    print(f"nccl          {results[0]['nccl']}")
    print(f"ranks         {len(results)} of {expected} reported")
    print(f"verified      {'PASS' if verdicts and all(verdicts) else 'FAIL or not run'} "
          f"({len(verdicts)} generator rank(s) checked byte-exact)")

    print()
    print("bootstrap, paid once per group (max over ranks)")
    for label in ("channel_s", "initialize_s", "bootstrap_s"):
        print(f"  {label:<22} {max(r[label] for r in results) * 1e3:8.1f} ms")
    phases: dict[str, float] = {}
    for r in results:
        for label, entry in r["phases"].items():
            phases[label] = max(phases.get(label, 0.0), entry["total_s"])
    for label in ("join", "publish_bootstrap", "await_ready", "comm_init", "bootstrap_barrier"):
        if label in phases:
            print(f"    {label:<20} {phases[label] * 1e3:8.1f} ms")

    starts = [r["join_start_wall"] for r in results if "join_start_wall" in r]
    if starts:
        print(f"    {'rank stagger':<20} {(max(starts) - min(starts)) * 1e3:8.1f} ms  "
              f"(spread between the first and last rank reaching compute_plan;\n"
              f"{'':<31}await_ready cannot be shorter than this, and it is the launcher's, not MX's)")

    print()
    print("per refit round (max over ranks)")
    print(f"  {'round':>5} {'total ms':>10} {'enqueue ms':>11} {'drain ms':>10} {'GiB/s':>8}")
    n_rounds = min(len(r["rounds"]) for r in results)
    steady = []
    for index in range(n_rounds):
        total = max(r["rounds"][index]["total_s"] for r in results)
        publish = max(r["rounds"][index]["publish_s"] for r in results)
        finish = max(r["rounds"][index]["finish_s"] for r in results)
        rate = volume / total / GIB
        print(f"  {index:>5} {total * 1e3:10.1f} {publish * 1e3:11.1f} {finish * 1e3:10.1f} {rate:8.2f}")
        if index > 0:
            steady.append(total)
    if steady:
        best = min(steady)
        median = sorted(steady)[len(steady) // 2]
        print()
        print(f"  steady state (rounds 1..{n_rounds - 1}): median {median * 1e3:.1f} ms "
              f"= {volume / median / GIB:.2f} GiB/s delivered, best {best * 1e3:.1f} ms "
              f"= {volume / best / GIB:.2f} GiB/s")
    print()
    return 0 if (verdicts and all(verdicts) and len(results) == expected) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mxbench/out"))
