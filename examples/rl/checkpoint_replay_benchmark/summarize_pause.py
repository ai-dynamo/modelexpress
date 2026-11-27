# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate complete paired streaming trials; never include pilot/warmup rows."""

import argparse
import json
from pathlib import Path
from statistics import median

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
p.add_argument("--runs", type=int, default=3)
a = p.parse_args()
all_pairs = []
starts = []
controls = []
reference_hashes = None
for run in range(1, a.runs + 1):
    rows = [
        json.loads(x)
        for x in (a.directory / f"pause-run{run}.jsonl").read_text().splitlines()
    ]
    complete = next(x for x in rows if x["kind"] == "complete")
    q = next(x for x in rows if x["kind"] == "qualification")
    if q["result"] != "PASS":
        raise ValueError("qualification failed")
    if reference_hashes is None:
        reference_hashes = q["reference_hashes"]
    if reference_hashes != q["reference_hashes"]:
        raise ValueError("runtime target weights differ between process starts")
    pairs = {}
    for row in rows:
        if row["kind"] != "measurement":
            continue
        if row["arm"] == "control":
            controls.append(row)
            continue
        if row["correctness"] != "PASS":
            raise ValueError("post-update correctness failed")
        pair = pairs.setdefault(row["pair"], {})
        if row["arm"] in pair:
            raise ValueError("duplicate arm")
        if len(row["stream_gaps_s"]) != 4:
            raise ValueError("missing stream gaps")
        if any(r["tokens"] != 384 for r in row["requests"]):
            raise ValueError("incomplete request")
        row["median_request_latency_s"] = median(
            r["end"] - r["start"] for r in row["requests"]
        )
        pair[row["arm"]] = row
    if set(pairs) != set(range(complete["pairs"])) or any(
        set(x) != {"ordinary", "mapped"} for x in pairs.values()
    ):
        raise ValueError("incomplete pair set")
    values = list(pairs.values())
    all_pairs.extend(values)
    starts.append(
        {
            "run": run,
            "pairs": len(values),
            "capture_s": q["capture_s"],
            "ordinary_pause_median_s": median(
                x["ordinary"]["pause_rpc_s"] for x in values
            ),
            "mapped_pause_median_s": median(x["mapped"]["pause_rpc_s"] for x in values),
        }
    )
result = {
    "status": "COMPLETE_WITH_PRIOR_CROSS_PROCESS_ORACLE_CAVEAT",
    "process_starts": a.runs,
    "pairs": len(all_pairs),
    "starts": starts,
    "scope": "live AsyncLLM pause/resume and in-process streaming gaps; synthetic checkpoint; no HTTP, S3, trainer or full delta-update timing",
    "topology": "Qwen3-30B-A3B BF16; vLLM TP4; one node with four GPUs; record actual hardware separately",
    "pause_mode": "keep",
    "clear_cache": False,
    "concurrent_streams": 4,
    "tokens_per_request": 384,
    "correctness": "Every accepted arm matched same-process ordinary-refit registered tensor bytes/storage and fresh-request outputs after draining and cache reset. In-flight responses span weight versions and are not cold-model correctness checks. Prior cross-process cold-start output variation remains unresolved.",
    "controls_pause_median_s": median(x["pause_rpc_s"] for x in controls),
    "controls_stream_gap_median_s": median(x["max_stream_gap_s"] for x in controls),
    "events_during_hold": sum(
        x[arm]["events_during_hold"]
        for x in all_pairs
        for arm in ("ordinary", "mapped")
    ),
}
for metric in (
    "pause_rpc_s",
    "scheduler_hold_s",
    "install_rpc_s",
    "max_stream_gap_s",
    "median_request_latency_s",
):
    av = [x["ordinary"][metric] for x in all_pairs]
    bv = [x["mapped"][metric] for x in all_pairs]
    result[metric] = {
        "ordinary_median_s": median(av),
        "mapped_median_s": median(bv),
        "ordinary_range_s": [min(av), max(av)],
        "mapped_range_s": [min(bv), max(bv)],
        "paired_saving_median_s": median(x - y for x, y in zip(av, bv)),
        "paired_reduction_median_pct": median(
            100 * (x - y) / x for x, y in zip(av, bv)
        ),
    }
(a.directory / "pause-summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
