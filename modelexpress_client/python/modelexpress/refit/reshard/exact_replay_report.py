# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Summarise the in-process exact-segment replay gate across ranks and steps.

The receiver already fails the refit on a replay mismatch, so this tool is not the
thing that catches a defect - it is the thing that establishes what a *passing* run
actually covered. That distinction is the whole reason it exists. The gate's silent
failure is not a false pass, it is a run that emits records, mismatches zero, and
compared almost nothing; ``checked == 0`` means no evidence, and a small non-zero
``checked`` means correspondingly little. On Topology B, 12 675 of 18 867 sources
take the exact path, so a report of "0 mismatches" carries weight only alongside the
number of parameters that were in fact replayed.

So the verdict here keys on coverage, not just on agreement:

* ``PASS`` - every rank and step replayed something, and nothing disagreed.
* ``FAIL`` - at least one parameter disagreed. Both sides read the same staging
  buffer, so this is a segment offset or stride defect, not a transfer fault.
* ``NO_EVIDENCE`` - no records, or nothing was comparable anywhere.
* ``PASS_PARTIAL`` - agreement everywhere it looked, but some rank/step pair
  replayed nothing, or the plan reported segments it could not map into staging.
  Distinguished from ``PASS`` because the untested remainder is exactly the
  population the gate was built for.

Unlike :mod:`dest_digest_report`, one run is enough and no assumption is made about
weights holding still: both implementations here read one set of received bytes
inside a single refit. That is what makes the gate usable past step 1, where the
two-arm comparison was drowned by GRPO's run-to-run nondeterminism.

The gate's limit, restated because a summary invites over-reading: replay executes
the exact plan's segments as local copies out of staging. It checks the descriptor
arithmetic - offsets, strides, segment boundaries - and not the fetch that would
have carried those descriptors over the wire.

Usage::

    python -m modelexpress.refit.reshard.exact_replay_report receiver.log
"""

from __future__ import annotations

import json
import sys

RECORD_MARKER = "MX_REFIT_EXACT_REPLAY "
SCHEMA = "refit-exact-replay-v1"

# Stats that mean "the gate could not look here". Reported separately from
# mismatches: an unmapped segment is a gap in coverage, not a wrong byte, and
# folding the two together would let a shrinking denominator read as a clean run.
_COVERAGE_GAPS = (
    "uncovered_params",
    "params_with_unstaged_sources",
    "copies_the_exact_path_could_not_plan",
    "segments_outside_any_staged_shard",
)

_USAGE = "usage: python -m modelexpress.refit.reshard.exact_replay_report LOG"


def parse_records(text: str) -> tuple[list[dict], int]:
    """Pull replay records out of a receiver log, tolerating trailing text.

    Ray appends ``[repeated N x across cluster]`` to deduplicated log lines, which
    a whole-line ``json.loads`` rejects. Dropping those records silently is how a
    run under-reports its own coverage, so decode a prefix and count what is
    genuinely unparseable.
    """
    records: list[dict] = []
    skipped = 0
    decoder = json.JSONDecoder()
    for line in text.splitlines():
        at = line.find(RECORD_MARKER)
        if at < 0:
            continue
        start = line.find("{", at)
        if start < 0:
            skipped += 1
            continue
        try:
            record, _ = decoder.raw_decode(line[start:])
        except ValueError:
            skipped += 1
            continue
        if record.get("schema") != SCHEMA:
            skipped += 1
            continue
        records.append(record)
    return records, skipped


def summarise(records: list[dict]) -> dict:
    """Aggregate replay records into one verdict plus its coverage evidence."""
    if not records:
        return {
            "verdict": "NO_EVIDENCE",
            "reason": "no refit-exact-replay-v1 records in the log",
            "checked": 0,
            "mismatches": 0,
        }

    # De-duplication keys on (rank, step), so a record without a rank makes every
    # receiver look like one receiver retrying. Refuse rather than report a
    # sixteenth of the coverage as if it were all of it - an early build of the
    # receiver omitted this field, and the failure is invisible in the output.
    rankless = sum(1 for r in records if r.get("rank") is None)
    if rankless:
        return {
            "verdict": "INVALID_NO_RANK",
            "reason": (
                f"{rankless} of {len(records)} record(s) carry no rank, so per-rank "
                f"records cannot be told apart from one rank's retries and coverage "
                f"cannot be totalled. Producer is too old; re-run on a receiver that "
                f"emits rank."
            ),
            "records": len(records),
            "checked": 0,
            "mismatches": 0,
        }

    checked = 0
    mismatches = 0
    detail: list[dict] = []
    gaps = {key: 0 for key in _COVERAGE_GAPS}
    # Keyed per rank and step because a gate that covered 4 000 params on one rank
    # and nothing on another is not a pass, and a total would hide that.
    empty: list[dict] = []
    seen: set[tuple] = set()
    forced = set()

    for record in records:
        rank = record.get("rank")
        step = record.get("step")
        key = (rank, step)
        # A refit that retries emits the record more than once for one step. Keep
        # the pair once so the denominator stays a parameter count, not a count of
        # how many times the step was attempted.
        if key in seen:
            continue
        seen.add(key)
        record_checked = int(record.get("checked") or 0)
        checked += record_checked
        mismatches += int(record.get("mismatches") or 0)
        detail.extend(record.get("detail") or [])
        for gap in _COVERAGE_GAPS:
            gaps[gap] += int(record.get(gap) or 0)
        if "forced_full_pull" in record:
            forced.add(bool(record["forced_full_pull"]))
        if not record_checked:
            empty.append({"rank": rank, "step": step})

    report = {
        "records": len(records),
        "rank_step_pairs": len(seen),
        "steps": sorted({s for _, s in seen if s is not None}),
        "ranks": sorted({r for r, _ in seen if r is not None}),
        "checked": checked,
        "mismatches": mismatches,
        "coverage_gaps": gaps,
        "rank_steps_with_no_coverage": empty[:20],
    }
    # A run where this is False is one where the gate ran without the staging it
    # needs; the reading of a low `checked` changes completely, so surface it.
    if forced:
        report["forced_full_pull"] = sorted(forced)

    if mismatches:
        report["verdict"] = "FAIL"
        report["reason"] = (
            f"{mismatches} destination param(s) differ between the exact segment "
            f"plan and a local re-slice of the same received bytes. Both read the "
            f"same staging buffer, so this is a plan_pull offset/stride defect."
        )
        report["detail"] = detail[:20]
        return report
    if not checked:
        report["verdict"] = "NO_EVIDENCE"
        report["reason"] = (
            "records were emitted but nothing was comparable, so the exact path is "
            "unchecked; zero mismatches here is not a pass"
        )
        return report
    if empty or gaps["segments_outside_any_staged_shard"] or gaps[
        "copies_the_exact_path_could_not_plan"
    ]:
        report["verdict"] = "PASS_PARTIAL"
        report["reason"] = (
            f"{checked} param comparison(s) agreed and none disagreed, but coverage "
            f"is incomplete: {len(empty)} rank/step pair(s) replayed nothing, "
            f"{gaps['segments_outside_any_staged_shard']} segment(s) mapped outside "
            f"any staged shard, "
            f"{gaps['copies_the_exact_path_could_not_plan']} copy/copies could not "
            f"be planned by the exact path"
        )
        return report
    report["verdict"] = "PASS"
    report["reason"] = (
        f"{checked} param comparison(s) across {len(seen)} rank/step pair(s) agreed, "
        f"none disagreed, and every pair replayed something. Covers descriptor "
        f"arithmetic only, not wire execution."
    )
    return report


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print(_USAGE, file=sys.stderr)
        return 2
    with open(argv[0]) as handle:
        records, skipped = parse_records(handle.read())
    report = summarise(records)
    if skipped:
        report["records_skipped"] = skipped
    print(json.dumps(report, indent=2))
    # PASS_PARTIAL and NO_EVIDENCE share exit 3: neither is a defect, and both mean
    # the run covered less than the caller probably assumes. Separating them from
    # FAIL keeps a coverage gap from sending someone hunting a planner bug.
    return {"PASS": 0, "PASS_PARTIAL": 3, "NO_EVIDENCE": 3}.get(
        report["verdict"], 1
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
