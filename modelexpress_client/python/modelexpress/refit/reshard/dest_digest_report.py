# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Difference two refit runs' destination digests to gate the exact-fetch path.

``verify_full_pulls`` checks source shards that land whole in a staging buffer. On
Topology B that was 6 192 of 18 867 sources; the other 12 675 arrive as exact
segments written straight into the receive buffers, where no gate could see them. A
wrong offset or stride in ``plan_pull`` would deliver plausible bytes to the wrong
place and every existing check would pass.

This module supplies the missing half. Two runs are needed:

* **subject** - planned normally, so the exact-segment path is exercised.
* **reference** - planned with ``MX_RESHARD_FORCE_FULL_PULL=1``, so every source
  arrives through the full-pull path, whose staging contents are independently
  confirmed against the publishers' digests by ``verify_full_pulls``.

Both emit a ``refit-dest-digest-v1`` record per rank per step, digesting the
assembled receive buffers before install. Differencing them per parameter checks
the untested path against a confirmed one and names any tensor that disagrees.

Two paths agreeing is not proof in the abstract - they could be wrong identically -
but they are separate code (segment planning versus contiguous staging plus a local
narrow), so agreement is meaningful evidence and disagreement is conclusive.

**The comparison is only valid if the weights held still between the two runs.**
Training moves weights legitimately, and a moved weight is indistinguishable here
from a mis-sliced one - the trap Bug 9 fell into. Run against a quiesced publisher.
When most parameters differ, suspect the setup rather than the planner; the report
says so explicitly rather than leaving it to the reader.

Usage::

    python -m modelexpress.refit.reshard.dest_digest_report subject.log reference.log
"""

from __future__ import annotations

import json
import sys

RECORD_MARKER = "MX_REFIT_DEST_DIGEST "
SCHEMA = "refit-dest-digest-v1"

# Above this fraction of parameters differing, the finding is reported as a probable
# setup fault rather than a planner bug. A mis-sliced tensor is a localised defect;
# weights that moved between runs, or a subject/reference pairing that is really two
# copies of the same arm, move essentially everything. Calling the latter a
# correctness failure would send someone hunting a planner bug that is not there.
_WHOLESALE_FRACTION = 0.5


def dest_digest_record(
    *,
    step: int,
    rank: int,
    forced_full_pull: bool,
    digests: dict,
    unbounded_sources: list | None = None,
    fallback_sources: list | None = None,
) -> dict:
    """Build the record the receiver emits and :func:`compare` consumes.

    Shared by both sides on purpose. The producer previously built this inline,
    which let the emitted key names drift from the ones the comparator reads - and a
    comparator that finds no ``digests`` key sees empty records, compares nothing,
    and would have reported that as a pass. One constructor makes that drift a
    test failure instead of a silent non-check.

    ``unbounded_sources`` is what keeps the reference arm honest, and it is not
    optional bookkeeping. ``plan_transfer`` cannot bound a source whose shards do not
    tile dim 0 contiguously, and when bounding fails it *preserves the exact plan*
    rather than failing. So ``MX_RESHARD_FORCE_FULL_PULL=1`` does not guarantee a
    full pull for every source: the ones listed here were fetched by the exact path
    in both arms, which makes their agreement vacuous - the gate compared the path
    under test against itself. Recording the list lets :func:`compare` report the
    result as partial instead of claiming coverage it does not have.

    ``fallback_sources`` are not refit at all, so they hold whatever the engine
    already had. Their bytes should match trivially, but if the two arms disagree on
    *which* sources fell back, mismatches follow for reasons unrelated to slicing.
    """
    return {
        "schema": SCHEMA,
        "step": step,
        "rank": rank,
        "force_full_pull": bool(forced_full_pull),
        "params": len(digests),
        "digests": digests,
        "unbounded_sources": sorted(unbounded_sources or ()),
        "fallback_sources": sorted(fallback_sources or ()),
    }


def parse_records(text: str) -> list[dict]:
    """Extract ``refit-dest-digest-v1`` records from arbitrary log text.

    Logs interleave ranks and carry prefixes, so records are found by marker and
    parsed individually; an unparseable line is skipped rather than failing the
    run, since a truncated final line is normal when a log is still being written.
    """
    records = []
    for line in text.splitlines():
        index = line.find(RECORD_MARKER)
        if index < 0:
            continue
        try:
            record = json.loads(line[index + len(RECORD_MARKER) :])
        except ValueError:
            continue
        if record.get("schema") == SCHEMA:
            records.append(record)
    return records


def _index(records: list[dict]) -> dict:
    """Index records by ``(rank, step)``.

    Ranks hold different shards of different parameters, and steps hold different
    weights, so only a matching pair is comparable. A later record for the same key
    wins, which is what re-reading a growing log should do.
    """
    return {(record.get("rank"), record.get("step")): record for record in records}


def compare(subject: list[dict], reference: list[dict]) -> dict:
    """Compare per-parameter destination digests between the two arms.

    Returns a report with a ``verdict`` of ``PASS``, ``FAIL``,
    ``INVALID_NOT_TWO_ARMS`` or ``NO_COMPARABLE_RECORDS``. The two invalid verdicts
    exist because this gate's predecessor was undone by exactly that failure: a
    check that silently cannot run reports zero mismatches and reads as a clean run.
    """
    subject_arms = {bool(r.get("force_full_pull")) for r in subject}
    reference_arms = {bool(r.get("force_full_pull")) for r in reference}

    # The gate rests entirely on the two inputs being different arms. Passing the
    # same log twice, or forgetting MX_RESHARD_FORCE_FULL_PULL on the reference,
    # compares a path against itself: every digest matches and the result looks
    # like a pass while testing nothing.
    if subject_arms != {False} or reference_arms != {True}:
        return {
            "verdict": "INVALID_NOT_TWO_ARMS",
            "reason": (
                "subject must be planned normally and reference with "
                "MX_RESHARD_FORCE_FULL_PULL=1; got subject force_full_pull="
                f"{sorted(subject_arms)} reference force_full_pull="
                f"{sorted(reference_arms)}"
            ),
            "compared_records": 0,
            "compared_params": 0,
            "mismatches": 0,
        }

    subject_index, reference_index = _index(subject), _index(reference)
    shared_keys = sorted(
        set(subject_index) & set(reference_index), key=lambda k: (k[0] or 0, k[1] or 0)
    )
    if not shared_keys:
        return {
            "verdict": "NO_COMPARABLE_RECORDS",
            "reason": (
                "no (rank, step) pair appears in both runs, so nothing was "
                f"compared; subject has {sorted(subject_index)[:8]} and reference "
                f"has {sorted(reference_index)[:8]}"
            ),
            "compared_records": 0,
            "compared_params": 0,
            "mismatches": 0,
        }

    compared_params = 0
    mismatches: list[dict] = []
    # A parameter digested by one arm and not the other is a coverage difference,
    # not a byte difference. Forcing full pulls should not change which parameters
    # are installed, so this being non-zero is itself a finding - and a different
    # one from a digest mismatch, with a different fix.
    only_subject: list[str] = []
    only_reference: list[str] = []

    for key in shared_keys:
        subject_digests = subject_index[key].get("digests") or {}
        reference_digests = reference_index[key].get("digests") or {}
        for name in sorted(set(subject_digests) | set(reference_digests)):
            in_subject = name in subject_digests
            in_reference = name in reference_digests
            if not in_reference:
                only_subject.append(f"rank{key[0]}/step{key[1]}:{name}")
                continue
            if not in_subject:
                only_reference.append(f"rank{key[0]}/step{key[1]}:{name}")
                continue
            compared_params += 1
            if subject_digests[name] != reference_digests[name]:
                mismatches.append(
                    {
                        "rank": key[0],
                        "step": key[1],
                        "param": name,
                        "subject": subject_digests[name],
                        "reference": reference_digests[name],
                    }
                )

    # Sources the reference arm could not actually bound into a full pull, so it
    # fetched them by the exact path too. Their agreement is guaranteed and vacuous.
    unbounded = sorted(
        {
            name
            for key in shared_keys
            for name in (reference_index[key].get("unbounded_sources") or ())
        }
    )
    # Arms that disagree on which sources were skipped entirely will disagree on
    # bytes for reasons that have nothing to do with segment slicing.
    fallback_disagreement: set = set()
    for key in shared_keys:
        subject_fallback = set(subject_index[key].get("fallback_sources") or ())
        reference_fallback = set(reference_index[key].get("fallback_sources") or ())
        fallback_disagreement |= subject_fallback ^ reference_fallback
    fallback_differs = sorted(fallback_disagreement)

    fraction = len(mismatches) / compared_params if compared_params else 0.0
    report = {
        "verdict": "PASS" if not mismatches else "FAIL",
        "compared_records": len(shared_keys),
        "compared_params": compared_params,
        "mismatches": len(mismatches),
        "mismatch_fraction": round(fraction, 6),
        "detail": mismatches[:20],
        "detail_truncated": len(mismatches) > 20,
        "only_in_subject": len(only_subject),
        "only_in_reference": len(only_reference),
        "coverage_detail": (only_subject[:10] + only_reference[:10]) or [],
        # Non-zero means the reference was not independent for these sources, so a
        # pass does not cover them. Reported even when the verdict is FAIL, because
        # it bounds what a *subsequent* fix can claim to have verified.
        "reference_not_independent": len(unbounded),
        "reference_not_independent_detail": unbounded[:20],
        "fallback_disagreement": len(fallback_differs),
        "fallback_disagreement_detail": fallback_differs[:20],
    }
    if not mismatches and unbounded:
        report["verdict"] = "PASS_PARTIAL"
        report["reason"] = (
            f"no digest disagreed, but {len(unbounded)} source(s) could not be "
            "bounded into a full pull, so the reference arm fetched them by the "
            "same exact path as the subject. Their agreement is vacuous and this "
            "run does not verify them. First: " + ", ".join(unbounded[:5])
        )
    if fallback_differs:
        report["fallback_warning"] = (
            f"{len(fallback_differs)} source(s) fell back in one arm but not the "
            "other; those parameters are not refit at all in one run, so any "
            "mismatch among them is a planning difference, not a slicing bug"
        )
    if mismatches and fraction > _WHOLESALE_FRACTION:
        report["verdict"] = "FAIL_LIKELY_SETUP"
        report["reason"] = (
            f"{len(mismatches)} of {compared_params} parameters differ "
            f"({fraction:.1%}). A mis-sliced tensor is localised; this much "
            "disagreement usually means the weights moved between the two runs "
            "(publisher not quiesced) or the two logs are not the same step. "
            "Re-run with training frozen before treating this as a planner bug."
        )
    if not compared_params:
        # Every shared record was empty, so the pairing matched but nothing was
        # actually checked. Same trap as above, one level down.
        report["verdict"] = "NO_COMPARABLE_RECORDS"
        report["reason"] = (
            f"{len(shared_keys)} record pair(s) matched but carried no overlapping "
            "parameters, so nothing was compared"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print(
            "usage: python -m modelexpress.refit.reshard.dest_digest_report "
            "SUBJECT_LOG REFERENCE_LOG\n"
            "  SUBJECT_LOG    run planned normally (MX_RESHARD_DEST_DIGEST=1)\n"
            "  REFERENCE_LOG  run with MX_RESHARD_FORCE_FULL_PULL=1 as well",
            file=sys.stderr,
        )
        return 2
    with open(argv[0]) as handle:
        subject = parse_records(handle.read())
    with open(argv[1]) as handle:
        reference = parse_records(handle.read())
    report = compare(subject, reference)
    print(json.dumps(report, indent=2))
    # Partial coverage gets its own code rather than being folded into either
    # outcome: reporting it as success would overclaim, and as failure would send
    # someone looking for a defect when the finding is a gap in what was checked.
    return {"PASS": 0, "PASS_PARTIAL": 3}.get(report["verdict"], 1)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
