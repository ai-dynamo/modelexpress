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

That warning turned out to be too weak to be useful on its own, because the
interference is not always wholesale. The first real pairing (2026-07-29, small rig,
Qwen3-30B) reported FAIL on 8 of 870 parameters - 0.92%, comfortably localised, and
so exactly the shape a mis-sliced tensor would take. It was not one. The refit had
gone into a retry loop at step 3, and ``verify_full_pulls`` had independently flagged
360 source-level mismatches at that same step in the subject arm and 118 in the
reference, while steps 1, 2, 4 and 5 were clean in both. Step 3 was also the only
step the two arms had in common, so the gate compared the one step whose *sources*
were already known to be inconsistent, and dressed the result up as a planner bug.
One of the eight, ``o_proj``, is a full-pull source at generator TP2 and therefore
took the same path in both arms - its digests could not have differed for any reason
this gate is looking for.

So the source-level verdict is now a precondition rather than a footnote: a step is
only compared when ``verify_full_pulls`` reported zero mismatches for it in *both*
arms. Steps that fail that test are named and excluded, and a pairing with no clean
shared step reports ``INVALID_NO_CLEAN_STEP`` rather than a mismatch count nobody
should act on.

Usage::

    python -m modelexpress.refit.reshard.dest_digest_report subject.log reference.log
"""

from __future__ import annotations

import json
import sys

RECORD_MARKER = "MX_REFIT_DEST_DIGEST "
SCHEMA = "refit-dest-digest-v2"

# v1 had no ``source_digests``. Still accepted, because the archived evidence from
# 2026-07-29/30 is v1 and re-reading it is how the noise floor was established;
# rejecting it would strand the only pairing we have. Freshness auditing degrades to
# "no evidence" on a v1 record rather than being silently skipped.
SCHEMA_V1 = "refit-dest-digest-v1"
ACCEPTED_SCHEMAS = (SCHEMA, SCHEMA_V1)

# The source-level gate, emitted by the same receiver into the same log. Read here so
# a destination comparison can refuse steps whose sources already disagreed with the
# publishers, which is otherwise indistinguishable from a slicing bug.
VERIFY_MARKER = "MX_REFIT_VERIFY "
VERIFY_SCHEMA = "refit-verify-v1"
# v2 adds ``rank``, without which N receiver ranks' records are indistinguishable.
VERIFY_SCHEMAS = (VERIFY_SCHEMA, "refit-verify-v2")

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
    source_digests: dict | None = None,
    source_digest_stats: dict | None = None,
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

    ``source_digests`` is what makes a single run self-checking, and it is the field
    the v1 schema most needed. See
    :func:`modelexpress.refit.reshard.verify.source_expectation_digests` for why a
    cross-run comparison cannot answer the staleness question and this can.
    """
    record = {
        "schema": SCHEMA,
        "step": step,
        "rank": rank,
        "force_full_pull": bool(forced_full_pull),
        "params": len(digests),
        "digests": digests,
        "unbounded_sources": sorted(unbounded_sources or ()),
        "fallback_sources": sorted(fallback_sources or ()),
        # Always present on a v2 record, null when the receiver could not build it,
        # so a reader can tell an old receiver from one that had nothing to report.
        "source_digests": source_digests,
    }
    if source_digest_stats:
        record["source_digest_stats"] = source_digest_stats
    return record


def parse_records(text: str) -> list[dict]:
    """Extract destination-digest records (v1 or v2) from arbitrary log text.

    Logs interleave ranks and carry prefixes, so records are found by marker and
    parsed individually; an unparseable line is skipped rather than failing the
    run, since a truncated final line is normal when a log is still being written.

    A record is not always the last thing on its line - a Ray driver log appends
    ``[repeated N x across cluster]`` - so the JSON is read with a decoder that
    stops at the end of the object instead of requiring it to end the line. Before
    this, such a record raised ``ValueError`` and was dropped by the skip path
    below, silently shrinking coverage with no signal. Skips are counted for the
    same reason: a gate that quietly compares fewer params than it claims reports a
    pass it has not earned.
    """
    return parse_records_with_skips(text)[0]


def parse_records_with_skips(text: str) -> tuple[list[dict], int]:
    """:func:`parse_records`, plus the number of marked records it could not use.

    The count is returned rather than accumulated somewhere, because the interesting
    case is "this log had 188 records and we used 170", which only means anything
    next to the text it came from.
    """
    records: list[dict] = []
    skipped = 0
    decoder = json.JSONDecoder()
    for line in text.splitlines():
        index = line.find(RECORD_MARKER)
        if index < 0:
            continue
        try:
            record, _end = decoder.raw_decode(line[index + len(RECORD_MARKER) :])
        except ValueError:
            skipped += 1
            continue
        if isinstance(record, dict) and record.get("schema") in ACCEPTED_SCHEMAS:
            records.append(record)
        else:
            skipped += 1
    return records, skipped


def parse_verify_records(text: str) -> list[dict]:
    """Extract ``refit-verify-v1``/``v2`` records from the same log text.

    Deliberately separate from :func:`parse_records` so a caller can supply the two
    kinds from different places, but in practice both come from one receiver log.

    ``raw_decode`` rather than ``json.loads`` for the same reason as
    :func:`parse_records`: Ray appends ``[repeated N x across cluster]`` after the
    record, and decoding the whole remainder of the line then raises and drops it.
    Silently, and precisely on the runs with the most ranks - so the reader would go
    blind exactly when the source gate matters most.

    v2 adds ``rank``. Both are accepted, since a fleet is upgraded one image at a
    time and a mixed-version log is normal.
    """
    records = []
    decoder = json.JSONDecoder()
    for line in text.splitlines():
        index = line.find(VERIFY_MARKER)
        if index < 0:
            continue
        try:
            record, _ = decoder.raw_decode(line[index + len(VERIFY_MARKER) :].strip())
        except ValueError:
            continue
        if record.get("schema") in VERIFY_SCHEMAS:
            records.append(record)
    return records


def _dirty_steps(verify: list[dict]) -> dict:
    """Sum source-level mismatches per step, keeping only the steps that had any.

    Summed across ranks and across repeated attempts: a retry loop emits many records
    for one step, and a mismatch in any of them means the sources for that step were
    not consistent while it was being read.
    """
    totals: dict = {}
    for record in verify:
        count = int(record.get("mismatches") or 0)
        if count:
            step = record.get("step")
            totals[step] = totals.get(step, 0) + count
    return totals


def audit_freshness(records: list[dict]) -> dict:
    """Audit one run for staleness, using the source digests in a v2 record.

    Needs no second run, which is the point: the cross-run comparison in
    :func:`compare` is only valid where both arms hold identical source weights,
    and after one training step they do not. This looks along the *steps of a
    single run* instead and asks whether each destination digest moved consistently
    with the publisher claims behind it.

    Three findings, and they do not carry the same weight. The asymmetry comes from
    the source fingerprint covering every shard of every contributing source, while
    the destination holds only this rank's slice - so the fingerprint is a superset
    of what the rank actually reads.

    * ``dest_moved_alone`` - **strong.** The destination changed while every
      contributing shard digest stayed put. Superset or not, if nothing in the
      superset changed then nothing this rank read changed, so the assembled bytes
      had no business changing. Wrong bytes, or bytes from somewhere else.
    * ``dest_reverted`` - **strong**, when the source never reverted. A destination
      digest returning to a value from an earlier step while the publishers' claims
      moved only forward means this step installed an earlier step's bytes. This is
      the staleness class, and it is the finding this function was added for: an
      unexplained instance of exactly this shape (two tensors' step-5 digests equal
      to their own step-3 digests) is what the v1 schema could not attribute.
    * ``source_moved_dest_static`` - **advisory only.** Reported because a genuinely
      missed update looks like this, but a shard changing in a region this rank
      never reads looks identical, and the superset fingerprint cannot tell them
      apart. Do not fail a run on it.

    A param whose source digest is ``None`` at any step is excluded and counted in
    ``no_evidence``, so thin coverage cannot read as a clean audit. A v1 record has
    no source digests at all, which lands the whole run in ``NO_EVIDENCE``.
    """
    by_rank: dict = {}
    for record in records:
        by_rank.setdefault(record.get("rank"), {})[record.get("step")] = record

    dest_moved_alone: list = []
    dest_reverted: list = []
    source_moved_dest_static: list = []
    compared = 0
    no_evidence = 0
    params_audited = 0

    for rank in sorted(by_rank, key=lambda r: (r is None, r)):
        steps = sorted(by_rank[rank], key=lambda s: (s is None, s))
        if len(steps) < 2:
            continue
        first = by_rank[rank][steps[0]]
        for param in sorted((first.get("digests") or {})):
            dest_seq = []
            src_seq = []
            usable = True
            for step in steps:
                record = by_rank[rank][step]
                dest = (record.get("digests") or {}).get(param)
                src = (record.get("source_digests") or {}).get(param)
                if dest is None or src is None:
                    usable = False
                    break
                dest_seq.append((step, dest))
                src_seq.append((step, src))
            if not usable:
                no_evidence += 1
                continue
            params_audited += 1

            # Reversion is judged over the whole sequence, not step to step: the
            # signature is a value *reappearing* after something else intervened.
            src_reverted = _has_reversion([d for _s, d in src_seq])
            if _has_reversion([d for _s, d in dest_seq]) and not src_reverted:
                dest_reverted.append(
                    {
                        "rank": rank,
                        "param": param,
                        "dest": [[s, d[:12]] for s, d in dest_seq],
                        "source": [[s, d[:12]] for s, d in src_seq],
                    }
                )

            for index in range(1, len(steps)):
                compared += 1
                dest_changed = dest_seq[index][1] != dest_seq[index - 1][1]
                src_changed = src_seq[index][1] != src_seq[index - 1][1]
                if dest_changed and not src_changed:
                    dest_moved_alone.append(
                        {
                            "rank": rank,
                            "param": param,
                            "step": steps[index],
                            "prev_step": steps[index - 1],
                        }
                    )
                elif src_changed and not dest_changed:
                    source_moved_dest_static.append(
                        {
                            "rank": rank,
                            "param": param,
                            "step": steps[index],
                            "prev_step": steps[index - 1],
                        }
                    )

    if not params_audited:
        verdict = "NO_EVIDENCE"
    elif dest_moved_alone or dest_reverted:
        verdict = "FAIL"
    else:
        verdict = "PASS"

    return {
        "verdict": verdict,
        "params_audited": params_audited,
        "no_evidence": no_evidence,
        "step_transitions_compared": compared,
        "dest_moved_alone": dest_moved_alone[:20],
        "dest_moved_alone_count": len(dest_moved_alone),
        "dest_reverted": dest_reverted[:20],
        "dest_reverted_count": len(dest_reverted),
        "source_moved_dest_static_count": len(source_moved_dest_static),
        "source_moved_dest_static": source_moved_dest_static[:20],
        "advisory": (
            "source_moved_dest_static is advisory: the source fingerprint covers "
            "shards this rank may not read, so it can change without the "
            "destination changing. Do not fail a run on it."
        ),
    }


def _trajectory_safety(key, subject_record: dict, reference_record: dict) -> dict | None:
    """Decide whether one (rank, step) pair may be compared across two runs.

    Returns ``None`` when the pair is safe, or a row describing why it is not.

    This exists because the gate's central assumption turned out to be false on
    real hardware. Comparing two runs only means anything if both arms held the
    same source weights, and GRPO on the 2026-07-30 rig is not bitwise
    reproducible: two runs with *identical* settings disagreed on 10 of 4 350
    parameters, an order of magnitude more than the 1 mismatch the cross-arm
    comparison had reported. The gate was measuring trajectory drift and calling
    it a planner defect.

    Where the record carries source digests, that question is answerable rather
    than assumed: if the digests of the shards feeding this rank agree across the
    arms, the two runs really were looking at the same weights. A v1 record cannot
    answer it, and there the only defensible pair is the first refit, before any
    optimizer step has been applied and while both arms still hold the checkpoint
    as loaded. That is the pairing the 870/870 result rests on.
    """
    step = key[1]
    subject_sources = subject_record.get("source_digests") or {}
    reference_sources = reference_record.get("source_digests") or {}

    if subject_sources and reference_sources:
        shared = set(subject_sources) & set(reference_sources)
        if not shared:
            return {
                "rank": key[0],
                "step": step,
                "reason": "arms share no source parameter, so the snapshots cannot be shown to match",
            }
        differing = sorted(
            name for name in shared if subject_sources[name] != reference_sources[name]
        )
        if differing:
            return {
                "rank": key[0],
                "step": step,
                "reason": (
                    f"{len(differing)} of {len(shared)} source expectation(s) differ "
                    f"between the arms, so the publishers had moved and a destination "
                    f"difference cannot be attributed to slicing"
                ),
                "examples": differing[:5],
            }
        return None

    # No source digests: fall back to the one step whose sources are equal by
    # construction rather than by measurement.
    if step == 1:
        return None
    return {
        "rank": key[0],
        "step": step,
        "reason": (
            "record carries no source digests (schema v1), so a shared snapshot "
            "cannot be demonstrated, and past the first refit training has moved "
            "the weights by an amount that exceeds what this gate can resolve"
        ),
    }


def _has_reversion(sequence: list) -> bool:
    """True when a value reappears after a different value intervened.

    Not the same as "has duplicates": a value repeating on consecutive steps is a
    weight that simply did not move, which is the common case here - at lr=3e-7 in
    bf16 only 7 of 870 params moved at all over five steps.
    """
    last_position: dict = {}
    for position, value in enumerate(sequence):
        previous = last_position.get(value)
        if previous is not None and previous != position - 1:
            return True
        last_position[value] = position
    return False


def _index(records: list[dict]) -> dict:
    """Index records by ``(rank, step)``.

    Ranks hold different shards of different parameters, and steps hold different
    weights, so only a matching pair is comparable. A later record for the same key
    wins, which is what re-reading a growing log should do.
    """
    return {(record.get("rank"), record.get("step")): record for record in records}


def compare(
    subject: list[dict],
    reference: list[dict],
    *,
    subject_verify: list[dict] | None = None,
    reference_verify: list[dict] | None = None,
) -> dict:
    """Compare per-parameter destination digests between the two arms.

    Returns a report with a ``verdict`` of ``PASS``, ``FAIL``,
    ``INVALID_NOT_TWO_ARMS``, ``INVALID_NO_CLEAN_STEP`` or ``NO_COMPARABLE_RECORDS``.
    The invalid verdicts exist because this gate's predecessor was undone by exactly
    that failure: a check that silently cannot run reports zero mismatches and reads
    as a clean run.

    ``subject_verify`` and ``reference_verify`` are ``refit-verify-v1`` records from
    the same logs. When supplied, any step whose sources did not verify clean in both
    arms is excluded before comparing - see the module docstring for the run that
    made this mandatory. They are optional only so the digest logic stays testable in
    isolation; omitting them on real logs leaves the gate able to blame the planner
    for a publisher that was moving.
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

    # Drop steps whose sources were already inconsistent. Done after the shared-key
    # intersection so the report can say whether anything survived, and which steps
    # went, rather than silently comparing a smaller set.
    subject_dirty = _dirty_steps(subject_verify or [])
    reference_dirty = _dirty_steps(reference_verify or [])
    dirty = {**subject_dirty}
    for step, count in reference_dirty.items():
        dirty[step] = dirty.get(step, 0) + count
    excluded = sorted(
        (
            {
                "step": step,
                "subject_source_mismatches": subject_dirty.get(step, 0),
                "reference_source_mismatches": reference_dirty.get(step, 0),
            }
            for step in {k[1] for k in shared_keys} & set(dirty)
        ),
        key=lambda row: row["step"] or 0,
    )
    excluded_steps = {row["step"] for row in excluded}
    clean_keys = [key for key in shared_keys if key[1] not in excluded_steps]
    source_verify_checked = bool(subject_verify or reference_verify)

    if source_verify_checked and not clean_keys:
        return {
            "verdict": "INVALID_NO_CLEAN_STEP",
            "reason": (
                "every step the two arms share had source-level verify mismatches, so "
                "the destination digests cannot distinguish a slicing bug from "
                "publishers that were moving. Excluded: "
                + ", ".join(
                    f"step {row['step']} (subject {row['subject_source_mismatches']}, "
                    f"reference {row['reference_source_mismatches']})"
                    for row in excluded
                )
                + ". Compare a step that verified clean in both arms."
            ),
            "compared_records": 0,
            "compared_params": 0,
            "mismatches": 0,
            "source_verify_checked": True,
            "excluded_dirty_steps": excluded,
        }

    # Second precondition, independent of the source-verify one above: a step can
    # have perfectly clean sources in both arms and still be uncomparable, because
    # the two arms' sources moved to *different* clean values. That is the case the
    # noise-floor control exposed.
    unsafe = []
    safe_keys = []
    for key in clean_keys:
        row = _trajectory_safety(key, subject_index[key], reference_index[key])
        if row is None:
            safe_keys.append(key)
        else:
            unsafe.append(row)

    if not safe_keys:
        return {
            "verdict": "INVALID_NO_TRAJECTORY_SAFE_STEP",
            "reason": (
                "no (rank, step) pair could be shown to have the same source weights "
                "in both arms, so any destination difference is indistinguishable from "
                "training having moved between the runs. This is the failure the "
                "in-process replay gate exists to avoid; prefer "
                "exact_replay_report on a single run. Excluded: "
                + "; ".join(f"rank{r['rank']}/step{r['step']}: {r['reason']}" for r in unsafe[:6])
            ),
            "compared_records": 0,
            "compared_params": 0,
            "mismatches": 0,
            "source_verify_checked": source_verify_checked,
            "excluded_dirty_steps": excluded,
            "excluded_unsafe_pairs": unsafe[:20],
        }

    shared_keys = safe_keys
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
        # False means no verify records were supplied, so nothing ruled out a moving
        # publisher. A PASS is still meaningful; a FAIL is not yet attributable.
        "source_verify_checked": source_verify_checked,
        "excluded_dirty_steps": excluded,
        # Pairs dropped because the two arms' publishers had moved apart. Reported
        # on a pass as well: it is the difference between "the exact path agreed
        # everywhere" and "it agreed on the one step where agreement was possible",
        # and only the second of those is what a two-arm run on a nondeterministic
        # trainer can support.
        "excluded_unsafe_pairs": unsafe[:20],
    }
    if not mismatches and unsafe:
        report["verdict"] = "PASS_PARTIAL"
        report["reason"] = (
            f"no digest disagreed over {compared_params} comparison(s), but "
            f"{len(unsafe)} rank/step pair(s) were excluded because the two arms' "
            f"source weights could not be shown to match. The exact path is verified "
            f"only for the pairs that survived: {sorted(safe_keys)[:6]}."
        )
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
    if mismatches and not source_verify_checked:
        report["attribution_warning"] = (
            "no refit-verify-v1 records were supplied, so this FAIL is not "
            "attributable: a step whose sources disagreed with the publishers "
            "produces localised destination mismatches that look exactly like a "
            "slicing bug. Pass the receiver logs so dirty steps can be excluded."
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


_USAGE = (
    "usage: python -m modelexpress.refit.reshard.dest_digest_report "
    "SUBJECT_LOG [REFERENCE_LOG]\n"
    "  SUBJECT_LOG    run planned normally (MX_RESHARD_DEST_DIGEST=1)\n"
    "  REFERENCE_LOG  run with MX_RESHARD_FORCE_FULL_PULL=1 as well\n"
    "\n"
    "With two logs, differences the exact-fetch path against the full-pull path.\n"
    "That comparison is only valid where both runs hold identical source weights,\n"
    "which in practice means the first step: training is not bitwise reproducible,\n"
    "and past step 1 the run-to-run noise floor exceeded the effect being measured.\n"
    "\n"
    "With one log, audits that run against its own source digests instead, which\n"
    "needs no second run and covers every step. Requires a v2 record."
)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) not in (1, 2):
        print(_USAGE, file=sys.stderr)
        return 2
    if len(argv) == 1:
        with open(argv[0]) as handle:
            records, skipped = parse_records_with_skips(handle.read())
        report = audit_freshness(records)
        report["records"] = len(records)
        if skipped:
            report["records_skipped"] = skipped
        print(json.dumps(report, indent=2))
        # NO_EVIDENCE is exit 3, alongside PASS_PARTIAL: both mean the check did not
        # cover what the caller assumes, and neither is a defect.
        return {"PASS": 0, "NO_EVIDENCE": 3}.get(report["verdict"], 1)
    # Both record kinds come from the same receiver log, so read each file once and
    # parse both rather than asking the caller to pass four paths.
    with open(argv[0]) as handle:
        subject_text = handle.read()
    with open(argv[1]) as handle:
        reference_text = handle.read()
    subject, subject_skips = parse_records_with_skips(subject_text)
    reference, reference_skips = parse_records_with_skips(reference_text)
    report = compare(
        subject,
        reference,
        subject_verify=parse_verify_records(subject_text),
        reference_verify=parse_verify_records(reference_text),
    )
    if subject_skips or reference_skips:
        report["records_skipped"] = {
            "subject": subject_skips,
            "reference": reference_skips,
        }
    print(json.dumps(report, indent=2))
    # Partial coverage gets its own code rather than being folded into either
    # outcome: reporting it as success would overclaim, and as failure would send
    # someone looking for a defect when the finding is a gap in what was checked.
    # "Could not be measured" is likewise not "failed", and conflating the two is how
    # the first real pairing came within one reading of being filed as a planner bug.
    return {
        "PASS": 0,
        "PASS_PARTIAL": 3,
        "INVALID_NO_CLEAN_STEP": 4,
        # Also 4. A pairing that cannot be shown to share source weights was not
        # measured, and exiting 1 would put it in the same bucket as a planner
        # defect - the exact conflation this verdict was added to prevent.
        "INVALID_NO_TRAJECTORY_SAFE_STEP": 4,
    }.get(report["verdict"], 1)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
