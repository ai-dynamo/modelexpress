# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the destination-digest gate that covers the exact-fetch path.

The gate's predecessor left two thirds of sources unchecked, so the tests that earn
their place here are the ones proving this one *refuses to pass* when it is not
actually comparing anything: the same arm twice, a missing force-full-pull
reference, no overlapping ranks or steps, empty records. A gate that cannot run must
fail rather than report zero mismatches, because zero mismatches from a check that
never ran is indistinguishable in a report from a clean run - which is how an
earlier series of runs came to be labelled verified while proving nothing.
"""

from __future__ import annotations

import json

import pytest

from modelexpress.refit.reshard.dest_digest_report import (
    RECORD_MARKER,
    SCHEMA,
    SCHEMA_V1,
    audit_freshness,
    compare,
    dest_digest_record,
    main,
    parse_records,
    parse_records_with_skips,
    parse_verify_records,
)


def _record(rank, step, digests, *, forced=False, unbounded=None, fallback=None):
    """Built with the production constructor, so a producer/consumer key drift
    fails these tests rather than silently emptying the comparison."""
    return dest_digest_record(
        step=step,
        rank=rank,
        forced_full_pull=forced,
        digests=digests,
        unbounded_sources=unbounded,
        fallback_sources=fallback,
    )


def _log(records):
    """Records as they appear in a real log: prefixed, interleaved with noise."""
    lines = ["INFO some unrelated line", "WARNING MX_REFIT_VERIFY {\"schema\": \"x\"}"]
    for record in records:
        lines.append(
            "WARNING 2026-07-29 rank3 MX_REFIT_DEST_DIGEST " + json.dumps(record)
        )
    lines.append("INFO trailing line")
    return "\n".join(lines)


# ------------------------------------------------------------------- parsing
def test_parses_records_out_of_noisy_log():
    text = _log([_record(0, 1, {"a": "d1"}), _record(1, 1, {"a": "d2"})])
    records = parse_records(text)
    assert [r["rank"] for r in records] == [0, 1]


def test_ignores_truncated_trailing_record():
    """Normal when reading a log that is still being written."""
    text = _log([_record(0, 1, {"a": "d1"})])
    text += "\nWARNING MX_REFIT_DEST_DIGEST {\"schema\": \"refit-dest"
    assert len(parse_records(text)) == 1


def test_ignores_other_schemas():
    text = "MX_REFIT_DEST_DIGEST " + json.dumps({"schema": "something-else"})
    assert parse_records(text) == []


# --------------------------------------------------------- the gate passing
def test_pass_when_both_arms_agree():
    digests = {"layer.0.weight": "aa", "layer.1.weight": "bb"}
    report = compare([_record(0, 1, digests)], [_record(0, 1, digests, forced=True)])
    assert report["verdict"] == "PASS"
    assert report["compared_params"] == 2
    assert report["mismatches"] == 0


def test_compares_each_rank_and_step_separately():
    subject = [_record(0, 1, {"w": "a"}), _record(1, 1, {"w": "b"})]
    reference = [
        _record(0, 1, {"w": "a"}, forced=True),
        _record(1, 1, {"w": "b"}, forced=True),
    ]
    report = compare(subject, reference)
    assert report["verdict"] == "PASS"
    # Two records, not one merged set: rank 0 and rank 1 hold different bytes and
    # pooling them would have made these two digests look like a mismatch.
    assert report["compared_records"] == 2
    assert report["compared_params"] == 2


# --------------------------------------------------------- the gate failing
def test_localises_a_single_mismatched_parameter():
    """The whole point: name the tensor the exact-fetch path got wrong."""
    subject = _record(0, 1, {"good": "aa", "bad": "WRONG", "also_good": "cc"})
    reference = _record(
        0, 1, {"good": "aa", "bad": "right", "also_good": "cc"}, forced=True
    )
    report = compare([subject], [reference])
    assert report["verdict"] == "FAIL"
    assert report["mismatches"] == 1
    assert report["detail"][0]["param"] == "bad"
    assert report["detail"][0]["subject"] == "WRONG"
    assert report["detail"][0]["reference"] == "right"


def test_wholesale_disagreement_is_flagged_as_setup_not_planner():
    """Weights that moved between runs look exactly like corruption.

    Reporting that as a planner bug would send someone hunting a defect that is not
    there, so a majority of parameters differing is called out separately.
    """
    subject = _record(0, 1, {f"p{i}": f"s{i}" for i in range(10)})
    reference = _record(0, 1, {f"p{i}": f"r{i}" for i in range(10)}, forced=True)
    report = compare([subject], [reference])
    assert report["verdict"] == "FAIL_LIKELY_SETUP"
    assert "quiesced" in report["reason"]
    assert report["mismatches"] == 10


def test_a_minority_of_mismatches_stays_a_planner_failure():
    """The boundary case of the above: localised damage must not be excused."""
    digests = {f"p{i}": f"same{i}" for i in range(10)}
    subject = _record(0, 1, dict(digests, p3="DIFFERENT"))
    reference = _record(0, 1, digests, forced=True)
    report = compare([subject], [reference])
    assert report["verdict"] == "FAIL"


# ------------------------------------------- the gate refusing to run at all
def test_same_arm_twice_is_invalid_not_a_pass():
    """The failure this gate most needs to resist.

    Comparing the normally-planned run against itself matches every digest. Without
    this check the result is an apparent pass that exercised nothing.
    """
    digests = {"w": "aa"}
    report = compare([_record(0, 1, digests)], [_record(0, 1, digests)])
    assert report["verdict"] == "INVALID_NOT_TWO_ARMS"
    assert report["mismatches"] == 0


def test_forgetting_force_full_pull_on_the_reference_is_invalid():
    digests = {"w": "aa"}
    report = compare(
        [_record(0, 1, digests, forced=True)], [_record(0, 1, digests, forced=True)]
    )
    assert report["verdict"] == "INVALID_NOT_TWO_ARMS"


def test_swapped_arms_are_invalid():
    """Subject forced and reference normal is the comparison backwards."""
    digests = {"w": "aa"}
    report = compare(
        [_record(0, 1, digests, forced=True)], [_record(0, 1, digests)]
    )
    assert report["verdict"] == "INVALID_NOT_TWO_ARMS"


def test_no_overlapping_rank_or_step_is_not_a_pass():
    report = compare(
        [_record(0, 1, {"w": "aa"})], [_record(7, 9, {"w": "aa"}, forced=True)]
    )
    assert report["verdict"] == "NO_COMPARABLE_RECORDS"
    assert report["compared_params"] == 0


def test_empty_records_are_not_a_pass():
    """Matching pair, nothing inside it: the same trap one level down."""
    report = compare([_record(0, 1, {})], [_record(0, 1, {}, forced=True)])
    assert report["verdict"] == "NO_COMPARABLE_RECORDS"


def test_empty_inputs_are_not_a_pass():
    assert compare([], [])["verdict"] == "INVALID_NOT_TWO_ARMS"


# --------------------------------------------------------------- coverage diff
def test_parameter_present_in_only_one_arm_is_reported_separately():
    """A coverage difference is a different finding from a byte difference.

    Forcing full pulls must not change *which* parameters get installed, so this
    being non-zero is a finding in its own right - and one whose fix is unrelated
    to a mis-sliced tensor's.
    """
    subject = _record(0, 1, {"shared": "aa", "subject_only": "bb"})
    reference = _record(0, 1, {"shared": "aa", "reference_only": "cc"}, forced=True)
    report = compare([subject], [reference])
    assert report["verdict"] == "PASS"  # no *digest* disagreed
    assert report["only_in_subject"] == 1
    assert report["only_in_reference"] == 1
    assert report["compared_params"] == 1


# ------------------------------------- reference arm that is not independent
def test_unbounded_reference_source_downgrades_the_pass():
    """The trap one level below the arms check.

    ``plan_transfer`` preserves the exact plan when a source cannot be bounded into
    a full pull, so forcing full pulls does not guarantee one. Those sources were
    fetched the same way in both arms, and calling their agreement a pass would
    claim coverage the run does not have.
    """
    digests = {"w": "aa", "x": "bb"}
    report = compare(
        [_record(0, 1, digests)],
        [_record(0, 1, digests, forced=True, unbounded=["mlp.down_proj.weight"])],
    )
    assert report["verdict"] == "PASS_PARTIAL"
    assert report["reference_not_independent"] == 1
    assert "vacuous" in report["reason"]


def test_clean_reference_gives_an_unqualified_pass():
    digests = {"w": "aa"}
    report = compare([_record(0, 1, digests)], [_record(0, 1, digests, forced=True)])
    assert report["verdict"] == "PASS"
    assert report["reference_not_independent"] == 0
    assert "reason" not in report


def test_unbounded_sources_are_reported_even_on_a_failure():
    """A later fix must not be able to claim this run verified them."""
    digests = {f"p{i}": f"same{i}" for i in range(10)}
    report = compare(
        [_record(0, 1, dict(digests, p4="DIFFERENT"))],
        [_record(0, 1, digests, forced=True, unbounded=["s1", "s2"])],
    )
    assert report["verdict"] == "FAIL"
    assert report["reference_not_independent"] == 2


def test_fallback_disagreement_between_arms_is_flagged():
    """A source refit in one arm and skipped in the other differs for a reason
    that has nothing to do with slicing."""
    report = compare(
        [_record(0, 1, {"w": "aa"}, fallback=["skipped.weight"])],
        [_record(0, 1, {"w": "aa"}, forced=True, fallback=[])],
    )
    assert report["fallback_disagreement"] == 1
    assert "not refit at all" in report["fallback_warning"]


def test_matching_fallback_sets_are_not_flagged():
    shared = ["skipped.weight"]
    report = compare(
        [_record(0, 1, {"w": "aa"}, fallback=shared)],
        [_record(0, 1, {"w": "aa"}, forced=True, fallback=shared)],
    )
    assert report["fallback_disagreement"] == 0
    assert "fallback_warning" not in report


# ------------------------------------------------------------------------ CLI
def test_cli_exit_code_reflects_the_verdict(tmp_path):
    digests = {"w": "aa"}
    subject = tmp_path / "subject.log"
    reference = tmp_path / "reference.log"
    subject.write_text(_log([_record(0, 1, digests)]))
    reference.write_text(_log([_record(0, 1, digests, forced=True)]))
    assert main([str(subject), str(reference)]) == 0

    subject.write_text(_log([_record(0, 1, {"w": "DIFFERENT"})]))
    assert main([str(subject), str(reference)]) == 1


def test_cli_signals_partial_coverage_distinctly(tmp_path):
    """3, not 0 and not 1: nothing disagreed, but not everything was checked."""
    digests = {"w": "aa"}
    subject = tmp_path / "subject.log"
    reference = tmp_path / "reference.log"
    subject.write_text(_log([_record(0, 1, digests)]))
    reference.write_text(
        _log([_record(0, 1, digests, forced=True, unbounded=["unbounded.weight"])])
    )
    assert main([str(subject), str(reference)]) == 3


def test_cli_usage_error_is_distinct_from_a_failed_gate():
    assert main([]) == 2


# ------------------------------------------------ digest over real tensors
def test_digest_destination_keys_by_param_and_is_order_stable():
    torch = pytest.importorskip("torch")
    from modelexpress.refit.reshard.verify import digest_destination

    buffers = {
        "b.weight": torch.arange(2048, dtype=torch.int32),
        "a.weight": torch.arange(2048, dtype=torch.int32) + 1,
    }
    digests = digest_destination(buffers)
    assert list(digests) == ["a.weight", "b.weight"]
    assert digests["a.weight"] != digests["b.weight"]
    # Re-digesting the same buffers must reproduce the record exactly, or a
    # comparison between two runs would report noise as corruption.
    assert digest_destination(buffers) == digests


def test_round_trip_from_emitted_log_line_to_verdict():
    """The loop the gate actually runs in, with no shared in-memory state.

    Digest real buffers, emit exactly as the receiver does, re-read from text, and
    compare. This is what catches the marker or schema constant drifting on one side
    only - a break that would leave the comparator finding nothing to check.
    """
    torch = pytest.importorskip("torch")
    from modelexpress.refit.reshard.verify import digest_destination

    buffers = {
        "layer.0.weight": torch.arange(2048, dtype=torch.int32),
        "layer.1.weight": torch.arange(2048, dtype=torch.int32) * 3,
    }
    # Both arms see identical bytes, which is the passing case.
    subject_line = RECORD_MARKER + json.dumps(
        dest_digest_record(
            step=1, rank=0, forced_full_pull=False,
            digests=digest_destination(buffers),
        )
    )
    reference_line = RECORD_MARKER + json.dumps(
        dest_digest_record(
            step=1, rank=0, forced_full_pull=True,
            digests=digest_destination(buffers),
        )
    )
    report = compare(
        parse_records("noise\n" + subject_line), parse_records(reference_line)
    )
    assert report["verdict"] == "PASS"
    assert report["compared_params"] == 2

    # Now corrupt one buffer on the subject side only, as a mis-sliced exact fetch
    # would, and confirm the same loop names it.
    corrupted = dict(buffers)
    corrupted["layer.1.weight"] = torch.roll(buffers["layer.1.weight"].clone(), 16)
    bad_line = RECORD_MARKER + json.dumps(
        dest_digest_record(
            step=1, rank=0, forced_full_pull=False,
            digests=digest_destination(corrupted),
        )
    )
    report = compare(parse_records(bad_line), parse_records(reference_line))
    assert report["verdict"] == "FAIL"
    assert [d["param"] for d in report["detail"]] == ["layer.1.weight"]


def test_digest_destination_catches_a_wrongly_placed_write():
    """The bug class this exists for: right bytes, wrong offset.

    A mis-sliced exact fetch writes correct-looking values at the wrong position,
    which is invisible to any order-independent statistic.
    """
    torch = pytest.importorskip("torch")
    from modelexpress.refit.reshard.verify import digest_destination

    good = torch.arange(4096, dtype=torch.int32)
    shifted = torch.roll(good.clone(), 8)
    assert (
        digest_destination({"w": good})["w"] != digest_destination({"w": shifted})["w"]
    )
    assert good.sum() == shifted.sum()  # a checksum would have missed it


# ------------------------------------------- source-verify precondition
# These cover the failure that nearly turned the first real pairing into a filed
# planner bug: the only step the two arms shared was a step whose *sources* had
# already failed verify_full_pulls, so localised destination mismatches were being
# read as mis-sliced tensors.
def _verify(step, mismatches, *, checked=6192, rank=0):
    return {
        "schema": "refit-verify-v1",
        "step": step,
        "rank": rank,
        "checked": checked,
        "mismatches": mismatches,
        "detail": [],
    }


def _verify_log(records):
    return "\n".join(
        "WARNING 2026-07-29 MX_REFIT_VERIFY " + json.dumps(r) for r in records
    )


def test_parses_verify_records():
    from modelexpress.refit.reshard.dest_digest_report import parse_verify_records

    text = _verify_log([_verify(1, 0), _verify(3, 118)])
    parsed = parse_verify_records(text)
    assert [(r["step"], r["mismatches"]) for r in parsed] == [(1, 0), (3, 118)]


def test_parse_verify_ignores_other_schemas():
    from modelexpress.refit.reshard.dest_digest_report import parse_verify_records

    assert parse_verify_records('MX_REFIT_VERIFY {"schema": "something-else"}') == []


def test_dirty_step_is_excluded_and_a_clean_step_still_decides():
    """A mismatch on a dirty step must not count; a clean step must still be judged."""
    subject = [
        _record(0, 1, {"a": "d1"}),
        _record(0, 3, {"a": "WRONG"}),
    ]
    reference = [
        _record(0, 1, {"a": "d1"}, forced=True),
        _record(0, 3, {"a": "d3"}, forced=True),
    ]
    report = compare(
        subject,
        reference,
        subject_verify=[_verify(1, 0), _verify(3, 360)],
        reference_verify=[_verify(1, 0), _verify(3, 118)],
    )
    assert report["verdict"] == "PASS"
    assert report["compared_records"] == 1
    assert [row["step"] for row in report["excluded_dirty_steps"]] == [3]
    assert report["excluded_dirty_steps"][0]["subject_source_mismatches"] == 360
    assert report["excluded_dirty_steps"][0]["reference_source_mismatches"] == 118


def test_all_shared_steps_dirty_is_invalid_not_a_failure():
    """The real 2026-07-29 shape: one shared step, and it was dirty in both arms."""
    report = compare(
        [_record(0, 3, {"a": "x"})],
        [_record(0, 3, {"a": "y"}, forced=True)],
        subject_verify=[_verify(3, 360)],
        reference_verify=[_verify(3, 118)],
    )
    assert report["verdict"] == "INVALID_NO_CLEAN_STEP"
    assert report["mismatches"] == 0
    assert "step 3" in report["reason"]
    assert report["source_verify_checked"] is True


def test_a_step_dirty_in_only_one_arm_is_still_excluded():
    """Inconsistent sources in either arm are enough to make the pair unusable."""
    report = compare(
        [_record(0, 4, {"a": "x"})],
        [_record(0, 4, {"a": "y"}, forced=True)],
        subject_verify=[_verify(4, 0)],
        reference_verify=[_verify(4, 7)],
    )
    assert report["verdict"] == "INVALID_NO_CLEAN_STEP"


def test_verify_records_from_other_steps_do_not_exclude_anything():
    # Step 1 because the trajectory precondition only admits the first refit on a
    # record with no source digests; the step number is incidental to this test.
    report = compare(
        [_record(0, 1, {"a": "same"})],
        [_record(0, 1, {"a": "same"}, forced=True)],
        subject_verify=[_verify(3, 999)],
        reference_verify=[_verify(3, 999)],
    )
    assert report["verdict"] == "PASS"
    assert report["excluded_dirty_steps"] == []


def test_fail_without_verify_records_is_marked_unattributable():
    """Omitting verify records must not silently restore the old, wrong reading."""
    report = compare(
        [_record(0, 1, {"a": "x", "b": "same"})],
        [_record(0, 1, {"a": "y", "b": "same"}, forced=True)],
    )
    assert report["verdict"] == "FAIL"
    assert report["source_verify_checked"] is False
    assert "not attributable" in report["attribution_warning"]


def test_pass_without_verify_records_carries_no_warning():
    report = compare(
        [_record(0, 1, {"a": "same"})],
        [_record(0, 1, {"a": "same"}, forced=True)],
    )
    assert report["verdict"] == "PASS"
    assert "attribution_warning" not in report


def test_cli_exit_code_distinguishes_unmeasurable_from_failed(tmp_path, capsys):
    """Exit 4, not 1: "could not be measured" is not "the planner is wrong"."""
    subj = tmp_path / "s.log"
    ref = tmp_path / "r.log"
    subj.write_text(
        _log([_record(0, 3, {"a": "x"})]) + "\n" + _verify_log([_verify(3, 360)])
    )
    ref.write_text(
        _log([_record(0, 3, {"a": "y"}, forced=True)])
        + "\n"
        + _verify_log([_verify(3, 118)])
    )
    code = main([str(subj), str(ref)])
    assert code == 4
    assert json.loads(capsys.readouterr().out)["verdict"] == "INVALID_NO_CLEAN_STEP"


# ------------------------------------------------- source digests, single run
# The cross-run comparison above is only sound where both arms hold identical
# source weights, which after one training step they do not: two runs differing in
# nothing disagreed on 10 of 4,350 param-steps, against 1 for the two runs
# differing in the path under test. These cover the field that lets one run be
# checked on its own, and they weight heavily toward the false-positive directions,
# because an audit that cries staleness over ordinary training is worse than none.
def _rec2(rank, step, digests, source_digests):
    return dest_digest_record(
        step=step,
        rank=rank,
        forced_full_pull=False,
        digests=digests,
        source_digests=source_digests,
    )


def test_record_carries_source_digests_and_stays_v2():
    record = _rec2(0, 1, {"w": "d"}, {"w": "s"})
    assert record["schema"] == SCHEMA
    assert record["source_digests"] == {"w": "s"}


def test_v1_record_without_source_digests_still_parses():
    """The only pairing we have is v1, and re-reading it is how the noise floor
    was measured; rejecting it would strand that evidence."""
    v1 = dict(_record(0, 1, {"w": "d"}), schema=SCHEMA_V1)
    v1.pop("source_digests", None)
    assert len(parse_records(_log([v1]))) == 1


def test_record_not_last_on_line_is_parsed_not_dropped():
    """A Ray driver log appends its dedup suffix after the JSON. This used to
    raise and be swallowed, shrinking coverage with no signal."""
    text = (
        "WARNING MX_REFIT_DEST_DIGEST "
        + json.dumps(_rec2(0, 1, {"w": "d"}, {"w": "s"}))
        + " [repeated 15x across cluster]"
    )
    records, skipped = parse_records_with_skips(text)
    assert len(records) == 1
    assert skipped == 0


def test_audit_passes_when_destination_tracks_its_sources():
    """Ordinary training: the weight moves and the publisher claim moves with it."""
    records = [
        _rec2(0, 1, {"w": "d1"}, {"w": "s1"}),
        _rec2(0, 2, {"w": "d2"}, {"w": "s2"}),
        _rec2(0, 3, {"w": "d3"}, {"w": "s3"}),
    ]
    report = audit_freshness(records)
    assert report["verdict"] == "PASS"
    assert report["params_audited"] == 1


def test_audit_passes_when_nothing_moves():
    """At lr=3e-7 in bf16 most params never move; that is not a finding."""
    records = [
        _rec2(0, step, {"w": "d1"}, {"w": "s1"}) for step in (1, 2, 3, 4, 5)
    ]
    assert audit_freshness(records)["verdict"] == "PASS"


def test_audit_flags_destination_moving_while_sources_hold_still():
    """Nothing in the superset of readable shards changed, so the assembled bytes
    had no business changing."""
    records = [
        _rec2(0, 1, {"w": "d1"}, {"w": "s1"}),
        _rec2(0, 2, {"w": "CHANGED"}, {"w": "s1"}),
    ]
    report = audit_freshness(records)
    assert report["verdict"] == "FAIL"
    assert report["dest_moved_alone_count"] == 1
    assert report["dest_moved_alone"][0]["param"] == "w"


def test_audit_flags_destination_reverting_to_an_earlier_step():
    """The staleness signature, and the reason this field was added: the v1 schema
    could not say whether the trainer's own copy had reverted too."""
    records = [
        _rec2(0, 3, {"w": "A"}, {"w": "s3"}),
        _rec2(0, 4, {"w": "B"}, {"w": "s4"}),
        _rec2(0, 5, {"w": "A"}, {"w": "s5"}),
    ]
    report = audit_freshness(records)
    assert report["verdict"] == "FAIL"
    assert report["dest_reverted_count"] == 1


def test_audit_accepts_reversion_the_sources_also_made():
    """A bf16 element on a rounding boundary can flip back. If the publisher claim
    returns to its earlier value too, the destination doing so is correct."""
    records = [
        _rec2(0, 3, {"w": "A"}, {"w": "sA"}),
        _rec2(0, 4, {"w": "B"}, {"w": "sB"}),
        _rec2(0, 5, {"w": "A"}, {"w": "sA"}),
    ]
    assert audit_freshness(records)["verdict"] == "PASS"


def test_audit_does_not_fail_on_the_advisory_finding():
    """A shard changing where this rank does not read looks exactly like a missed
    update, and the superset fingerprint cannot separate them."""
    records = [
        _rec2(0, 1, {"w": "d1"}, {"w": "s1"}),
        _rec2(0, 2, {"w": "d1"}, {"w": "s2"}),
    ]
    report = audit_freshness(records)
    assert report["verdict"] == "PASS"
    assert report["source_moved_dest_static_count"] == 1
    assert "advisory" in report


def test_audit_reports_no_evidence_rather_than_passing_a_v1_run():
    records = [_record(0, 1, {"w": "d1"}), _record(0, 2, {"w": "d2"})]
    report = audit_freshness(records)
    assert report["verdict"] == "NO_EVIDENCE"
    assert report["params_audited"] == 0
    assert report["no_evidence"] == 1


def test_audit_excludes_params_whose_sources_carried_no_digest():
    """Publishers predating the digest must degrade to no evidence, not a pass."""
    records = [
        _rec2(0, 1, {"a": "d1", "b": "e1"}, {"a": "s1", "b": None}),
        _rec2(0, 2, {"a": "d2", "b": "e2"}, {"a": "s2", "b": None}),
    ]
    report = audit_freshness(records)
    assert report["verdict"] == "PASS"
    assert report["params_audited"] == 1
    assert report["no_evidence"] == 1


def test_audit_needs_two_steps_to_say_anything():
    assert audit_freshness([_rec2(0, 1, {"w": "d"}, {"w": "s"})])[
        "verdict"
    ] == "NO_EVIDENCE"


def test_audit_keeps_ranks_separate():
    """Rank 0 holding d1 while rank 1 holds d2 is different shards, not a change."""
    records = [
        _rec2(0, 1, {"w": "d1"}, {"w": "s1"}),
        _rec2(1, 1, {"w": "d2"}, {"w": "s2"}),
        _rec2(0, 2, {"w": "d1"}, {"w": "s1"}),
        _rec2(1, 2, {"w": "d2"}, {"w": "s2"}),
    ]
    assert audit_freshness(records)["verdict"] == "PASS"


# --- the trajectory precondition ------------------------------------------------
# Two arms only test the planner if they were looking at the same weights. On the
# 2026-07-30 rig they were not: two runs with identical settings disagreed on 10 of
# 4 350 parameters, against the 1 mismatch the cross-arm comparison reported. These
# tests pin the gate refusing that comparison instead of attributing it.


def _rec_src(rank, step, digests, sources, *, forced=False):
    return dest_digest_record(
        step=step,
        rank=rank,
        forced_full_pull=forced,
        digests=digests,
        source_digests=sources,
    )


def test_a_late_step_is_comparable_when_the_arms_share_their_sources():
    report = compare(
        [_rec_src(0, 5, {"a": "same"}, {"a": "src1"})],
        [_rec_src(0, 5, {"a": "same"}, {"a": "src1"}, forced=True)],
    )

    assert report["verdict"] == "PASS"
    assert report["compared_params"] == 1
    assert report["excluded_unsafe_pairs"] == []


def test_a_late_step_is_refused_when_the_arms_sources_moved_apart():
    """The noise-floor case: both arms clean, both arms different."""
    report = compare(
        [_rec_src(0, 5, {"a": "x"}, {"a": "src1"})],
        [_rec_src(0, 5, {"a": "y"}, {"a": "src2"}, forced=True)],
    )

    assert report["verdict"] == "INVALID_NO_TRAJECTORY_SAFE_STEP"
    assert report["mismatches"] == 0
    assert "indistinguishable from" in report["reason"]


def test_a_moved_source_is_not_reported_as_a_planner_defect():
    """Without this the differing digest reads as FAIL, which is how the first
    pairing came within one reading of being filed as a plan_pull bug."""
    report = compare(
        [_rec_src(0, 5, {"a": "x"}, {"a": "src1"})],
        [_rec_src(0, 5, {"a": "y"}, {"a": "src2"}, forced=True)],
    )

    assert report["verdict"] != "FAIL"


def test_first_refit_is_comparable_without_source_digests():
    """The archived v1 evidence is the only pairing we have, and its step 1 result
    stands: weights are the checkpoint as loaded in both arms."""
    report = compare(
        [_record(0, 1, {"a": "same"})],
        [_record(0, 1, {"a": "same"}, forced=True)],
    )

    assert report["verdict"] == "PASS"


def test_a_v1_late_step_is_refused():
    report = compare(
        [_record(0, 4, {"a": "same"})],
        [_record(0, 4, {"a": "same"}, forced=True)],
    )

    assert report["verdict"] == "INVALID_NO_TRAJECTORY_SAFE_STEP"
    assert "schema v1" in report["excluded_unsafe_pairs"][0]["reason"]


def test_a_safe_step_still_reports_a_real_mismatch():
    """The precondition must not become a way for a defect to escape."""
    report = compare(
        [_rec_src(0, 5, {"a": "x", "b": "same"}, {"a": "src1", "b": "src1"})],
        [
            _rec_src(
                0, 5, {"a": "y", "b": "same"}, {"a": "src1", "b": "src1"}, forced=True
            )
        ],
    )

    assert report["verdict"] == "FAIL"
    assert report["mismatches"] == 1


def test_surviving_pairs_are_compared_and_the_dropped_ones_named():
    report = compare(
        [
            _rec_src(0, 5, {"a": "same"}, {"a": "src1"}),
            _rec_src(1, 5, {"a": "x"}, {"a": "src1"}),
        ],
        [
            _rec_src(0, 5, {"a": "same"}, {"a": "src1"}, forced=True),
            _rec_src(1, 5, {"a": "y"}, {"a": "src9"}, forced=True),
        ],
    )

    assert report["verdict"] == "PASS_PARTIAL"
    assert report["compared_params"] == 1
    assert report["excluded_unsafe_pairs"][0]["rank"] == 1
    assert "verified only for the pairs that survived" in report["reason"]


def test_arms_sharing_no_source_parameter_are_refused():
    report = compare(
        [_rec_src(0, 5, {"a": "same"}, {"a": "src1"})],
        [_rec_src(0, 5, {"a": "same"}, {"zz": "src1"}, forced=True)],
    )

    assert report["verdict"] == "INVALID_NO_TRAJECTORY_SAFE_STEP"
    assert "share no source parameter" in report["excluded_unsafe_pairs"][0]["reason"]


def test_trajectory_refusal_exits_four_not_one(tmp_path, capsys):
    """Exit 1 would file an unmeasurable pairing as a planner defect."""
    subj = tmp_path / "s.log"
    ref = tmp_path / "r.log"
    subj.write_text(_log([_rec_src(0, 5, {"a": "x"}, {"a": "src1"})]))
    ref.write_text(_log([_rec_src(0, 5, {"a": "y"}, {"a": "src2"}, forced=True)]))

    assert main([str(subj), str(ref)]) == 4
    assert json.loads(capsys.readouterr().out)["mismatches"] == 0


# --- the verify-record reader must not go blind on a busy fleet -----------------


def _verify_line(rec, suffix=""):
    return "MX_REFIT_VERIFY " + json.dumps(rec) + suffix


def _vrec(step=1, rank=0, schema="refit-verify-v2", **kw):
    rec = {"schema": schema, "step": step, "checked": 10, "mismatches": 0, "detail": []}
    if rank is not None:
        rec["rank"] = rank
    rec.update(kw)
    return rec


def test_verify_reader_accepts_the_v2_record_with_a_rank():
    got = parse_verify_records(_verify_line(_vrec(rank=7)))

    assert len(got) == 1
    assert got[0]["rank"] == 7


def test_verify_reader_still_accepts_a_v1_record():
    """A fleet is upgraded one image at a time, so mixed-version logs are normal."""
    got = parse_verify_records(_verify_line(_vrec(schema="refit-verify-v1", rank=None)))

    assert len(got) == 1
    assert "rank" not in got[0]


def test_verify_reader_keeps_a_record_with_a_ray_dedup_suffix():
    """The regression. json.loads over the rest of the line raises on the suffix and
    the record vanishes - on exactly the runs with the most ranks."""
    line = _verify_line(_vrec(rank=3), suffix=" [repeated 15x across cluster]")

    got = parse_verify_records(line)

    assert len(got) == 1, "a Ray-deduplicated verify record was dropped"
    assert got[0]["rank"] == 3


def test_verify_reader_keeps_records_from_every_rank():
    text = "\n".join(
        _verify_line(_vrec(step=2, rank=r), suffix=" [repeated 2x across cluster]")
        for r in range(16)
    )

    got = parse_verify_records(text)

    assert len({r["rank"] for r in got}) == 16


def test_verify_reader_ignores_a_foreign_schema():
    got = parse_verify_records(_verify_line(_vrec(schema="refit-verify-v99")))

    assert got == []
