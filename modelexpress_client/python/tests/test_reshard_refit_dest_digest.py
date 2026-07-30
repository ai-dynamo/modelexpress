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
    compare,
    dest_digest_record,
    main,
    parse_records,
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
