# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""The replay summary's job is to make a thin pass look thin, so most of these
tests are about coverage rather than about catching a mismatch."""

from __future__ import annotations

import json

from modelexpress.refit.reshard.exact_replay_report import (
    RECORD_MARKER,
    SCHEMA,
    main,
    parse_records,
    summarise,
)


def _rec(rank, step, checked, mismatches=0, detail=None, **gaps):
    body = {
        "schema": SCHEMA,
        "rank": rank,
        "step": step,
        "checked": checked,
        "mismatches": mismatches,
        "detail": detail or [],
        "params": checked,
        "uncovered_params": 0,
        "params_with_unstaged_sources": 0,
        "copies_the_exact_path_could_not_plan": 0,
        "segments_outside_any_staged_shard": 0,
        "forced_full_pull": True,
    }
    body.update(gaps)
    return body


def _log(*records):
    return "\n".join(f"prefix {RECORD_MARKER}{json.dumps(r)}" for r in records)


def test_clean_run_passes_and_reports_what_it_covered():
    report = summarise([_rec(0, 1, 400), _rec(1, 1, 400)])

    assert report["verdict"] == "PASS"
    assert report["checked"] == 800
    assert report["ranks"] == [0, 1]
    assert "not wire execution" in report["reason"]


def test_a_disagreement_fails_and_is_named():
    bad = [{"param": "layers.3.w13", "exact_replay": "aa", "received": "bb"}]
    report = summarise([_rec(0, 1, 400), _rec(1, 1, 400, mismatches=1, detail=bad)])

    assert report["verdict"] == "FAIL"
    assert report["mismatches"] == 1
    assert report["detail"] == bad
    assert "plan_pull" in report["reason"]


def test_zero_coverage_is_no_evidence_not_a_pass():
    """The gate's real failure mode: records present, nothing compared."""
    report = summarise([_rec(0, 1, 0), _rec(1, 1, 0)])

    assert report["verdict"] == "NO_EVIDENCE"
    assert report["mismatches"] == 0
    assert "not a pass" in report["reason"]


def test_one_blind_rank_downgrades_an_otherwise_clean_run():
    report = summarise([_rec(0, 1, 400), _rec(1, 1, 0)])

    assert report["verdict"] == "PASS_PARTIAL"
    assert report["checked"] == 400
    assert {"rank": 1, "step": 1} in report["rank_steps_with_no_coverage"]


def test_unmapped_segments_downgrade_even_with_full_agreement():
    report = summarise(
        [_rec(0, 1, 400, segments_outside_any_staged_shard=3), _rec(1, 1, 400)]
    )

    assert report["verdict"] == "PASS_PARTIAL"
    assert report["coverage_gaps"]["segments_outside_any_staged_shard"] == 3


def test_unplannable_copies_downgrade_too():
    report = summarise(
        [_rec(0, 1, 400, copies_the_exact_path_could_not_plan=2), _rec(1, 1, 400)]
    )

    assert report["verdict"] == "PASS_PARTIAL"


def test_partly_staged_sources_are_counted_but_do_not_downgrade_alone():
    """A source shared with a fallback param is expected to be partly staged; it is
    reported so the denominator can be judged, not treated as a finding."""
    report = summarise(
        [_rec(0, 1, 400, params_with_unstaged_sources=5), _rec(1, 1, 400)]
    )

    assert report["verdict"] == "PASS"
    assert report["coverage_gaps"]["params_with_unstaged_sources"] == 5


def test_an_empty_log_is_no_evidence():
    report = summarise([])

    assert report["verdict"] == "NO_EVIDENCE"
    assert report["checked"] == 0


def test_a_retried_step_is_not_counted_twice():
    """A refit that retries re-emits the record for one step. Summing those would
    inflate the denominator and make coverage look better than it was."""
    report = summarise([_rec(0, 3, 400), _rec(0, 3, 400), _rec(0, 3, 400)])

    assert report["checked"] == 400
    assert report["rank_step_pairs"] == 1
    assert report["records"] == 3


def test_a_run_without_forced_full_pull_is_flagged():
    report = summarise([_rec(0, 1, 12, forced_full_pull=False)])

    assert report["forced_full_pull"] == [False]


def test_records_survive_a_ray_dedup_suffix():
    text = _log(_rec(0, 1, 400)) + " [repeated 15x across cluster]"
    records, skipped = parse_records(text)

    assert len(records) == 1
    assert skipped == 0


def test_a_foreign_schema_is_skipped_not_summarised():
    stale = dict(_rec(0, 1, 400), schema="refit-exact-replay-v0")
    records, skipped = parse_records(_log(stale))

    assert records == []
    assert skipped == 1


def test_truncated_json_is_counted_not_crashed():
    records, skipped = parse_records(f"x {RECORD_MARKER}" + '{"schema": "refit-')

    assert records == []
    assert skipped == 1


def test_lines_without_the_marker_are_ignored():
    records, skipped = parse_records("just a log line\nanother one\n")

    assert (records, skipped) == ([], 0)


def test_exit_codes_separate_a_defect_from_a_gap(tmp_path, capsys):
    clean = tmp_path / "clean.log"
    clean.write_text(_log(_rec(0, 1, 400), _rec(1, 1, 400)))
    assert main([str(clean)]) == 0

    thin = tmp_path / "thin.log"
    thin.write_text(_log(_rec(0, 1, 400), _rec(1, 1, 0)))
    assert main([str(thin)]) == 3

    broken = tmp_path / "broken.log"
    broken.write_text(_log(_rec(0, 1, 400, mismatches=1)))
    assert main([str(broken)]) == 1

    empty = tmp_path / "empty.log"
    empty.write_text("nothing here\n")
    assert main([str(empty)]) == 3

    assert main([]) == 2
    capsys.readouterr()


def test_skipped_records_are_reported_to_the_reader(tmp_path, capsys):
    log = tmp_path / "x.log"
    log.write_text(_log(_rec(0, 1, 400)) + f"\nz {RECORD_MARKER}" + '{"schema": "ref')

    main([str(log)])

    assert json.loads(capsys.readouterr().out)["records_skipped"] == 1


def test_a_rankless_record_is_refused_rather_than_undercounted():
    """An early receiver build omitted rank. De-duplicating retries by (rank, step)
    would then keep one of sixteen ranks and report a sixteenth of the coverage as
    if it were the whole run."""
    rankless = _rec(0, 1, 400)
    del rankless["rank"]

    report = summarise([rankless, _rec(1, 1, 400)])

    assert report["verdict"] == "INVALID_NO_RANK"
    assert report["checked"] == 0
    assert "sixteenth" not in report["reason"]  # says what happened, not our rig


def test_rankless_run_exits_nonzero(tmp_path, capsys):
    rankless = _rec(0, 1, 400)
    del rankless["rank"]
    log = tmp_path / "old.log"
    log.write_text(_log(rankless))

    assert main([str(log)]) == 1
    capsys.readouterr()
