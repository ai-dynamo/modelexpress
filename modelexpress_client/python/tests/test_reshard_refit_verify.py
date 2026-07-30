# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the parameter-equality gate.

The gate exists to catch a refit that installs the wrong bytes while reporting
healthy timings, so the tests that matter are the ones proving it *fails* when it
should: a corrupted byte, and - the case plain checksums miss - a pure permutation
of correct bytes.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modelexpress.refit.reshard.slice_plan import Shard  # noqa: E402
from modelexpress.refit.reshard.verify import (  # noqa: E402
    shard_region,
    source_expectation_digests,
    tensor_digest,
    verify_full_pulls,
)


class _Source:
    """Minimal stand-in for SourceInfo: only the fields the gate reads."""

    def __init__(self, global_shape, shards):
        self.global_shape = global_shape
        self.shards = shards


def _shard(offset, shape, digest=None, session="trainer-r0"):
    return Shard(
        shard_offset=offset,
        shape=shape,
        session=session,
        addr=0,
        elsize=2,
        digest=digest,
    )


# ------------------------------------------------------------------- the digest
def test_digest_is_stable_for_identical_bytes():
    a = torch.arange(4096, dtype=torch.int32)
    assert tensor_digest(a) == tensor_digest(a.clone())


def test_digest_changes_when_a_single_value_changes():
    a = torch.arange(4096, dtype=torch.int32)
    b = a.clone()
    b[1234] += 1
    assert tensor_digest(a) != tensor_digest(b)


def test_digest_detects_a_permutation():
    """The reason this is not a plain sum.

    A plan that copies the right bytes to the wrong offsets preserves every
    order-independent statistic, so a sum-based digest would call it equal.
    """
    a = torch.arange(8192, dtype=torch.int32)
    b = a.clone()
    # swap two whole digest rows, so the multiset of values is untouched
    row = 1024
    b[0:row], b[row : 2 * row] = a[row : 2 * row].clone(), a[0:row].clone()
    assert a.sum() == b.sum(), "precondition: sums must agree, else the test is trivial"
    assert tensor_digest(a) != tensor_digest(b)


def test_digest_covers_a_non_word_multiple_tail():
    """Sizes that are not a multiple of the row or word width must still be digested.

    An implementation that silently drops the ragged tail would report equal for
    tensors differing only in their last bytes.
    """
    a = torch.arange(1024 + 7, dtype=torch.int16)
    b = a.clone()
    b[-1] += 1
    assert tensor_digest(a) != tensor_digest(b)


def test_digest_handles_a_strided_input():
    dense = torch.arange(64, dtype=torch.int32).reshape(8, 8)
    strided = dense[:, ::2]
    assert tensor_digest(strided) == tensor_digest(strided.contiguous())


# -------------------------------------------------------------------- the region
def test_shard_region_extracts_the_publishers_box():
    full = torch.arange(24, dtype=torch.int32).reshape(4, 6)
    region = shard_region(full.reshape(-1), (4, 6), (1, 2), (2, 3))
    assert torch.equal(region, full[1:3, 2:5])


# ---------------------------------------------------------------------- the gate
def test_matching_bytes_pass():
    full = torch.arange(64, dtype=torch.int16)
    digest = tensor_digest(shard_region(full, (64,), (0,), (64,)))
    report = verify_full_pulls(
        full_staging={"w": full},
        sources={"w": _Source((64,), [_shard((0,), (64,), digest)])},
    )
    assert report == {
        "checked": 1,
        "skipped_no_digest": 0,
        "mismatches": 0,
        "detail": [],
        "detail_truncated": False,
        "divergent_replicas": 0,
        "divergent_detail": [],
        # No fresh table supplied, so the gate fell back to the prepare-time
        # digests. Reported rather than implied: whether the expectation was
        # current decides whether a clean report means anything on a later step.
        "digests_refreshed": 0,
        "digests_refreshed_via_replica": 0,
        "digest_source": "prepare",
        # No publisher step stamps supplied, so freshness falls back to the inference
        # over refreshed digests. Named in the report so a reader can tell which of the
        # two the verdict rests on - they are not equally trustworthy.
        "mismatches_from_stale_publishers": 0,
        "attributable_mismatches": 0,
        "stale_publisher_sessions": [],
        "freshness_evidence": "refresh_inference",
        # No step passed, so the gate makes no claim about how old the reference is.
        "reference_is_current": True,
    }


def test_corrupted_bytes_are_reported_with_the_source_name():
    full = torch.arange(64, dtype=torch.int16)
    digest = tensor_digest(shard_region(full, (64,), (0,), (64,)))
    full[5] += 1  # corrupt after the publisher digested
    report = verify_full_pulls(
        full_staging={"w": full},
        sources={"w": _Source((64,), [_shard((0,), (64,), digest)])},
    )
    assert report["checked"] == 1
    assert report["mismatches"] == 1
    assert report["detail"][0]["source"] == "w"
    assert report["detail"][0]["session"] == "trainer-r0"


def test_only_the_wrong_shard_is_blamed():
    """Fan-in: with several publishers per tensor the report must localise which."""
    full = torch.arange(64, dtype=torch.int16)
    good = tensor_digest(shard_region(full, (64,), (0,), (32,)))
    bad = tensor_digest(shard_region(full, (64,), (32,), (32,)))
    full[40] += 1  # only the second shard's region
    report = verify_full_pulls(
        full_staging={"w": full},
        sources={
            "w": _Source(
                (64,),
                [
                    _shard((0,), (32,), good, session="trainer-r0"),
                    _shard((32,), (32,), bad, session="trainer-r1"),
                ],
            )
        },
    )
    assert report["checked"] == 2
    assert report["mismatches"] == 1
    assert report["detail"][0]["session"] == "trainer-r1"
    assert report["detail"][0]["shard_offset"] == [32]


def test_a_publisher_without_a_digest_is_skipped_not_failed():
    """A mixed fleet must degrade to "no evidence", never to a false failure.

    Note this is a statement about the *report*, not about the caller: the caller
    is required to treat ``checked == 0`` as a failure, because a report of zero
    mismatches over zero checks is indistinguishable from a pass. See
    ``VERIFY_STRICT`` and the receiver's use of it.
    """
    full = torch.arange(64, dtype=torch.int16)
    report = verify_full_pulls(
        full_staging={"w": full},
        sources={"w": _Source((64,), [_shard((0,), (64,), None)])},
    )
    assert report["checked"] == 0
    assert report["skipped_no_digest"] == 1
    assert report["mismatches"] == 0


@pytest.mark.parametrize(
    "env, strict",
    [(None, True), ("1", True), ("0", False)],
)
def test_strict_is_the_default_so_an_unrunnable_gate_cannot_read_as_a_pass(
    monkeypatch, env, strict
):
    """The whole point of Bug 5: an all-skipped report must not read as a pass.

    The enforcement lives at the caller, so what matters here is the default. If
    someone flips it back to permissive, the runs that quietly turn green are
    precisely the ones proving nothing.
    """
    import importlib

    import modelexpress.refit.reshard.verify as verify_mod

    if env is None:
        monkeypatch.delenv("MX_RESHARD_VERIFY_STRICT", raising=False)
    else:
        monkeypatch.setenv("MX_RESHARD_VERIFY_STRICT", env)

    assert importlib.reload(verify_mod).VERIFY_STRICT is strict
    monkeypatch.undo()
    importlib.reload(verify_mod)


def test_mismatch_detail_is_capped_but_the_count_is_not():
    """A systematically wrong plan must not emit one log line per tensor - but it
    must still be distinguishable from a handful of bad shards.

    The count and the sample are different things. Reporting ``len(detail)`` made
    an entirely wrong refit read as "20 of 6144", which is the difference between
    a curiosity and a stop-everything result.
    """
    full = torch.arange(64, dtype=torch.int16)
    wrong = tensor_digest(torch.zeros(32, dtype=torch.int16))
    sources = {
        f"w{i}": _Source((64,), [_shard((0,), (32,), wrong)]) for i in range(50)
    }
    report = verify_full_pulls(
        full_staging={f"w{i}": full for i in range(50)},
        sources=sources,
        max_report=5,
    )
    assert report["checked"] == 50
    assert len(report["detail"]) == 5, "the sample is capped"
    assert report["mismatches"] == 50, "the count is not"
    assert report["detail_truncated"] is True


def test_replicas_offering_different_bytes_are_reported_separately():
    """A mismatch and a replica disagreement have opposite fixes.

    If two ranks offer the same box with different digests, the receiver read one
    of them faithfully and is being compared against the other. Blaming the wire
    there sends the investigation to the transport, when the problem is upstream
    in what the publishers hold.
    """
    full = torch.arange(64, dtype=torch.int16)
    good = tensor_digest(shard_region(full, (64,), (0,), (64,)))
    other = tensor_digest(torch.zeros(64, dtype=torch.int16))
    report = verify_full_pulls(
        full_staging={"w": full},
        sources={
            "w": _Source(
                (64,),
                [
                    _shard((0,), (64,), good, session="r0"),
                    _shard((0,), (64,), other, session="r1"),
                ],
            )
        },
    )

    assert report["divergent_replicas"] == 1
    assert report["divergent_detail"][0]["offers"] == {good: ["r0"], other: ["r1"]}


def test_agreeing_replicas_are_not_reported():
    """The common case must stay silent, or the signal is worthless."""
    full = torch.arange(64, dtype=torch.int16)
    good = tensor_digest(shard_region(full, (64,), (0,), (64,)))
    report = verify_full_pulls(
        full_staging={"w": full},
        sources={
            "w": _Source(
                (64,),
                [
                    _shard((0,), (64,), good, session="r0"),
                    _shard((0,), (64,), good, session="r1"),
                ],
            )
        },
    )

    assert report["divergent_replicas"] == 0
    assert report["mismatches"] == 0


def test_sentinel_fill_separates_never_written_from_written_wrong(monkeypatch):
    """The staging arena is reused across steps, so an unwritten region holds the
    previous step's weights - plausible values that mismatch. Pre-filling with a
    sentinel is what tells "the wire skipped this" apart from "the wire delivered
    the wrong bytes", and those have entirely different fixes.
    """
    import importlib

    import modelexpress.refit.reshard.verify as verify_mod

    monkeypatch.setenv("MX_RESHARD_FILL_SENTINEL", "1")
    v = importlib.reload(verify_mod)

    expected = tensor_digest(torch.arange(64, dtype=torch.int16))

    # Never written: still entirely sentinel after the wire.
    untouched = torch.zeros(64, dtype=torch.int16)
    v.fill_sentinel({"w": untouched})
    report = v.verify_full_pulls(
        full_staging={"w": untouched},
        sources={"w": _Source((64,), [_shard((0,), (64,), expected)])},
    )
    assert report["mismatches"] == 1
    assert report["never_written"] == 1
    assert report["mean_sentinel_frac"] == 1.0

    # Written, but wrong: no sentinel left, so the wire did touch it.
    wrong = torch.zeros(64, dtype=torch.int16)
    v.fill_sentinel({"w": wrong})
    wrong.copy_(torch.arange(100, 164, dtype=torch.int16))
    report = v.verify_full_pulls(
        full_staging={"w": wrong},
        sources={"w": _Source((64,), [_shard((0,), (64,), expected)])},
    )
    assert report["mismatches"] == 1
    assert report["never_written"] == 0
    assert report["mean_sentinel_frac"] == 0.0

    monkeypatch.undo()
    importlib.reload(verify_mod)


def test_sentinel_fields_are_absent_by_default():
    """The diagnostic costs a memset of the whole staging arena every step, so it
    must not appear in - or slow down - an ordinary verified run."""
    full = torch.arange(64, dtype=torch.int16)
    report = verify_full_pulls(
        full_staging={"w": full},
        sources={"w": _Source((64,), [_shard((0,), (64,), "deadbeef")])},
    )
    assert "never_written" not in report
    assert "mean_sentinel_frac" not in report


# -------------------------------------------------- Bug 9: stale expected digests
#
# `sources` comes from `_prepare()`, which runs once. Its digests therefore describe
# the weights as of the FIRST refit, and every later step compares current bytes
# against that frozen expectation - so a parameter training legitimately updated is
# reported as corruption.
#
# This is measured, not hypothetical. On Topology B (Qwen3-30B-A3B, PP2/EP8/DP8 ->
# TP2x8) the gate reported exactly one bad tensor,
# model.layers.18.self_attn.o_proj.weight, and across two runs its `want` was
# byte-identical while `got` tracked training. The addr-recheck diagnostic showed
# addr_changed 0 over ~4.4M comparisons with digest_changed 1 on precisely the ranks
# that flagged it: no source ever moved, the wire delivered current weights, and the
# gate was wrong.
def test_a_source_that_changed_since_prepare_is_not_a_mismatch():
    """The Bug 9 regression.

    Staging holds what the wire delivered - the CURRENT bytes. The prepare-time
    digest describes the old ones. Verifying against the fresh table must pass.
    """
    current = torch.arange(64, dtype=torch.int16)
    current_digest = tensor_digest(shard_region(current, (64,), (0,), (64,)))
    stale_digest = tensor_digest(torch.zeros(64, dtype=torch.int16))
    assert stale_digest != current_digest

    report = verify_full_pulls(
        full_staging={"w": current},
        sources={"w": _Source((64,), [_shard((0,), (64,), stale_digest)])},
        fresh_sources={"w": _Source((64,), [_shard((0,), (64,), current_digest)])},
    )
    assert report["mismatches"] == 0
    assert report["checked"] == 1
    assert report["digests_refreshed"] == 1
    assert report["digest_source"] == "fresh"


def test_without_the_fresh_table_the_same_case_is_reported_as_corruption():
    """Pins the old behaviour, so the fix is demonstrably the thing that changed."""
    current = torch.arange(64, dtype=torch.int16)
    stale_digest = tensor_digest(torch.zeros(64, dtype=torch.int16))
    report = verify_full_pulls(
        full_staging={"w": current},
        sources={"w": _Source((64,), [_shard((0,), (64,), stale_digest)])},
    )
    assert report["mismatches"] == 1
    assert report["digest_source"] == "prepare"
    assert report["digests_refreshed"] == 0


def test_genuinely_wrong_bytes_still_fail_against_a_fresh_table():
    """The fix must not become a way to pass by refreshing the expectation.

    Fresh digest describes the publisher's current bytes; staging holds something
    else. That is a real transport fault and must still fail.
    """
    delivered = torch.zeros(64, dtype=torch.int16)
    publisher = torch.arange(64, dtype=torch.int16)
    fresh_digest = tensor_digest(shard_region(publisher, (64,), (0,), (64,)))
    report = verify_full_pulls(
        full_staging={"w": delivered},
        sources={"w": _Source((64,), [_shard((0,), (64,), fresh_digest)])},
        fresh_sources={"w": _Source((64,), [_shard((0,), (64,), fresh_digest)])},
    )
    assert report["mismatches"] == 1
    assert report["detail"][0]["source"] == "w"


def test_refresh_is_keyed_by_session_and_offset_not_by_position():
    """Discovery order is not stable and replicas rotate.

    merge_shard_tables keeps the first offer of each geometry, so two discoveries
    legitimately pin the same box to different but byte-identical replicas. A
    positional match would pair a shard with another rank's digest - the mistake the
    first version of the addr-recheck probe made, which claimed up to 100% of
    addresses had moved.
    """
    current = torch.arange(64, dtype=torch.int16)
    d = tensor_digest(shard_region(current, (64,), (0,), (64,)))
    wrong = tensor_digest(torch.ones(64, dtype=torch.int16))
    # Fresh table lists the same two sessions in the opposite order.
    report = verify_full_pulls(
        full_staging={"w": current},
        sources={"w": _Source((64,), [_shard((0,), (64,), d, session="trainer-r1")])},
        fresh_sources={
            "w": _Source(
                (64,),
                [
                    _shard((0,), (64,), wrong, session="trainer-r0"),
                    _shard((0,), (64,), d, session="trainer-r1"),
                ],
            )
        },
    )
    assert report["mismatches"] == 0, "matched the wrong session's digest"


def test_a_source_absent_from_the_fresh_table_falls_back_to_prepare():
    """A shard the fresh discovery does not offer must not become unverifiable.

    Silently skipping it would shrink `checked` without saying so, and `checked` is
    the number a run is judged on.
    """
    current = torch.arange(64, dtype=torch.int16)
    d = tensor_digest(shard_region(current, (64,), (0,), (64,)))
    report = verify_full_pulls(
        full_staging={"w": current},
        sources={"w": _Source((64,), [_shard((0,), (64,), d)])},
        fresh_sources={"other": _Source((64,), [_shard((0,), (64,), d)])},
    )
    assert report["checked"] == 1
    assert report["mismatches"] == 0
    assert report["digests_refreshed"] == 0


def test_publishers_without_digests_are_still_skipped_not_invented():
    current = torch.arange(64, dtype=torch.int16)
    report = verify_full_pulls(
        full_staging={"w": current},
        sources={"w": _Source((64,), [_shard((0,), (64,), None)])},
        fresh_sources={"w": _Source((64,), [_shard((0,), (64,), None)])},
    )
    assert report["checked"] == 0
    assert report["skipped_no_digest"] == 1


def test_digests_refreshed_is_zero_when_nothing_moved():
    """The weak-evidence signal.

    On the run that resolved Bug 9, exactly 1 of ~18865 sources re-digested between
    consecutive training steps - a GRPO step with zero advantage produces no policy
    gradient. A gate run against weights that did not move proves much less than a
    clean report suggests, so the count is surfaced rather than left implicit.
    """
    current = torch.arange(64, dtype=torch.int16)
    d = tensor_digest(shard_region(current, (64,), (0,), (64,)))
    report = verify_full_pulls(
        full_staging={"w": current},
        sources={"w": _Source((64,), [_shard((0,), (64,), d)])},
        fresh_sources={"w": _Source((64,), [_shard((0,), (64,), d)])},
    )
    assert report["digests_refreshed"] == 0
    assert report["digest_source"] == "fresh"
    assert report["mismatches"] == 0


# --------------------------------------------------- refresh across a reselection
# Why these exist: the session-keyed refresh above was shipped as the Bug 9 fix and
# did not hold. Run v43 (Topology B, 4 EFAs, heartbeat fix in place) still reported
# exactly one bad tensor - model.layers.18.self_attn.o_proj.weight, owner
# trainer-r1 - while the addr-recheck probe on the same step reported
# digest_changed 2 and reselected_to_other_rank 867. So the publisher WAS
# advertising a moved digest and the gate still did not pick it up: the box had been
# reselected to a different replica, the owner key missed, and the refresh quietly
# fell back to the prepare-time expectation. digests_refreshed 0 alongside
# digest_changed 2 is the contradiction that gives this away.
def test_a_box_reselected_to_another_replica_still_refreshes():
    """The v43 regression: the owner key misses, a sibling offer must be used."""
    current = torch.arange(64, dtype=torch.int16)
    current_digest = tensor_digest(shard_region(current, (64,), (0,), (64,)))
    stale_digest = tensor_digest(torch.zeros(64, dtype=torch.int16))

    report = verify_full_pulls(
        full_staging={"w": current},
        sources={
            "w": _Source((64,), [_shard((0,), (64,), stale_digest, session="trainer-r1")])
        },
        # Same box, now offered only by the replica the planner reselected to.
        fresh_sources={
            "w": _Source(
                (64,), [_shard((0,), (64,), current_digest, session="trainer-r9")]
            )
        },
    )
    assert report["mismatches"] == 0, "owner-keyed miss left the stale expectation"
    assert report["digests_refreshed"] == 1
    assert report["digests_refreshed_via_replica"] == 1


def test_the_owner_offer_wins_over_a_sibling_offer():
    """The fallback must not override a digest the actual owner still advertises."""
    current = torch.arange(64, dtype=torch.int16)
    owner_digest = tensor_digest(shard_region(current, (64,), (0,), (64,)))
    sibling_digest = tensor_digest(torch.ones(64, dtype=torch.int16))
    assert owner_digest != sibling_digest

    report = verify_full_pulls(
        full_staging={"w": current},
        sources={
            "w": _Source((64,), [_shard((0,), (64,), owner_digest, session="trainer-r1")])
        },
        fresh_sources={
            "w": _Source(
                (64,),
                [
                    _shard((0,), (64,), sibling_digest, session="trainer-r0"),
                    _shard((0,), (64,), owner_digest, session="trainer-r1"),
                ],
            )
        },
    )
    assert report["mismatches"] == 0
    assert report["digests_refreshed_via_replica"] == 0


def test_the_fallback_is_refused_when_the_replicas_disagree():
    """Adopting one of two disagreeing offers would be guessing.

    Replicas of a box are required to be byte-identical; when they are not, which
    one is 'current' is undefined, and picking either could turn a real transport
    fault into a pass. The divergence is reported elsewhere - here the refresh must
    simply decline, leaving the prepare-time expectation to be judged on its merits.
    """
    current = torch.arange(64, dtype=torch.int16)
    stale_digest = tensor_digest(torch.zeros(64, dtype=torch.int16))
    one = tensor_digest(torch.ones(64, dtype=torch.int16))
    two = tensor_digest(torch.full((64,), 2, dtype=torch.int16))
    assert one != two

    report = verify_full_pulls(
        full_staging={"w": current},
        sources={
            "w": _Source((64,), [_shard((0,), (64,), stale_digest, session="trainer-r1")])
        },
        fresh_sources={
            "w": _Source(
                (64,),
                [
                    _shard((0,), (64,), one, session="trainer-r7"),
                    _shard((0,), (64,), two, session="trainer-r9"),
                ],
            )
        },
    )
    assert report["digests_refreshed"] == 0
    assert report["digests_refreshed_via_replica"] == 0
    assert report["mismatches"] == 1, "declining to refresh must not also skip the check"


def test_a_reselected_box_at_a_different_offset_is_not_borrowed():
    """The fallback is per box, not per source.

    Borrowing any digest published under the same tensor name would compare a shard
    against a different slice of the same tensor and call disagreement corruption.
    """
    current = torch.arange(128, dtype=torch.int16)
    d0 = tensor_digest(shard_region(current, (128,), (0,), (64,)))
    report = verify_full_pulls(
        full_staging={"w": current},
        sources={
            "w": _Source((128,), [_shard((0,), (64,), d0, session="trainer-r1")])
        },
        # Only the OTHER half of the tensor is offered fresh.
        fresh_sources={
            "w": _Source(
                (128,),
                [_shard((64,), (64,), "deadbeef", session="trainer-r9")],
            )
        },
    )
    assert report["digests_refreshed"] == 0
    assert report["mismatches"] == 0, "fell back to prepare, which was correct here"


# ------------------------------------------- source expectations, for one run
# The destination digests these pair with are only comparable across two runs, and
# two runs stop being comparable after one training step. These cover the pairing
# that lets a single run be audited, and they lean on the stale-expectation cases,
# because a frozen expectation here manufactures the audit's strongest finding out
# of ordinary training - which is Bug 9 wearing a different hat.
def test_source_expectation_covers_a_param_whose_shards_all_carry_digests():
    sources = {"src": _Source((4, 2), [_shard([0, 0], [2, 2], digest="a"),
                                       _shard([2, 0], [2, 2], digest="b")])}
    digests, stats = source_expectation_digests(
        dest_sources={"param": ["src"]}, sources=sources
    )
    assert digests["param"] is not None
    assert stats["covered"] == 1
    assert stats["uncovered"] == 0


def test_source_expectation_changes_when_a_shard_digest_changes():
    def build(second):
        return {"src": _Source((4, 2), [_shard([0, 0], [2, 2], digest="a"),
                                        _shard([2, 0], [2, 2], digest=second)])}

    before, _ = source_expectation_digests(
        dest_sources={"param": ["src"]}, sources=build("b")
    )
    after, _ = source_expectation_digests(
        dest_sources={"param": ["src"]}, sources=build("MOVED")
    )
    assert before["param"] != after["param"]


def test_source_expectation_ignores_shard_visit_order():
    """Replica reselection between discoveries reorders the visit; it is not a
    change, and reporting it as one is how the first fix for Bug 9 failed."""
    forward = {"src": _Source((4, 2), [_shard([0, 0], [2, 2], digest="a"),
                                       _shard([2, 0], [2, 2], digest="b")])}
    reversed_ = {"src": _Source((4, 2), [_shard([2, 0], [2, 2], digest="b"),
                                         _shard([0, 0], [2, 2], digest="a")])}
    a, _ = source_expectation_digests(dest_sources={"param": ["src"]}, sources=forward)
    b, _ = source_expectation_digests(dest_sources={"param": ["src"]}, sources=reversed_)
    assert a["param"] == b["param"]


def test_source_expectation_is_none_when_a_shard_has_no_digest():
    """Publishers predating the digest must read as no evidence, never as a pass."""
    sources = {"src": _Source((4, 2), [_shard([0, 0], [2, 2], digest="a"),
                                       _shard([2, 0], [2, 2])])}
    digests, stats = source_expectation_digests(
        dest_sources={"param": ["src"]}, sources=sources
    )
    assert digests["param"] is None
    assert stats["uncovered"] == 1


def test_source_expectation_prefers_the_freshly_discovered_digest():
    """The frozen-expectation fix: the bytes were read from an address that has not
    moved, so what landed is what the publisher holds now."""
    stale = {"src": _Source((4, 2), [_shard([0, 0], [4, 2], digest="old")])}
    fresh = {"src": _Source((4, 2), [_shard([0, 0], [4, 2], digest="new")])}
    digests, stats = source_expectation_digests(
        dest_sources={"param": ["src"]}, sources=stale, fresh_sources=fresh
    )
    only_stale, _ = source_expectation_digests(
        dest_sources={"param": ["src"]}, sources=stale
    )
    assert digests["param"] != only_stale["param"]
    assert stats["shard_claims_from_fresh_table"] == 1


def test_source_expectation_refuses_to_emit_a_stale_expectation():
    """Worse than none: a table that cannot change turns every legitimate weight
    update into 'the destination moved by itself', the audit's strongest finding."""
    sources = {"src": _Source((4, 2), [_shard([0, 0], [4, 2], digest="a")])}
    digests, stats = source_expectation_digests(
        dest_sources={"param": ["src"]},
        sources=sources,
        expectation_is_current=False,
    )
    assert digests == {"param": None}
    assert stats["covered"] == 0
    assert "reason" in stats


def test_source_expectation_marks_a_param_whose_source_is_absent():
    digests, stats = source_expectation_digests(
        dest_sources={"param": ["missing"]}, sources={}
    )
    assert digests["param"] is None
    assert stats["uncovered"] == 1


# --- a frozen reference must not read as a wire fault --------------------------
# On 2026-07-30 every `want` in a run's mismatches was bit-for-bit the initial
# checkpoint digest while `got` tracked training, and VERIFY_STRICT aborted two runs
# on it. A difference against a reference older than the bytes says nothing about
# the wire, and must not be fatal - nor may it be quietly dropped.


def _stale_rig(published_digest="not-the-real-digest"):
    """One full-pulled source whose staged bytes do not match the advertised digest."""
    staging = torch.arange(4096, dtype=torch.int32)
    sources = {
        "w": _Source(
            global_shape=tuple(staging.shape),
            shards=[_shard((0,), tuple(staging.shape), digest=published_digest)],
        )
    }
    return {"w": staging}, sources


def test_a_mismatch_at_step_one_is_still_attributable():
    """Nothing has moved yet at the first refit, so refreshing nothing is the
    correct answer there and the reference really is current."""
    staging, sources = _stale_rig()

    report = verify_full_pulls(
        full_staging=staging, sources=sources, fresh_sources=sources, step=1
    )

    assert report["mismatches"] == 1
    assert report["reference_is_current"] is True
    assert "stale_reference_suspected" not in report


def test_a_mismatch_past_step_one_with_nothing_refreshed_is_unattributable():
    staging, sources = _stale_rig()

    report = verify_full_pulls(
        full_staging=staging, sources=sources, fresh_sources=sources, step=3
    )

    assert report["digests_refreshed"] == 0
    assert report["reference_is_current"] is False
    assert report["stale_reference_suspected"] is True
    assert "UNVERIFIED" in report["unattributable_reason"]


def test_unattributable_is_not_reported_as_clean():
    """The opposite failure: swallowing the mismatch so the run looks verified."""
    staging, sources = _stale_rig()

    report = verify_full_pulls(
        full_staging=staging, sources=sources, fresh_sources=sources, step=3
    )

    assert report["mismatches"] == 1, "the mismatch must stay visible"
    assert report["detail"], "the offending shard must still be named"


def test_a_reference_that_did_refresh_stays_attributable_and_fatal():
    """If the table refreshed something it is tracking the publisher, so a
    remaining mismatch is a real finding and must not be excused."""
    staging = torch.arange(4096, dtype=torch.int32)
    stale = {
        "w": _Source(
            global_shape=tuple(staging.shape),
            shards=[_shard((0,), tuple(staging.shape), digest="old")],
        )
    }
    fresh = {
        "w": _Source(
            global_shape=tuple(staging.shape),
            shards=[_shard((0,), tuple(staging.shape), digest="refreshed-still-wrong")],
        )
    }

    report = verify_full_pulls(
        full_staging={"w": staging}, sources=stale, fresh_sources=fresh, step=3
    )

    assert report["digests_refreshed"] == 1
    assert report["reference_is_current"] is True
    assert report["mismatches"] == 1
    assert "stale_reference_suspected" not in report


def test_step_is_optional_so_existing_callers_keep_their_behaviour():
    staging, sources = _stale_rig()

    report = verify_full_pulls(full_staging=staging, sources=sources)

    assert report["reference_is_current"] is True
    assert report["mismatches"] == 1


def test_a_clean_run_past_step_one_carries_no_staleness_claim():
    """The flag explains a mismatch; it does not editorialise on every report."""
    staging = torch.arange(4096, dtype=torch.int32)
    sources = {
        "w": _Source(
            global_shape=tuple(staging.shape),
            shards=[
                _shard((0,), tuple(staging.shape), digest=tensor_digest(staging))
            ],
        )
    }

    report = verify_full_pulls(
        full_staging={"w": staging}, sources=sources, fresh_sources=sources, step=5
    )

    assert report["mismatches"] == 0
    assert "stale_reference_suspected" not in report


# ------------------------------------------------- the publisher step stamp
#
# The stamp exists because deducing freshness from "did any digest refresh?" is a
# whole-discovery verdict, and publishers propagate independently. The test that earns
# its place here is the partial-propagation one: something refreshed, so the inference
# pronounces the reference current and reports a lagging publisher's shard as a hard
# defect. That is a false abort, it gets likelier with every publisher added, and it is
# the failure the stamp removes.


def _two_publisher_rig():
    """One source, two sessions, and one of them mismatching."""
    full = torch.arange(64, dtype=torch.int16)
    good = tensor_digest(shard_region(full, (64,), (0,), (64,)))
    bad = "0" * len(good)
    return full, good, bad


def test_a_lagging_publisher_is_not_reported_as_a_wire_fault():
    full, _good, bad = _two_publisher_rig()

    report = verify_full_pulls(
        full_staging={"w": full},
        sources={"w": _Source((64,), [_shard((0,), (64,), bad, session="r0")])},
        step=2,
        stale_sessions={"r0"},
    )

    assert report["mismatches"] == 1, "the difference must stay visible"
    assert report["mismatches_from_stale_publishers"] == 1
    assert report["attributable_mismatches"] == 0
    assert report["reference_is_current"] is False
    assert report["stale_reference_suspected"] is True
    assert report["freshness_evidence"] == "publisher_step_stamp"


def test_a_mismatch_from_a_current_publisher_stays_fatal():
    """The stamp must not become a blanket excuse.

    ``r0`` lags, but the mismatching shard belongs to ``r1``, which does not. The abort
    signal is ``attributable_mismatches``: it is what the receiver keys on, and it is
    what must stay non-zero here.
    """
    full, _good, bad = _two_publisher_rig()

    report = verify_full_pulls(
        full_staging={"w": full},
        sources={"w": _Source((64,), [_shard((0,), (64,), bad, session="r1")])},
        step=2,
        stale_sessions={"r0"},
    )

    assert report["mismatches"] == 1
    assert report["mismatches_from_stale_publishers"] == 0
    assert report["attributable_mismatches"] == 1, "must remain fatal for the caller"


def test_stamps_with_nothing_lagging_assert_currency_positively():
    """The reason ``stamps_seen`` is separate from a non-empty stale set.

    Nothing refreshed here, so the inference would call this reference stale and give up
    on verifying the step. The stamps say every publisher advanced, which is a positive
    statement that the table is current - so the mismatch is real and fatal.
    """
    full, _good, bad = _two_publisher_rig()

    report = verify_full_pulls(
        full_staging={"w": full},
        sources={"w": _Source((64,), [_shard((0,), (64,), bad, session="r0")])},
        step=4,
        stale_sessions=frozenset(),
        stamps_seen=True,
    )

    assert report["digests_refreshed"] == 0, "rig must give the inference nothing to go on"
    assert report["freshness_evidence"] == "publisher_step_stamp"
    assert report["reference_is_current"] is True
    assert report["attributable_mismatches"] == 1, (
        "verification the stamps earned must not be discarded"
    )


def test_without_stamps_the_same_rig_gives_up():
    """The contrast, and the cost of having only the inference."""
    full, _good, bad = _two_publisher_rig()

    report = verify_full_pulls(
        full_staging={"w": full},
        sources={"w": _Source((64,), [_shard((0,), (64,), bad, session="r0")])},
        step=4,
    )

    assert report["freshness_evidence"] == "refresh_inference"
    assert report["reference_is_current"] is False
    assert report["attributable_mismatches"] == 0, "unverifiable, so nothing is claimed"


def test_partial_propagation_does_not_condemn_the_lagging_publisher():
    """The case the inference gets wrong.

    Two publishers, one lagging and mismatching, one current and clean. Some digest
    refreshed, so ``refreshed > 0`` and the old inference calls the whole reference
    current - making the lagging publisher's shard look like a real defect. With stamps
    the lagging shard is excused and the report still says nothing is attributable.
    """
    full, good, bad = _two_publisher_rig()
    sources = {
        "w": _Source(
            (64,),
            [
                _shard((0,), (64,), bad, session="lagging"),
                _shard((0,), (64,), good, session="current"),
            ],
        )
    }

    report = verify_full_pulls(
        full_staging={"w": full},
        sources=sources,
        step=4,
        stale_sessions={"lagging"},
    )

    assert report["mismatches"] == 1
    assert report["attributable_mismatches"] == 0, (
        "a lagging publisher under partial propagation must not read as a wire fault"
    )
    assert report["reference_is_current"] is False


def _partial_propagation_rig():
    """The hardware situation, in miniature.

    Two publishers for one shard box. ``current`` has published this step's table, so
    its digest refreshed against prepare. ``lagging`` has not, so its digest is
    unchanged and no longer describes the bytes on the wire. That is the state the
    inference cannot represent: *something* refreshed, so it pronounces the whole
    reference current, and ``lagging``'s stale comparison is then reported as a real
    defect - aborting a healthy run.
    """
    full = torch.arange(64, dtype=torch.int16)
    good = tensor_digest(shard_region(full, (64,), (0,), (64,)))
    stale_digest = "0" * len(good)
    prepare_digest_for_current = "1" * len(good)
    sources = {
        "w": _Source(
            (64,),
            [
                _shard((0,), (64,), stale_digest, session="lagging"),
                _shard((0,), (64,), prepare_digest_for_current, session="current"),
            ],
        )
    }
    fresh = {
        "w": _Source(
            (64,),
            [
                # unchanged since prepare - this publisher has not caught up
                _shard((0,), (64,), stale_digest, session="lagging"),
                # refreshed, and matches the bytes actually delivered
                _shard((0,), (64,), good, session="current"),
            ],
        )
    }
    return full, sources, fresh


def test_the_inference_alone_would_have_aborted_a_healthy_run():
    """Pins the defect being fixed. No stamps: the run dies on a lagging publisher."""
    full, sources, fresh = _partial_propagation_rig()

    report = verify_full_pulls(
        full_staging={"w": full}, sources=sources, fresh_sources=fresh, step=4
    )

    assert report["digests_refreshed"] > 0, "rig must reproduce partial propagation"
    assert report["freshness_evidence"] == "refresh_inference"
    assert report["reference_is_current"] is True, (
        "the inference is fooled by the one publisher that did refresh"
    )
    assert report["attributable_mismatches"] == 1, (
        "and so the lagging publisher's shard is condemned - the false abort"
    )


def test_the_stamp_prevents_that_abort():
    """Same rig, same bytes, stamps supplied. Nothing attributable, nothing fatal."""
    full, sources, fresh = _partial_propagation_rig()

    report = verify_full_pulls(
        full_staging={"w": full},
        sources=sources,
        fresh_sources=fresh,
        step=4,
        stale_sessions={"lagging"},
    )

    assert report["digests_refreshed"] > 0
    assert report["mismatches"] == 1, "the difference is still reported"
    assert report["attributable_mismatches"] == 0
    assert report["reference_is_current"] is False, "so the caller must not abort"
    assert report["freshness_evidence"] == "publisher_step_stamp"


def test_an_unattributable_report_does_not_claim_attributable_mismatches():
    """The two fields must not contradict each other in one record."""
    full, _good, bad = _two_publisher_rig()

    report = verify_full_pulls(
        full_staging={"w": full},
        sources={"w": _Source((64,), [_shard((0,), (64,), bad, session="r0")])},
        step=4,
    )

    assert report["reference_is_current"] is False
    assert report["attributable_mismatches"] == 0


def test_a_clean_run_with_stamps_is_still_clean():
    """A lagging publisher that nonetheless matches raises nothing.

    ``reference_is_current`` is False because ``r0`` is behind, but there is no mismatch
    to excuse, so no staleness claim is attached and nothing is attributable.
    """
    full, good, _bad = _two_publisher_rig()

    report = verify_full_pulls(
        full_staging={"w": full},
        sources={"w": _Source((64,), [_shard((0,), (64,), good, session="r0")])},
        step=3,
        stale_sessions={"r0"},
    )

    assert report["mismatches"] == 0
    assert report["attributable_mismatches"] == 0
    assert "stale_reference_suspected" not in report


def test_stamps_are_reported_so_a_reader_can_audit_the_verdict():
    full, good, _bad = _two_publisher_rig()

    report = verify_full_pulls(
        full_staging={"w": full},
        sources={"w": _Source((64,), [_shard((0,), (64,), good, session="r0")])},
        step=3,
        stale_sessions={"r1", "r0"},
    )

    assert report["stale_publisher_sessions"] == ["r0", "r1"]


def test_the_flagged_shard_names_its_publisher_as_stale():
    full, _good, bad = _two_publisher_rig()

    report = verify_full_pulls(
        full_staging={"w": full},
        sources={"w": _Source((64,), [_shard((0,), (64,), bad, session="r0")])},
        step=2,
        stale_sessions={"r0"},
    )

    assert report["detail"][0]["stale_publisher"] is True
