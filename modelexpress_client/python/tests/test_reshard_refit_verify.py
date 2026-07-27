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
