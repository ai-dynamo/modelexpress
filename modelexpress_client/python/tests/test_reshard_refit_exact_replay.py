# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the in-process exact-segment replay gate.

This gate exists because the two-arm destination-digest comparison turned out to be
usable only on step 1: it needs both runs to hold identical source weights, and a
same-arm control measured the run-to-run noise floor at 10 differing params in 4,350
against a cross-arm signal of 1. So the comparison moved inside a single refit, and
what it differences is no longer two runs but two *implementations* over one set of
received bytes - ``plan_pull``'s offset/stride arithmetic against the staged
re-slice.

The tests that earn their place are therefore the ones proving the gate (a) catches
a wrong destination offset, which is the failure that had no gate at all, and
(b) refuses to report a pass when it could not actually compare anything - the trap
every gate in this module has fallen into at least once.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modelexpress.refit.reshard.slice_plan import Shard  # noqa: E402
from modelexpress.refit.reshard.transfer_plan import (  # noqa: E402
    SourceInfo,
    plan_transfer,
)
from modelexpress.refit.reshard.types import CaptureResult, RecordedCopy  # noqa: E402
from modelexpress.refit.reshard.verify import (  # noqa: E402
    compare_exact_replay,
    digest_destination,
    exact_replay_digests,
)

EL = 4  # float32


class _Rig:
    """A source split across two dim-0 shards, and a destination that spans both.

    Spanning the shard boundary is the point: a destination slice fed by one shard
    exercises none of the arithmetic that matters. Rows 2 to 5 of an 8-row source
    sharded 0-3 / 4-7 take two rows from each side, so a segment offset that is
    wrong by a shard changes the bytes without changing their count.
    """

    def __init__(self, *, dest_start=2, dest_rows=4):
        self.full = torch.arange(32, dtype=torch.float32).reshape(8, 4).contiguous()
        # Real, separately allocated shards so their addresses are distinct, which
        # is what the replay's address-range lookup resolves against. Held on self
        # so they are not freed and their data_ptr()s reused.
        self.shard_tensors = [
            self.full[0:4].contiguous(),
            self.full[4:8].contiguous(),
        ]
        shards = [
            Shard(
                shard_offset=(0, 0),
                shape=(4, 4),
                session="trainer-r0",
                addr=self.shard_tensors[0].data_ptr(),
                elsize=EL,
            ),
            Shard(
                shard_offset=(4, 0),
                shape=(4, 4),
                session="trainer-r1",
                addr=self.shard_tensors[1].data_ptr(),
                elsize=EL,
            ),
        ]
        self.sources = {
            "src": SourceInfo(
                global_shape=(8, 4), dtype=torch.float32, elsize=EL, shards=shards
            )
        }
        copy = RecordedCopy(
            src_name="src",
            op_chain=[("narrow", (0, dest_start, dest_rows), {})],
            param_name="p",
            dest_offset=0,
            dest_shape=(dest_rows, 4),
            dest_stride=(4, 1),
            dest_dtype=torch.float32,
        )
        self.plan = plan_transfer(
            CaptureResult(copies=[copy]), self.sources, force_full_pull=True
        )
        # What the full-pull path produces: the whole source staged contiguously,
        # then narrowed locally.
        self.full_staging = {"src": self.full.clone()}
        self.recv_buffers = {
            "p": self.full[dest_start : dest_start + dest_rows].contiguous().clone()
        }

    def run(self):
        replayed, stats = exact_replay_digests(
            plan=self.plan,
            sources=self.sources,
            full_staging=self.full_staging,
            recv_buffers=self.recv_buffers,
        )
        report = compare_exact_replay(
            replayed=replayed, received=digest_destination(self.recv_buffers)
        )
        return report, stats


def test_the_rig_actually_forces_a_full_pull():
    """Guards the premise. If the source were not staged there would be nothing to
    replay from, and every assertion below would pass vacuously."""
    rig = _Rig()
    assert [fp.src_name for fp in rig.plan.full_pulls] == ["src"]
    assert rig.plan.forced_full_pull is True
    assert rig.plan.dest_sources == {"p": ["src"]}


def test_agrees_when_both_implementations_are_right():
    report, stats = _Rig().run()
    assert report["checked"] == 1
    assert report["mismatches"] == 0
    assert stats["segments_outside_any_staged_shard"] == 0
    assert stats["params_with_unstaged_sources"] == 0


def test_spans_the_shard_boundary():
    """Two rows from each shard, so the mapping from a segment's absolute source
    address back into staging is exercised on both."""
    rig = _Rig()
    staged = {
        (segment.session, segment.dst_byte)
        for full_pull in rig.plan.full_pulls
        for segment in full_pull.segments
    }
    assert staged == {("trainer-r0", 0), ("trainer-r1", 64)}
    assert rig.run()[0]["mismatches"] == 0


def test_catches_a_destination_that_holds_the_wrong_slice():
    """The failure with no gate before this: plausible bytes, right count, wrong
    place. Here the installed buffer holds a slice shifted by one row, which no
    byte-count or coverage check can see."""
    rig = _Rig()
    rig.recv_buffers["p"] = rig.full[3:7].contiguous().clone()
    report, _stats = rig.run()
    assert report["checked"] == 1
    assert report["mismatches"] == 1
    assert report["detail"][0]["param"] == "p"


def test_catches_a_permuted_destination():
    """Order-independent checks miss this; the digest is position-sensitive."""
    rig = _Rig()
    rig.recv_buffers["p"] = rig.recv_buffers["p"].flip(0).contiguous()
    assert rig.run()[0]["mismatches"] == 1


def test_catches_a_single_corrupted_value():
    rig = _Rig()
    rig.recv_buffers["p"][1, 2] += 1.0
    assert rig.run()[0]["mismatches"] == 1


def test_reports_no_evidence_when_the_source_was_not_staged():
    """The exact path leaves nothing in staging to replay from, so the honest
    result is checked == 0 rather than a pass. A caller treating zero mismatches
    as success without reading `checked` is the mistake this asserts against."""
    rig = _Rig()
    rig.full_staging = {}
    rig.plan.full_pulls = []
    replayed, stats = exact_replay_digests(
        plan=rig.plan,
        sources=rig.sources,
        full_staging=rig.full_staging,
        recv_buffers=rig.recv_buffers,
    )
    report = compare_exact_replay(
        replayed=replayed, received=digest_destination(rig.recv_buffers)
    )
    assert report["checked"] == 0
    assert report["mismatches"] == 0
    assert stats["params"] == 0


def test_omits_a_param_whose_sources_are_only_partly_staged():
    """A scratch buffer missing one source's contribution would differ everywhere,
    reporting a mismatch that says nothing about slicing."""
    rig = _Rig()
    rig.plan.dest_sources = {"p": ["src", "not_staged"]}
    replayed, stats = exact_replay_digests(
        plan=rig.plan,
        sources=rig.sources,
        full_staging=rig.full_staging,
        recv_buffers=rig.recv_buffers,
    )
    assert replayed == {}
    assert stats["params_with_unstaged_sources"] == 1


def test_reports_segments_that_fall_outside_every_staged_shard():
    """If the exact plan would read an address the full-pull plan never covered,
    the two are not describing the same bytes and the comparison is void. Counted
    and excluded rather than silently mapped to the wrong offset."""
    rig = _Rig()
    for full_pull in rig.plan.full_pulls:
        for segment in full_pull.segments:
            segment.src_addr += 1 << 20
    replayed, stats = exact_replay_digests(
        plan=rig.plan,
        sources=rig.sources,
        full_staging=rig.full_staging,
        recv_buffers=rig.recv_buffers,
    )
    assert replayed == {}
    assert stats["segments_outside_any_staged_shard"] >= 1


def test_whole_tensor_destination_is_covered_too():
    """Not just interior slices: a destination that is the entire source has a
    single segment per shard and must still agree."""
    rig = _Rig(dest_start=0, dest_rows=8)
    report, stats = rig.run()
    assert report["checked"] == 1
    assert report["mismatches"] == 0
    assert stats["uncovered_params"] == 0


def test_catches_a_planner_defect_not_just_a_corrupted_destination(monkeypatch):
    """The test that justifies the gate.

    Every mismatch case above corrupts the installed buffer, which is not the bug
    class this exists for. The real failure is ``plan_pull`` computing a wrong
    destination offset while the transfer executes faithfully - plausible bytes,
    right count, wrong place, and every other gate green. Here plan_pull is made to
    shift one destination offset by a row, standing in for that defect, and the
    replay must report it as a mismatch rather than excluding it.
    """
    from modelexpress.refit.reshard import slice_plan

    real_plan_pull = slice_plan.plan_pull

    def shifted(copy, global_shape, src_dtype, elsize, shards):
        segments = real_plan_pull(copy, global_shape, src_dtype, elsize, shards)
        row_bytes = 4 * EL
        for segment in segments[:1]:
            segment.dst_byte += row_bytes
        return segments

    monkeypatch.setattr(slice_plan, "plan_pull", shifted)

    rig = _Rig()
    report, stats = rig.run()
    assert report["checked"] == 1
    assert report["mismatches"] == 1
    # Excluded-for-cause counters must stay clean: this is a genuine disagreement,
    # not an unmappable segment, and conflating the two would hide the defect.
    assert stats["segments_outside_any_staged_shard"] == 0
    assert stats["copies_the_exact_path_could_not_plan"] == 0
