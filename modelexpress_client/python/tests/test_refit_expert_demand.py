# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Receiver-side expert-demand boundaries for the staged refit planner.

These call the real ``plan_transfer`` entry point with a synthetic grouped-expert
source. Nothing here uses a mock selector: the oracle is the ownership the
receiver declares, and the assertions are about which publishers the returned
plan actually reads from.

The layout under test is one source tensor holding several experts, sharded on
dim 0 across two publishers, which is what an expert-parallel trainer publishes
when it groups experts into a single tensor. A receiver that owns only some of
those experts must not read the publishers that hold the rest.
"""

import pytest
import torch
from modelexpress.refit.reshard.slice_plan import Shard
from modelexpress.refit.reshard.transfer_plan import SourceInfo, plan_transfer
from modelexpress.refit.reshard.types import CaptureResult, RecordedCopy

EXPERTS = 8
ROWS_PER_EXPERT = 64
COLS = 8
BYTES_PER_EXPERT = ROWS_PER_EXPERT * COLS * 4


def _grouped_expert_source() -> SourceInfo:
    """One grouped tensor holding 8 experts, split evenly across two publishers."""
    rows = EXPERTS * ROWS_PER_EXPERT
    return SourceInfo(
        global_shape=(rows, COLS),
        dtype=torch.float32,
        elsize=4,
        shards=[
            Shard((0, 0), (rows // 2, COLS), "pub0", 0x1000, 4),
            Shard((rows // 2, 0), (rows // 2, COLS), "pub1", 0x9000, 4),
        ],
    )


def _expert_copy(*, stride: tuple) -> RecordedCopy:
    """Read expert 0 only, into a destination with the given stride.

    Rows 0..63 are expert 0, and pub0 owns experts 0..3, so a plan that reads
    only owned bytes reads only pub0.
    """
    return RecordedCopy(
        src_name="grouped_experts",
        op_chain=(("narrow", (0, 0, ROWS_PER_EXPERT), ()),),
        param_name="experts",
        dest_offset=0,
        dest_shape=(ROWS_PER_EXPERT, COLS),
        dest_stride=stride,
        dest_dtype=torch.float32,
    )


def test_contiguous_demand_reads_only_the_publisher_that_owns_the_expert():
    """The ordinary case: a contiguous view needs one run and stays inside pub0."""
    plan = plan_transfer(
        CaptureResult(copies=[_expert_copy(stride=(COLS, 1))]),
        {"grouped_experts": _grouped_expert_source()},
    )

    assert plan.sessions() == {"pub0"}
    assert plan.full_pulls == []
    assert plan.bytes_planned() == BYTES_PER_EXPERT
    assert plan.extra_wire_bytes() == 0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Bounded fallback widens a receiver-owned demand to a whole-source pull, "
        "which reads publishers that hold experts this receiver does not own. "
        "The mechanism is deliberate and byte-accounted, but nothing keeps it "
        "inside the receiver's owned regions."
    ),
)
def test_bounded_fallback_stays_inside_the_receiver_owned_experts():
    """A descriptor-heavy copy of one expert must not pull another expert.

    A transposed destination needs one descriptor per column, which exceeds the
    production default bound and routes the copy through the full-source pull.
    The receiver owns experts 0..3 (pub0) only, so reading pub1 means its plan
    transferred weights that its own loader refused.
    """
    plan = plan_transfer(
        CaptureResult(copies=[_expert_copy(stride=(1, ROWS_PER_EXPERT))]),
        {"grouped_experts": _grouped_expert_source()},
    )

    assert plan.full_pulls, "expected the descriptor bound to engage"
    assert plan.sessions() == {"pub0"}, (
        "the bounded fallback read a publisher holding unowned experts"
    )


def test_bounded_fallback_reports_the_bytes_it_widened_to():
    """Pin the widening itself, so the metric cannot silently under-report.

    This is the behaviour the xfail above asks to change. Until that changes,
    the plan must at least account for every byte it widened to, otherwise a
    receiver cannot see the cost it is paying.
    """
    plan = plan_transfer(
        CaptureResult(copies=[_expert_copy(stride=(1, ROWS_PER_EXPERT))]),
        {"grouped_experts": _grouped_expert_source()},
    )

    assert plan.full_pulls, "expected the descriptor bound to engage"
    assert plan.exact_bytes == BYTES_PER_EXPERT
    # The full pull reconstructs the whole source: every expert, both publishers.
    whole_source = EXPERTS * ROWS_PER_EXPERT * COLS * 4
    assert plan.bytes_planned() == whole_source
    assert plan.extra_wire_bytes() == whole_source - BYTES_PER_EXPERT
    assert plan.sessions() == {"pub0", "pub1"}
    assert plan.descriptor_savings() > 0
