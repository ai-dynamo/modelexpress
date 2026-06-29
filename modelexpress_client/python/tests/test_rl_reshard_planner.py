# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`modelexpress.rl_reshard_planner`.

Exercises the rank-to-rank no-allgather contract:

- Trainer rank N (FSDP shard) publishes one ``SliceOwnership``.
- Inference rank M (TP shard) publishes a ``SliceRequest`` for its slice.
- Planner emits the right segments and never asks for bytes the source
  doesn't have.
"""

from __future__ import annotations

import pytest

from modelexpress.rl_reshard_planner import (
    collect_byte_savings_vs_allgather,
    plan_coverage,
    summarize_plan,
)
from modelexpress.rl_slice_descriptors import (
    QuantizationMetadataError,
    SliceOwnership,
    SliceRequest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _shard(
    *,
    name: str = "model.layers.0.self_attn.q_proj.weight",
    global_shape: tuple[int, ...] = (4096, 4096),
    rank: int = 0,
    lo: int,
    hi: int,
    dtype: str = "torch.bfloat16",
    compile_target: str = "bf16_cast",
    compile_metadata: dict | None = None,
    quant: str = "absent",
) -> SliceOwnership:
    """Build a SHARD-axis-0 SliceOwnership with byte_size derived from shape."""
    elem = {"torch.bfloat16": 2, "torch.float32": 4, "torch.float8_e4m3fn": 1}[dtype]
    byte_size = (hi - lo) * global_shape[1] * elem
    return SliceOwnership(
        model_name="m", tensor_name=name, global_shape=global_shape, dtype=dtype,
        placement_kind="SHARD", shard_axis=0, local_shard_range=(lo, hi),
        worker_rank=rank, nixl_addr=0x1000 + rank * 0x10000, byte_size=byte_size,
        compile_target=compile_target, compile_metadata=compile_metadata or {},
        quantization_scope=quant,
    )


def _req(
    *,
    name: str = "model.layers.0.self_attn.q_proj.weight",
    lo: int,
    hi: int,
    rank: int = 0,
    dtype: str = "torch.bfloat16",
    compile_target_filter=None,
    required_compile_metadata=None,
) -> SliceRequest:
    return SliceRequest(
        tensor_name=name, global_range=(lo, hi), shard_axis=0, dtype=dtype,
        receiver_rank=rank, target_addr=0x9000 + rank * 0x1000,
        compile_target_filter=compile_target_filter,
        required_compile_metadata=required_compile_metadata or {},
    )


# ---------------------------------------------------------------------------
# Same-shape happy path
# ---------------------------------------------------------------------------


def test_one_source_one_request_exact_match():
    """Trainer FSDP-2, Inference TP-2, same rank — should be one segment, same-rank routed."""
    sources = [_shard(rank=0, lo=0, hi=2048), _shard(rank=1, lo=2048, hi=4096)]
    requests = [_req(rank=0, lo=0, hi=2048)]
    plan = plan_coverage(sources, requests)
    assert plan.complete
    assert len(plan.segments) == 1
    seg = plan.segments[0]
    assert seg.source.worker_rank == 0  # preferred same-rank
    assert seg.source_range == (0, 2048)
    assert seg.target_range == (0, 2048)
    assert seg.byte_count == 2048 * 4096 * 2  # bf16


def test_same_rank_preferred_over_other_ranks():
    """Two sources cover the request; the same-rank one wins."""
    sources = [
        _shard(rank=0, lo=0, hi=4096),   # REPLICATE-like via covers
        _shard(rank=1, lo=0, hi=4096),
    ]
    # Make both SHARDs covering the full range so the same-rank tiebreaker matters.
    requests = [_req(rank=1, lo=0, hi=4096)]
    plan = plan_coverage(sources, requests)
    assert plan.complete
    assert plan.segments[0].source.worker_rank == 1


# ---------------------------------------------------------------------------
# Cross-parallelism resharding (the core scenario)
# ---------------------------------------------------------------------------


def test_fsdp4_to_tp2_resharding():
    """Trainer FSDP=4, inference TP=2. Each TP rank needs 2 trainer shards."""
    sources = [
        _shard(rank=0, lo=0, hi=1024),
        _shard(rank=1, lo=1024, hi=2048),
        _shard(rank=2, lo=2048, hi=3072),
        _shard(rank=3, lo=3072, hi=4096),
    ]
    # TP rank 0 wants rows [0, 2048) -> two segments
    req0 = _req(rank=0, lo=0, hi=2048)
    # TP rank 1 wants rows [2048, 4096) -> two segments
    req1 = _req(rank=1, lo=2048, hi=4096)
    plan = plan_coverage(sources, [req0, req1])
    assert plan.complete
    assert len(plan.segments) == 4

    # Group segments by request
    by_req = {}
    for s in plan.segments:
        by_req.setdefault(s.request.receiver_rank, []).append(s)
    assert len(by_req[0]) == 2
    assert len(by_req[1]) == 2
    # Verify request 0 is covered contiguously
    ranges0 = sorted(s.target_range for s in by_req[0])
    assert ranges0[0] == (0, 1024)
    assert ranges0[1] == (1024, 2048)


def test_fsdp2_to_tp4_one_source_two_requests():
    """Trainer FSDP=2, inference TP=4. One trainer shard covers two TP shards."""
    sources = [
        _shard(rank=0, lo=0, hi=2048),
        _shard(rank=1, lo=2048, hi=4096),
    ]
    # 4 receivers, each wants 1024 rows
    requests = [
        _req(rank=0, lo=0, hi=1024),
        _req(rank=1, lo=1024, hi=2048),
        _req(rank=2, lo=2048, hi=3072),
        _req(rank=3, lo=3072, hi=4096),
    ]
    plan = plan_coverage(sources, requests)
    assert plan.complete
    assert len(plan.segments) == 4
    # First two receivers should read from source rank 0; last two from rank 1.
    seg_by_recv = {s.request.receiver_rank: s for s in plan.segments}
    assert seg_by_recv[0].source.worker_rank == 0
    assert seg_by_recv[1].source.worker_rank == 0
    assert seg_by_recv[2].source.worker_rank == 1
    assert seg_by_recv[3].source.worker_rank == 1


# ---------------------------------------------------------------------------
# Missing coverage
# ---------------------------------------------------------------------------


def test_unknown_tensor_goes_to_missing():
    plan = plan_coverage(sources=[], requests=[_req(lo=0, hi=100)])
    assert not plan.complete
    assert plan.missing[0][2] == "no source published this tensor"


def test_dtype_mismatch_rejects_source():
    sources = [_shard(rank=0, lo=0, hi=4096, dtype="torch.bfloat16")]
    req = _req(lo=0, hi=4096, dtype="torch.float32")
    plan = plan_coverage(sources, [req])
    assert not plan.complete
    assert "dtype mismatch" in plan.missing[0][2]


def test_compile_target_filter_excludes_source():
    sources = [_shard(rank=0, lo=0, hi=4096, compile_target="bf16_cast")]
    req = _req(lo=0, hi=4096, compile_target_filter=frozenset({"cutlass_fp8"}))
    plan = plan_coverage(sources, [req])
    assert not plan.complete
    assert "compile_target" in plan.missing[0][2]


def test_compile_metadata_subset_check():
    """Source has block_size=64; request requires 128 -> rejected."""
    sources = [_shard(rank=0, lo=0, hi=4096, compile_metadata={"block_size": 64})]
    req = _req(lo=0, hi=4096, required_compile_metadata={"block_size": 128})
    plan = plan_coverage(sources, [req])
    assert not plan.complete
    assert "compile_metadata" in plan.missing[0][2]


def test_partial_coverage_reports_specific_gap():
    """Only rank 0 published [0,1024); request for [0,2048) should report a gap."""
    sources = [_shard(rank=0, lo=0, hi=1024)]
    req = _req(lo=0, hi=2048)
    plan = plan_coverage(sources, [req])
    # One segment covers [0,1024), gap is [1024,2048).
    assert len(plan.segments) == 1
    assert plan.segments[0].target_range == (0, 1024)
    assert plan.missing
    # The reported gap should at least overlap [1024, 2048)
    name, rng, _ = plan.missing[0]
    assert name == req.tensor_name
    assert rng[0] == 1024 and rng[1] == 2048


# ---------------------------------------------------------------------------
# REPLICATE handling
# ---------------------------------------------------------------------------


def test_replicate_source_covers_full_request():
    own = SliceOwnership(
        model_name="m", tensor_name="ln.weight",
        global_shape=(4096,), dtype="torch.bfloat16",
        placement_kind="REPLICATE",
        worker_rank=0, nixl_addr=0x1000,
        byte_size=4096 * 2,
    )
    req = SliceRequest(
        tensor_name="ln.weight", global_range=(0, 4096), shard_axis=None,
        dtype="torch.bfloat16", receiver_rank=0,
    )
    plan = plan_coverage([own], [req])
    assert plan.complete
    assert len(plan.segments) == 1


# ---------------------------------------------------------------------------
# Quantization scope
# ---------------------------------------------------------------------------


def test_global_required_quant_raises():
    """global-required quant scope should raise a clean error the caller can
    catch + fall back from."""
    sources = [_shard(rank=0, lo=0, hi=4096, quant="global-required")]
    req = _req(lo=0, hi=4096)
    with pytest.raises(QuantizationMetadataError, match="global-required"):
        plan_coverage(sources, [req])


def test_local_quant_scope_is_planable():
    sources = [_shard(rank=0, lo=0, hi=4096, quant="local")]
    plan = plan_coverage(sources, [_req(lo=0, hi=4096)])
    assert plan.complete


# ---------------------------------------------------------------------------
# Summaries + byte savings
# ---------------------------------------------------------------------------


def test_summarize_plan_reports_bytes_per_source():
    sources = [
        _shard(rank=0, lo=0, hi=2048),
        _shard(rank=1, lo=2048, hi=4096),
    ]
    requests = [_req(lo=0, hi=2048, rank=0), _req(lo=2048, hi=4096, rank=1)]
    plan = plan_coverage(sources, requests)
    summary = summarize_plan(plan)
    assert summary["complete"] is True
    assert summary["segment_count"] == 2
    assert summary["source_ranks_used"] == [0, 1]
    assert summary["bytes_per_source"][0] > 0
    assert summary["bytes_per_source"][1] > 0


def test_byte_savings_fsdp4_to_tp4_is_4x():
    """4-way FSDP, 4-way TP: rank-local pulls 1/4 of allgather bytes."""
    sources = [
        _shard(rank=0, lo=0, hi=1024),
        _shard(rank=1, lo=1024, hi=2048),
        _shard(rank=2, lo=2048, hi=3072),
        _shard(rank=3, lo=3072, hi=4096),
    ]
    # Single receiver pulling its 1/4 slice
    requests = [_req(rank=0, lo=0, hi=1024)]
    plan = plan_coverage(sources, requests)
    savings = collect_byte_savings_vs_allgather(plan, sources)
    assert savings["savings_factor"] == pytest.approx(4.0)
