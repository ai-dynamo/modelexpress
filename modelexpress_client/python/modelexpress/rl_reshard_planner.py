# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python reshard planner.

Given a catalog of published :class:`SliceOwnership` entries and a list of
receiver-side :class:`SliceRequest` entries, computes the
:class:`CoveragePlan` of one-sided RDMA reads the receiver should issue.

This module is deliberately byte-free: it doesn't touch torch tensors,
NIXL agents, or CUDA. It's a stateless pure function that operates on
descriptor metadata only. The receiver-side wrapper (e.g.
:class:`MxRefitReceiver` or the verl checkpoint engine) drives the
plan through NIXL.

See ``docs/RL/VERL_MX_OVERVIEW.md`` §8 ("How the planner works") for
the algorithm and the broader rank-to-rank design context.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from .rl_slice_descriptors import (
    CoveragePlan,
    QuantizationMetadataError,
    SegmentPlan,
    SliceOwnership,
    SliceRequest,
)

# ---------------------------------------------------------------------------
# Stride helpers
# ---------------------------------------------------------------------------


def _dtype_bytes(dtype: str) -> int:
    """Look up bytes-per-element for a torch dtype string.

    We avoid importing torch here so the planner stays pure-Python.
    The catalog is the source of truth for dtype; this map covers the
    dtypes used by every framework we integrate with today.
    """
    table = {
        "torch.bfloat16": 2, "bfloat16": 2, "bf16": 2,
        "torch.float16": 2, "float16": 2, "fp16": 2, "half": 2,
        "torch.float32": 4, "float32": 4, "fp32": 4, "float": 4,
        "torch.float8_e4m3fn": 1, "float8_e4m3fn": 1, "fp8_e4m3": 1, "fp8": 1,
        "torch.float8_e5m2": 1, "float8_e5m2": 1, "fp8_e5m2": 1,
        "torch.uint8": 1, "uint8": 1,
        "torch.int8": 1, "int8": 1,
    }
    if dtype not in table:
        raise ValueError(f"_dtype_bytes: unknown dtype {dtype!r}")
    return table[dtype]


def _row_stride_bytes(global_shape: tuple[int, ...], shard_axis: int, dtype: str) -> int:
    """Bytes per "row" along ``shard_axis`` — i.e. bytes that move when
    you advance one step along the shard axis.

    For a tensor of shape ``(rows, cols)`` sharded on axis 0, the row
    stride is ``cols * dtype_bytes``. For shape ``(experts, in, out)``
    sharded on axis 0, the row stride is ``in * out * dtype_bytes``.
    """
    elem = _dtype_bytes(dtype)
    inner = 1
    for ax, dim in enumerate(global_shape):
        if ax > shard_axis:
            inner *= dim
    return inner * elem


# ---------------------------------------------------------------------------
# Eligibility filter
# ---------------------------------------------------------------------------


def _source_matches_request(
    own: SliceOwnership, req: SliceRequest
) -> tuple[bool, str | None]:
    """Eligibility check before any range math.

    Returns ``(matches, rejection_reason)``. When ``matches`` is False,
    ``rejection_reason`` is a short human-readable string suitable for
    the ``missing`` list of :class:`CoveragePlan`.
    """
    # Name match — required.
    if own.tensor_name != req.tensor_name:
        return False, f"tensor_name mismatch: {own.tensor_name!r} vs {req.tensor_name!r}"

    # Dtype match — required. The planner does not silently cast.
    if own.dtype != req.dtype:
        return False, f"dtype mismatch: source {own.dtype!r} vs request {req.dtype!r}"

    # Compile-target filter — Phase 3b safety net.
    if req.compile_target_filter is not None:
        if own.compile_target not in req.compile_target_filter:
            return False, (
                f"compile_target {own.compile_target!r} not in filter "
                f"{sorted(req.compile_target_filter)!r}"
            )

    # Required compile metadata — subset-equality on keys the request demands.
    for key, want in req.required_compile_metadata.items():
        if own.compile_metadata.get(key) != want:
            return False, (
                f"compile_metadata[{key!r}]: source has "
                f"{own.compile_metadata.get(key)!r}, request wants {want!r}"
            )

    # Quantization scope — the planner refuses zero-copy on global-required
    # tensors because the receiver can't safely use just one source's slice.
    # Caller can catch QuantizationMetadataError and fall back to a full-copy
    # install path (e.g. the trainer republishes the global tensor as a
    # REPLICATE source, or the receiver runs a separate metadata fetch).
    if own.quantization_scope == "global-required":
        raise QuantizationMetadataError(
            f"tensor {own.tensor_name!r}: quantization_scope='global-required' "
            "cannot be zero-copy resharded; caller must use full-copy install path"
        )

    return True, None


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------


def plan_coverage(
    sources: Iterable[SliceOwnership],
    requests: Iterable[SliceRequest],
) -> CoveragePlan:
    """Compute the segment-plan that covers every request from the sources.

    Algorithm (per-request, since requests are independent):

    1. Filter sources by name + dtype + compile_target + compile_metadata.
    2. For each eligible source, compute the intersection of
       ``source.local_shard_range`` (or the whole tensor for REPLICATE)
       with ``request.global_range``.
    3. Greedily walk the request range from low to high, emitting one
       ``SegmentPlan`` per contiguous covered region. Prefers same-rank
       sources when multiple cover the same region (to keep multi-NIC
       routing happy on GB200) but doesn't enforce it.
    4. Any uncovered tail goes into ``missing``.

    The planner does NOT pick "the best" multi-source layout — when more
    than one source can cover the same region, it picks the first eligible
    one. Callers that want load-balancing or topology-aware selection
    should pre-sort the sources by their preferred policy (e.g.
    same-rank-first, then freshest, then least-loaded) before calling.

    Args:
        sources: iterable of :class:`SliceOwnership` from the catalog
            (i.e. ``MxClient.list_sources()`` + ``get_metadata()`` output).
        requests: iterable of :class:`SliceRequest` the receiver wants.

    Returns:
        A :class:`CoveragePlan` with one ``SegmentPlan`` per contiguous
        covered region and ``missing`` entries for any uncovered request
        ranges.

    Raises:
        QuantizationMetadataError: if any matched source has
            ``quantization_scope='global-required'``. Caller should catch
            and fall back to a non-zero-copy install path for that tensor.
    """
    sources = list(sources)
    requests = list(requests)

    # Group sources by tensor name for O(1) lookup per request.
    by_name: dict[str, list[SliceOwnership]] = defaultdict(list)
    for s in sources:
        by_name[s.tensor_name].append(s)

    segments: list[SegmentPlan] = []
    missing: list[tuple[str, tuple[int, int], str]] = []

    for req in requests:
        candidates = by_name.get(req.tensor_name, [])
        if not candidates:
            missing.append((req.tensor_name, req.global_range, "no source published this tensor"))
            continue

        # Filter to eligible sources + collect rejection reasons for diagnostics.
        eligible: list[SliceOwnership] = []
        last_reason: str | None = None
        for own in candidates:
            ok, reason = _source_matches_request(own, req)
            if ok:
                eligible.append(own)
            elif reason is not None:
                last_reason = reason

        if not eligible:
            missing.append(
                (
                    req.tensor_name,
                    req.global_range,
                    last_reason or "no eligible source after filtering",
                )
            )
            continue

        # Order: prefer same-rank, then freshest (proxied here by smaller
        # rank — the real freshness tiebreaker lives in MxV2RefitReceiver's
        # dedup_freshest_per_rank logic and runs upstream of this planner).
        eligible.sort(key=lambda s: (s.worker_rank != req.receiver_rank, s.worker_rank))

        # Greedy contiguous walk along the request range.
        req_lo, req_hi = req.global_range
        cursor = req_lo
        covered_by_request: list[SegmentPlan] = []
        while cursor < req_hi:
            picked: SegmentPlan | None = None
            for own in eligible:
                # For REPLICATE sources we can take the whole remaining range
                # from this one source.
                if own.placement_kind == "REPLICATE":
                    seg = _make_segment(own, req, (cursor, req_hi), req_lo)
                    picked = seg
                    break

                # For SHARD sources, find one whose range starts at-or-before
                # `cursor` and overlaps the remaining request.
                assert own.local_shard_range is not None  # SHARD invariant
                own_lo, own_hi = own.local_shard_range
                if own_lo <= cursor < own_hi:
                    target_hi = min(own_hi, req_hi)
                    seg = _make_segment(own, req, (cursor, target_hi), req_lo)
                    picked = seg
                    break

            if picked is None:
                # No source covers `cursor` — find the next source-start to
                # report the gap accurately rather than dumping everything.
                next_start = req_hi
                for own in eligible:
                    if own.placement_kind == "SHARD" and own.local_shard_range:
                        own_lo = own.local_shard_range[0]
                        if cursor < own_lo < next_start:
                            next_start = own_lo
                missing.append(
                    (
                        req.tensor_name,
                        (cursor, next_start),
                        "no source covers this range",
                    )
                )
                cursor = next_start
                continue

            covered_by_request.append(picked)
            cursor = picked.target_range[1] + req_lo  # target_range is request-local

        segments.extend(covered_by_request)

    return CoveragePlan(segments=tuple(segments), missing=tuple(missing))


# ---------------------------------------------------------------------------
# Segment construction
# ---------------------------------------------------------------------------


def _make_segment(
    own: SliceOwnership,
    req: SliceRequest,
    abs_range: tuple[int, int],
    req_lo: int,
) -> SegmentPlan:
    """Build one :class:`SegmentPlan`.

    Args:
        own: the source covering this segment.
        req: the request being partially satisfied.
        abs_range: (lo, hi) in *global / tensor-absolute* coordinates of
            what this segment covers.
        req_lo: the request's global start, used to convert ``abs_range``
            into request-local ``target_range``.

    The byte count is `(hi - lo) * row_stride_bytes(...)`. For ``REPLICATE``
    sources we use the request's shard_axis (the source's whole tensor is
    available so any axis the request asked for is fine).
    """
    abs_lo, abs_hi = abs_range
    n_rows = abs_hi - abs_lo

    # Compute byte count using the request's axis (REPLICATE source has
    # no shard_axis to inherit from).
    axis_for_stride = (
        own.shard_axis if own.placement_kind == "SHARD" else req.shard_axis
    )
    if axis_for_stride is None:
        # Whole-tensor receive of a REPLICATE source: byte count is the
        # entire ownership's footprint divided by global_shape[0] * n_rows.
        # Simpler: just use the source's full byte_size if it covers the
        # whole thing.
        byte_count = (
            own.byte_size
            if n_rows == own.global_shape[0] and abs_lo == 0
            else own.byte_size * n_rows // max(own.global_shape[0], 1)
        )
    else:
        row_stride = _row_stride_bytes(own.global_shape, axis_for_stride, own.dtype)
        byte_count = n_rows * row_stride

    # Source-local coordinates: subtract the source's local-shard start.
    src_offset = (
        abs_lo - own.local_shard_range[0]
        if own.placement_kind == "SHARD" and own.local_shard_range
        else abs_lo
    )
    source_range = (src_offset, src_offset + n_rows)

    # Request-local coordinates: subtract the request's global start.
    target_range = (abs_lo - req_lo, abs_hi - req_lo)

    return SegmentPlan(
        source=own,
        request=req,
        source_range=source_range,
        target_range=target_range,
        byte_count=byte_count,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def summarize_plan(plan: CoveragePlan) -> dict[str, object]:
    """Return a small dict summarizing the plan, useful for logging /
    dashboards / pytest assertions.

    Captures total bytes, segment count, unique source ranks, source/target
    balance (how many segments per source), and the missing list.
    """
    by_source: dict[int, int] = defaultdict(int)
    total_bytes = 0
    for s in plan.segments:
        by_source[s.source.worker_rank] += s.byte_count
        total_bytes += s.byte_count
    return {
        "complete": plan.complete,
        "segment_count": len(plan.segments),
        "total_bytes": total_bytes,
        "source_ranks_used": sorted(by_source.keys()),
        "bytes_per_source": dict(by_source),
        "missing_count": len(plan.missing),
        "missing": [
            {"tensor": name, "range": list(rng), "reason": reason}
            for name, rng, reason in plan.missing
        ],
    }


def collect_byte_savings_vs_allgather(
    plan: CoveragePlan, sources: Sequence[SliceOwnership]
) -> dict[str, int]:
    """Compare bytes-on-the-wire for this plan vs the "naive allgather"
    baseline (gather full model on rank 0, broadcast to every receiver).

    The allgather baseline byte count for a single receiver is the sum of
    every published source's ``byte_size`` (the trainer would gather all
    shards before broadcasting). Multiply by N receivers for the
    full-cluster comparison.

    Returns a dict with both numbers and the savings ratio.
    """
    allgather_per_receiver = sum(s.byte_size for s in sources)
    actual = sum(s.byte_count for s in plan.segments)
    return {
        "allgather_per_receiver_bytes": allgather_per_receiver,
        "rank_to_rank_actual_bytes": actual,
        "savings_factor": (
            allgather_per_receiver / actual if actual > 0 else float("inf")
        ),
    }
