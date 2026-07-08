# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Transfer plan + pull execution.

Ties the pieces together: take a ``CaptureResult`` and the published source
shards, run ``plan_pull`` per captured copy, and collect the byte segments into a
``TransferPlan``. Any source whose slice can't be expressed as a box
(``UnsupportedReshard``), or that has no published shards, or that capture
already flagged, is routed to ``fallback`` for a full (non-sliced) pull instead
- the sync never aborts over one awkward tensor.

``execute_transfer`` turns the planned segments into absolute-address
``ReadDescriptor``s (destination param ``data_ptr()`` resolved at refit time)
and hands them to a ``Transport``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from modelexpress.reshard_refit.slice_plan import _row_major_strides, plan_pull
from modelexpress.reshard_refit.transport import ReadDescriptor, Transport
from modelexpress.reshard_refit.types import CaptureResult, RecordedCopy, UnsupportedReshard


@dataclass
class SourceInfo:
    """Everything the planner needs about one full source tensor: its full shape,
    element dtype/size, and the shards the trainer published for it."""

    global_shape: tuple
    dtype: Any
    elsize: int
    shards: list  # list[Shard]


@dataclass
class ConvertSource:
    """A source whose SERVED dtype (``src_dtype``) differs from its dest param's
    dtype (e.g. a bf16-served router for an fp32 dest). The geometry is fine - only
    the dtype differs - and a raw RDMA copy is same-dtype only. So ``segments``
    target a per-param STAGING buffer of the SERVED dtype (fresh, contiguous,
    offset 0): the caller pulls into staging (same-dtype, valid), then casts
    staging -> the live param via ``copy_``. Keyed by ``param_name``."""

    param_name: str
    dest_shape: tuple
    src_dtype: Any  # served/wire dtype -> the staging buffer's dtype
    segments: list  # list[PullSegment], into the staging buffer


@dataclass
class TransferPlan:
    """Planned pull: ``segments`` are byte runs READ straight into live params;
    ``converts`` are dtype-mismatched sources to pull into staging then cast; and
    ``fallback`` names sources that couldn't be sliced at all (non-box op) and need
    a full materialization by the caller (the engine's own loader run)."""

    segments: list = field(default_factory=list)
    fallback: list = field(default_factory=list)
    converts: list = field(default_factory=list)  # list[ConvertSource]

    def bytes_planned(self) -> int:
        return sum(s.nbytes for s in self.segments)


def plan_transfer(capture: CaptureResult, sources: dict) -> TransferPlan:
    """Build a ``TransferPlan`` from captured copies + published ``sources``
    (``{src_name: SourceInfo}``). Sources flagged unsupported at capture, missing
    from ``sources``, or non-box at ``plan_pull`` fall back to a full pull."""
    plan = TransferPlan()
    fallback_seen: set = set()

    def mark_fallback(name: str) -> None:
        if name not in fallback_seen:
            fallback_seen.add(name)
            plan.fallback.append(name)

    # Sources whose loader used an unsupported op never made it into copies.
    for name in capture.unsupported:
        mark_fallback(name)

    for copy in capture.copies:
        src = sources.get(copy.src_name)
        if src is None:
            mark_fallback(copy.src_name)
            continue

        if src.dtype != copy.dest_dtype:
            # Served dtype != dest dtype (e.g. bf16-served router for an fp32 dest).
            # A raw RDMA copy is same-dtype only, so slice into a STAGING buffer of
            # the SERVED dtype (fresh + contiguous: offset 0, row-major stride, src
            # dtype so plan_pull's dtype/shape checks pass); the caller pulls into
            # staging then casts staging -> the live param.
            staging_copy = RecordedCopy(
                copy.src_name,
                copy.op_chain,
                copy.param_name,
                0,
                copy.dest_shape,
                _row_major_strides(copy.dest_shape),
                src.dtype,
            )
            try:
                segments = plan_pull(staging_copy, src.global_shape, src.dtype, src.elsize, src.shards)
            except UnsupportedReshard:
                mark_fallback(copy.src_name)
                continue
            plan.converts.append(ConvertSource(copy.param_name, tuple(copy.dest_shape), src.dtype, segments))
            continue

        try:
            segments = plan_pull(copy, src.global_shape, src.dtype, src.elsize, src.shards)
        except UnsupportedReshard:
            mark_fallback(copy.src_name)
            continue
        plan.segments.extend(segments)

    return plan


def execute_transfer(
    plan: TransferPlan,
    resolve_param_ptr: Callable[[str], int],
    transport: Transport,
) -> dict:
    """Resolve each planned segment's destination param name to a live
    ``data_ptr()`` (via ``resolve_param_ptr``), form absolute-address
    ``ReadDescriptor``s, and execute them on ``transport``.

    Returns a small stats dict; ``fallback`` is passed through for the caller to
    handle out-of-band (full pull + real loader run)."""
    descriptors = [
        ReadDescriptor(
            session=seg.session,
            src_addr=seg.src_addr,
            dst_addr=resolve_param_ptr(seg.param_name) + seg.dst_byte,
            nbytes=seg.nbytes,
        )
        for seg in plan.segments
    ]
    transport.read(descriptors)
    return {
        "segments": len(descriptors),
        "bytes": sum(d.nbytes for d in descriptors),
        "fallback": list(plan.fallback),
    }
