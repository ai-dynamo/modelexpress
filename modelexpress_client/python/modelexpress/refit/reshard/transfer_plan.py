# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Transfer plan + pull execution.

Ties the pieces together: take a ``CaptureResult`` and the published source
shards, run ``plan_pull`` per captured copy, and collect the byte segments into a
``TransferPlan``. Any source whose slice can't be expressed as a box
(``UnsupportedReshard``), or that has no published shards, or that capture
already flagged, is routed to ``fallback``. Descriptor-heavy but otherwise
supported strided copies use a separate bounded full-source staging path.

``execute_transfer`` turns the planned segments into absolute-address
``ReadDescriptor``s (destination param ``data_ptr()`` resolved at refit time)
and hands them to a ``Transport``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from modelexpress.refit.reshard.slice_plan import (
    PullSegment,
    _row_major_strides,
    plan_pull,
)
from modelexpress.refit.reshard.transport import ReadDescriptor, Transport
from modelexpress.refit.reshard.types import (
    CaptureResult,
    RecordedCopy,
    UnsupportedReshard,
)


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
class FullPullSource:
    """Descriptor-heavy source reconstructed once in contiguous staging.

    Captured copies are replayed locally from the staged full tensor. This
    trades bounded extra wire bytes for a bounded number of NIXL descriptors.
    """

    src_name: str
    global_shape: tuple
    dtype: Any
    elsize: int
    segments: list
    copies: list


@dataclass
class TransferPlan:
    """Planned pull: ``segments`` are byte runs READ straight into live params;
    ``converts`` are dtype-mismatched sources to pull into staging then cast;
    ``full_pulls`` bound descriptor-heavy copies through contiguous source
    staging; and ``fallback`` names unsupported sources that cannot be updated."""

    segments: list = field(default_factory=list)
    fallback: list = field(default_factory=list)
    converts: list = field(default_factory=list)  # list[ConvertSource]
    full_pulls: list = field(default_factory=list)  # list[FullPullSource]
    unbounded_sources: list = field(default_factory=list)
    exact_descriptor_count: int = 0
    exact_bytes: int = 0
    # Which arm of the destination-digest comparison this plan is. Recorded on the
    # plan rather than re-read from the environment at emit time so a record cannot
    # claim to be one arm while having been planned as the other - the whole
    # comparison rests on telling them apart.
    forced_full_pull: bool = False

    def bytes_planned(self) -> int:
        return (
            sum(segment.nbytes for segment in self.segments)
            + sum(
                segment.nbytes
                for convert in self.converts
                for segment in convert.segments
            )
            + sum(
                segment.nbytes
                for full_pull in self.full_pulls
                for segment in full_pull.segments
            )
        )

    def descriptor_count(self) -> int:
        return (
            len(self.segments)
            + sum(len(convert.segments) for convert in self.converts)
            + sum(len(full_pull.segments) for full_pull in self.full_pulls)
        )

    def descriptor_savings(self) -> int:
        return max(0, self.exact_descriptor_count - self.descriptor_count())

    def extra_wire_bytes(self) -> int:
        return max(0, self.bytes_planned() - self.exact_bytes)


def _full_pull_segments(src_name: str, source: SourceInfo) -> list:
    """Reconstruct a complete dim-0-sharded tensor in contiguous staging.

    Layouts that are not a gap-free dim-0 partition are rejected so the caller
    can retain the exact descriptor plan rather than risking incorrect staging.
    """
    if not source.global_shape:
        raise UnsupportedReshard(f"{src_name}: scalar full-pull is unsupported")

    trailing = 1
    for extent in source.global_shape[1:]:
        trailing *= int(extent)

    ordered = sorted(source.shards, key=lambda shard: int(shard.shard_offset[0]))
    segments = []
    next_row = 0
    for shard in ordered:
        if len(shard.shape) != len(source.global_shape) or len(
            shard.shard_offset
        ) != len(source.global_shape):
            raise UnsupportedReshard(
                f"{src_name}: shard rank does not match global rank"
            )
        if tuple(shard.shape[1:]) != tuple(source.global_shape[1:]) or any(
            int(offset) != 0 for offset in shard.shard_offset[1:]
        ):
            raise UnsupportedReshard(
                f"{src_name}: bounded full-pull requires dim-0 shards, got "
                f"offset={shard.shard_offset} shape={shard.shape}"
            )

        start = int(shard.shard_offset[0])
        rows = int(shard.shape[0])
        if start != next_row:
            raise UnsupportedReshard(
                f"{src_name}: dim-0 shards have a gap or overlap at row "
                f"{next_row} (next shard starts at {start})"
            )
        elements = rows * trailing
        segments.append(
            PullSegment(
                session=shard.session,
                src_addr=shard.addr,
                param_name=src_name,
                dst_byte=start * trailing * source.elsize,
                nbytes=elements * source.elsize,
            )
        )
        next_row += rows

    if next_row != int(source.global_shape[0]):
        raise UnsupportedReshard(
            f"{src_name}: dim-0 shards cover {next_row} of "
            f"{source.global_shape[0]} rows"
        )
    return segments


def plan_transfer(
    capture: CaptureResult,
    sources: dict,
    *,
    max_segments_per_copy: int | None = None,
    force_full_pull: bool | None = None,
) -> TransferPlan:
    """Build a ``TransferPlan`` from captured copies + published ``sources``
    (``{src_name: SourceInfo}``). Sources flagged unsupported at capture, missing
    from ``sources``, or non-box at ``plan_pull`` fall back to a full pull.

    ``force_full_pull`` promotes *every* source to a full pull regardless of how few
    descriptors its exact fetch would need. It exists for verification runs, and it
    trades wire volume - which a correctness run does not care about - for gate
    coverage, which it does.

    The reason it is worth having: ``verify_full_pulls`` can only digest sources
    whose whole shard lands in a staging buffer, because an exact-fetch segment is
    scattered straight into a live param and digesting it would mean digesting the
    destination layout instead of the source shard. On Topology B that left 6 192 of
    18 867 sources checked - the row-parallel tensors that happened to be strided at
    generator TP2 - and said nothing about the other 12 675, which include every
    norm, the gate/up projections and all attention except ``o_proj``.

    Read a pass from this arm *alone* narrowly. It proves the publisher's bytes for
    every source arrive intact, which is what Bug 6 and Bug 9 were about. On its own
    it does **not** verify the exact-fetch path, because forcing full pulls replaces
    the segment planning under test rather than checking it, so a bug in
    ``plan_pull``'s slicing would be hidden rather than caught.

    What closes that gap is using this arm as the *reference* in a two-run
    comparison rather than as a gate in itself: run once normally and once forced,
    digest the assembled destination both times (``MX_RESHARD_DEST_DIGEST=1``), and
    diff per parameter. Bytes reaching the destination through exact segments are
    then checked against bytes reaching it through a path whose contents are
    independently confirmed against the publishers' digests, and a difference names
    the tensor. See     ``modelexpress.refit.reshard.verify.digest_destination`` and
    ``modelexpress.refit.reshard.dest_digest_report``. The comparison needs the
    weights to hold still across the two runs, so it wants a quiesced publisher.
    """
    plan = TransferPlan()
    fallback_seen: set = set()
    exact_by_source: dict[str, list[tuple[RecordedCopy, list]]] = {}
    full_pull_names: set[str] = set()
    if max_segments_per_copy is None:
        max_segments_per_copy = int(
            os.environ.get("MX_RESHARD_MAX_SEGMENTS_PER_COPY", "64")
        )
    if max_segments_per_copy < 1:
        raise ValueError("max_segments_per_copy must be at least 1")
    if force_full_pull is None:
        force_full_pull = (
            os.environ.get("MX_RESHARD_FORCE_FULL_PULL", "0") == "1"
        )
    plan.forced_full_pull = bool(force_full_pull)

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
                src_name=copy.src_name,
                op_chain=copy.op_chain,
                param_name=copy.param_name,
                dest_offset=0,
                dest_shape=copy.dest_shape,
                dest_stride=_row_major_strides(copy.dest_shape),
                dest_dtype=src.dtype,
            )
            try:
                segments = plan_pull(
                    staging_copy, src.global_shape, src.dtype, src.elsize, src.shards
                )
            except UnsupportedReshard:
                mark_fallback(copy.src_name)
                continue
            plan.exact_descriptor_count += len(segments)
            plan.exact_bytes += sum(segment.nbytes for segment in segments)
            plan.converts.append(
                ConvertSource(
                    copy.param_name, tuple(copy.dest_shape), src.dtype, segments
                )
            )
            continue

        try:
            segments = plan_pull(
                copy, src.global_shape, src.dtype, src.elsize, src.shards
            )
        except UnsupportedReshard:
            mark_fallback(copy.src_name)
            continue
        plan.exact_descriptor_count += len(segments)
        plan.exact_bytes += sum(segment.nbytes for segment in segments)
        exact_by_source.setdefault(copy.src_name, []).append((copy, segments))
        if force_full_pull or len(segments) > max_segments_per_copy:
            full_pull_names.add(copy.src_name)

    for src_name, entries in exact_by_source.items():
        if src_name not in full_pull_names:
            for _copy, segments in entries:
                plan.segments.extend(segments)
            continue

        source = sources[src_name]
        try:
            full_segments = _full_pull_segments(src_name, source)
        except UnsupportedReshard:
            # Descriptor bounding is an optimization. Preserve the known-correct
            # exact plan when this source cannot be reconstructed contiguously.
            plan.unbounded_sources.append(src_name)
            for _copy, segments in entries:
                plan.segments.extend(segments)
            continue

        plan.full_pulls.append(
            FullPullSource(
                src_name=src_name,
                global_shape=tuple(source.global_shape),
                dtype=source.dtype,
                elsize=source.elsize,
                segments=full_segments,
                copies=[copy for copy, _segments in entries],
            )
        )

    return plan


def session_distribution(plan: TransferPlan) -> dict:
    """Bytes and descriptors this plan will READ from each remote session.

    A session is one publishing trainer rank, so this is the per-peer read load a
    single receiver imposes. It matters because duplicate sources are discarded on
    a first-writer-wins basis (``rendezvous.merge_shard_tables``), which can
    concentrate the whole fleet's reads onto a few ranks while byte-identical DP /
    EDP replicas sit idle. Without this, a sender-side egress bottleneck is
    indistinguishable from a per-transfer efficiency limit.

    Returns ``{session: {"bytes": int, "descriptors": int}}`` over every phase.
    """
    dist: dict = {}

    def add(session, nbytes):
        entry = dist.setdefault(session, {"bytes": 0, "descriptors": 0})
        entry["bytes"] += nbytes
        entry["descriptors"] += 1

    for segment in plan.segments:
        add(segment.session, segment.nbytes)
    for convert in plan.converts:
        for segment in convert.segments:
            add(segment.session, segment.nbytes)
    for full_pull in plan.full_pulls:
        for segment in full_pull.segments:
            add(segment.session, segment.nbytes)
    return dist


def plan_threshold_curve(
    capture: CaptureResult,
    sources: dict,
    thresholds: list,
) -> list:
    """Descriptors-vs-bytes tradeoff at several promotion thresholds.

    The full-pull promotion in ``plan_transfer`` trades wire bytes for NIXL
    descriptors, but the threshold that controls it is a cliff rather than a
    comparison, so the shape of that tradeoff has to be measured before a cost
    model can replace it. Building a throwaway plan per threshold yields the
    whole curve from one process, instead of one point per redeployment.

    Diagnostic only - the returned plans are discarded and the caller's real plan
    is unaffected. Returns one row per threshold, in the order given.
    """
    curve = []
    for threshold in thresholds:
        plan = plan_transfer(capture, sources, max_segments_per_copy=threshold)
        curve.append(
            {
                "threshold": threshold,
                "descriptors": plan.descriptor_count(),
                "descriptors_saved": plan.descriptor_savings(),
                "bytes_planned": plan.bytes_planned(),
                "useful_bytes": plan.exact_bytes,
                "extra_wire_bytes": plan.extra_wire_bytes(),
                "full_pulls": len(plan.full_pulls),
                "unbounded_sources": len(plan.unbounded_sources),
                "converts": len(plan.converts),
                "fallback": len(plan.fallback),
            }
        )
    return curve


def exact_descriptors(
    plan: TransferPlan,
    resolve_param_ptr: Callable[[str], int],
) -> list:
    """Absolute-address ``ReadDescriptor``s for the exact-segment phase.

    Split out from :func:`execute_transfer` so a caller that wants to issue the
    exact, full-pull and convert phases as one batch can build the descriptors
    without also performing the read.
    """
    return [
        ReadDescriptor(
            session=seg.session,
            src_addr=seg.src_addr,
            dst_addr=resolve_param_ptr(seg.param_name) + seg.dst_byte,
            nbytes=seg.nbytes,
        )
        for seg in plan.segments
    ]


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
    descriptors = exact_descriptors(plan, resolve_param_ptr)
    transport.read(descriptors)
    return {
        "segments": len(descriptors),
        "bytes": sum(d.nbytes for d in descriptors),
        "fallback": list(plan.fallback),
    }
