# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Transfer plan + pull execution.

Ties the pieces together: take a ``CaptureResult`` and the published source
shards, run ``plan_pull`` per captured copy, and collect the byte segments into a
``LogicalTransferPlan``. Any source whose slice can't be expressed as a box
(``UnsupportedReshard``), or that has no published shards, or that capture
already flagged, is recorded in ``fallback`` as unsupported. The current
receiver rejects such plans before transfer. Descriptor-heavy but otherwise
supported strided copies use a separate bounded full-source staging path.

``bind_transfer`` resolves source and destination buffers into a separate
``PhysicalTransferPlan``. ``execute_transfer`` submits its exact phase; receivers
also use its full-pull and conversion phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from modelexpress import envs
from modelexpress.refit.reshard.slice_plan import (
    LogicalRead,
    SourceBuffer,
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


@dataclass(frozen=True)
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
    segments: tuple


@dataclass(frozen=True)
class FullPullSource:
    """Descriptor-heavy source reconstructed once in contiguous staging.

    Captured copies are replayed locally from the staged full tensor. This
    trades bounded extra wire bytes for a bounded number of NIXL descriptors.
    """

    src_name: str
    global_shape: tuple
    dtype: Any
    elsize: int
    segments: tuple
    copies: tuple


@dataclass(frozen=True)
class LogicalTransferPlan:
    """Planned pull: ``segments`` are byte runs READ straight into live params;
    ``converts`` are dtype-mismatched sources to pull into staging then cast;
    ``full_pulls`` bound descriptor-heavy copies through contiguous source
    staging; and ``fallback`` retains the historical field name for unsupported
    sources that cannot be updated. The current receiver fails closed when it is
    non-empty."""

    source_buffers: tuple[SourceBuffer, ...] = ()
    segments: tuple = ()
    fallback: tuple = ()
    converts: tuple = ()
    full_pulls: tuple = ()
    unbounded_sources: tuple = ()
    exact_descriptor_count: int = 0
    exact_bytes: int = 0
    verify_full_sources: bool = False

    def _all_segments(self):
        """Every planned segment, wherever it lands.

        Segments live in three places - read straight into live params, into
        dtype-conversion staging, and into full-pull staging - and any question
        about the plan as a whole has to ask all three.
        """
        yield from self.segments
        for convert in self.converts:
            yield from convert.segments
        for full_pull in self.full_pulls:
            yield from full_pull.segments

    def bytes_planned(self) -> int:
        return sum(segment.nbytes for segment in self._all_segments())

    def descriptor_count(self) -> int:
        return sum(1 for _ in self._all_segments())

    def descriptor_savings(self) -> int:
        return max(0, self.exact_descriptor_count - self.descriptor_count())

    def extra_wire_bytes(self) -> int:
        return max(0, self.bytes_planned() - self.exact_bytes)


def _full_pull_segments(
    src_name: str, source: SourceInfo, buffer_indices: tuple[int, ...]
) -> tuple[LogicalRead, ...]:
    """Reconstruct a complete dim-0-sharded tensor in contiguous staging.

    Layouts that are not a gap-free dim-0 partition are rejected so the caller
    can retain the exact descriptor plan rather than risking incorrect staging.
    """
    if not source.global_shape:
        raise UnsupportedReshard(f"{src_name}: scalar full-pull is unsupported")

    trailing = 1
    for extent in source.global_shape[1:]:
        trailing *= int(extent)

    ordered = sorted(
        zip(buffer_indices, source.shards, strict=True),
        key=lambda item: int(item[1].shard_offset[0]),
    )
    segments = []
    next_row = 0
    for buffer_index, shard in ordered:
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
            LogicalRead(
                source_buffer=buffer_index,
                src_byte=0,
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
    return tuple(segments)


def _plan_verified_transfer(
    capture, sources, source_buffers, source_indices, max_segments_per_copy
) -> LogicalTransferPlan:
    """Reconstruct sources without an identity copy for whole-shard checksums."""
    copies_by_source = {}
    for copy in capture.copies:
        copies_by_source.setdefault(copy.src_name, []).append(copy)
    segments, converts, full_pulls, unbounded_sources = [], [], [], []
    fallback = list(dict.fromkeys(capture.unsupported))
    exact_count = exact_bytes = 0
    for name, source in sources.items():
        copies = copies_by_source.pop(name, [])
        if not copies:
            continue
        if any(
            not copy.op_chain and tuple(copy.dest_shape) == tuple(source.global_shape)
            for copy in copies
        ):
            part = _plan_transfer(
                CaptureResult(copies=copies), {name: source},
                source_buffers, source_indices,
                max_segments_per_copy=max_segments_per_copy,
            )
            segments.extend(part.segments)
            converts.extend(part.converts)
            full_pulls.extend(part.full_pulls)
            unbounded_sources.extend(part.unbounded_sources)
            fallback.extend(item for item in part.fallback if item not in fallback)
            exact_count += part.exact_descriptor_count
            exact_bytes += part.exact_bytes
            continue
        identity = RecordedCopy(
            name, (), name, 0, tuple(source.global_shape),
            tuple(_row_major_strides(source.global_shape)), source.dtype,
        )
        reads = plan_pull(
            identity, source.global_shape, source.dtype, source.elsize,
            source.shards, buffer_indices=source_indices[name],
        )
        full_pulls.append(FullPullSource(
            name, tuple(source.global_shape), source.dtype, source.elsize,
            reads, tuple(copies),
        ))
        exact_count += len(reads)
        exact_bytes += sum(read.nbytes for read in reads)
    fallback.extend(name for name in copies_by_source if name not in fallback)
    return LogicalTransferPlan(
        source_buffers=source_buffers, segments=tuple(segments),
        fallback=tuple(fallback), converts=tuple(converts),
        full_pulls=tuple(full_pulls), unbounded_sources=tuple(unbounded_sources),
        exact_descriptor_count=exact_count, exact_bytes=exact_bytes,
        verify_full_sources=True,
    )


def plan_transfer(
    capture: CaptureResult,
    sources: dict,
    *,
    max_segments_per_copy: int | None = None,
    source_slots: dict[str, str] | None = None,
    verify_full_sources: bool = False,
) -> LogicalTransferPlan:
    """Build a ``LogicalTransferPlan`` from captured copies + published ``sources``
    (``{src_name: SourceInfo}``). Sources flagged unsupported at capture, missing
    from ``sources``, or non-box at ``plan_pull`` are marked unsupported.

    ``source_slots`` maps transport sessions to logical source slots; omission
    uses session identity. ``verify_full_sources`` reconstructs sources lacking
    an identity copy so receivers can verify whole-shard checksums.
    """
    source_buffers = []
    source_indices = {}
    needed_names = dict.fromkeys(copy.src_name for copy in capture.copies)
    for name in needed_names:
        source = sources.get(name)
        if source is None:
            continue
        indices = []
        for shard in source.shards:
            indices.append(len(source_buffers))
            source_buffers.append(SourceBuffer(
                shard.session if source_slots is None else source_slots[shard.session],
                name, tuple(shard.shard_offset), tuple(shard.shape),
            ))
        source_indices[name] = tuple(indices)
    source_buffers = tuple(source_buffers)
    return _plan_transfer(
        capture, sources, source_buffers, source_indices,
        max_segments_per_copy=max_segments_per_copy,
        verify_full_sources=verify_full_sources,
    )


def plan_transfer_groups(
    captures, sources: dict, complete_plan: LogicalTransferPlan, *,
    max_segments_per_copy: int | None = None,
):
    """Plan subsets of the complete capture using the same source geometry.

    The source-buffer table and one shared index map stay inside the planner.
    Full-source verification follows the complete plan.
    """
    source_indices = {}
    for index, buffer in enumerate(complete_plan.source_buffers):
        source_indices.setdefault(buffer.src_name, []).append(index)
    source_indices = {name: tuple(indices) for name, indices in source_indices.items()}
    for capture in captures:
        yield _plan_transfer(
            capture, sources, complete_plan.source_buffers, source_indices,
            max_segments_per_copy=max_segments_per_copy,
            verify_full_sources=complete_plan.verify_full_sources,
        )


def _plan_transfer(
    capture, sources, source_buffers, source_indices, *,
    max_segments_per_copy=None, verify_full_sources=False,
) -> LogicalTransferPlan:
    if max_segments_per_copy is None:
        max_segments_per_copy = envs.MX_RESHARD_MAX_SEGMENTS_PER_COPY
    if max_segments_per_copy < 1:
        raise ValueError("max_segments_per_copy must be at least 1")
    if verify_full_sources:
        return _plan_verified_transfer(
            capture, sources, source_buffers, source_indices, max_segments_per_copy
        )
    exact_segments = []
    fallback = []
    converts = []
    full_pulls = []
    unbounded_sources = []
    exact_descriptor_count = 0
    exact_bytes = 0
    fallback_seen: set = set()
    exact_by_source: dict[str, list[tuple[RecordedCopy, list]]] = {}
    full_pull_names: set[str] = set()

    def mark_fallback(name: str) -> None:
        if name not in fallback_seen:
            fallback_seen.add(name)
            fallback.append(name)

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
                dest_stride=tuple(_row_major_strides(copy.dest_shape)),
                dest_dtype=src.dtype,
            )
            try:
                segments = plan_pull(
                    staging_copy, src.global_shape, src.dtype, src.elsize, src.shards,
                    buffer_indices=source_indices[copy.src_name],
                )
            except UnsupportedReshard:
                mark_fallback(copy.src_name)
                continue
            exact_descriptor_count += len(segments)
            exact_bytes += sum(segment.nbytes for segment in segments)
            converts.append(
                ConvertSource(
                    copy.param_name, tuple(copy.dest_shape), src.dtype, segments
                )
            )
            continue

        try:
            segments = plan_pull(
                copy, src.global_shape, src.dtype, src.elsize, src.shards,
                buffer_indices=source_indices[copy.src_name],
            )
        except UnsupportedReshard:
            mark_fallback(copy.src_name)
            continue
        exact_descriptor_count += len(segments)
        exact_bytes += sum(segment.nbytes for segment in segments)
        exact_by_source.setdefault(copy.src_name, []).append((copy, segments))
        if len(segments) > max_segments_per_copy:
            full_pull_names.add(copy.src_name)

    for src_name, entries in exact_by_source.items():
        if src_name not in full_pull_names:
            for _copy, segments in entries:
                exact_segments.extend(segments)
            continue

        source = sources[src_name]
        try:
            full_segments = _full_pull_segments(src_name, source, source_indices[src_name])
        except UnsupportedReshard:
            # Descriptor bounding is an optimization. Preserve the known-correct
            # exact plan when this source cannot be reconstructed contiguously.
            unbounded_sources.append(src_name)
            for _copy, segments in entries:
                exact_segments.extend(segments)
            continue

        full_pulls.append(
            FullPullSource(
                src_name=src_name,
                global_shape=tuple(source.global_shape),
                dtype=source.dtype,
                elsize=source.elsize,
                segments=full_segments,
                copies=tuple(copy for copy, _segments in entries),
            )
        )

    return LogicalTransferPlan(
        source_buffers=tuple(source_buffers), segments=tuple(exact_segments),
        fallback=tuple(fallback), converts=tuple(converts), full_pulls=tuple(full_pulls),
        unbounded_sources=tuple(unbounded_sources),
        exact_descriptor_count=exact_descriptor_count, exact_bytes=exact_bytes,
    )


@dataclass(frozen=True)
class PhysicalTransferPlan:
    """An execution binding; its addresses and sessions never enter geometry."""

    segments: tuple[ReadDescriptor, ...]
    full_pulls: tuple[tuple[ReadDescriptor, ...], ...]
    converts: tuple[tuple[ReadDescriptor, ...], ...]
    fallback: tuple

    @property
    def descriptors(self) -> tuple[ReadDescriptor, ...]:
        return self.segments + tuple(
            read for phase in (*self.full_pulls, *self.converts) for read in phase
        )


def bind_transfer(
    plan: LogicalTransferPlan,
    resolve_source: Callable[[SourceBuffer], tuple[str, int]],
    resolve_destination: Callable[[str, str], int],
) -> PhysicalTransferPlan:
    """Resolve each used source buffer and destination once for this execution.

    ``resolve_source(buffer)`` returns ``(session, base_address)``.
    ``resolve_destination(kind, name)`` returns the destination base address;
    ``kind`` is ``recv``, ``full`` or ``convert``.
    """
    sources = {}
    destinations = {"recv": {}, "full": {}, "convert": {}}

    def bind(reads, kind):
        result = []
        targets = destinations[kind]
        for read in reads:
            index = read.source_buffer
            if index < 0 or index >= len(plan.source_buffers):
                raise ValueError(f"source buffer index out of range: {index}")
            if index not in sources:
                sources[index] = resolve_source(plan.source_buffers[index])
            session, address = sources[index]
            target = read.param_name
            if target not in targets:
                targets[target] = resolve_destination(kind, target)
            result.append(ReadDescriptor(
                session=session, src_addr=address + read.src_byte,
                dst_addr=targets[target] + read.dst_byte, nbytes=read.nbytes,
            ))
        return tuple(result)

    return PhysicalTransferPlan(
        bind(plan.segments, "recv"),
        tuple(bind(p.segments, "full") for p in plan.full_pulls),
        tuple(bind(p.segments, "convert") for p in plan.converts),
        plan.fallback,
    )


def execute_transfer(plan: PhysicalTransferPlan, transport: Transport) -> dict:
    """Execute the exact phase of a bound plan."""
    transport.read(list(plan.segments))
    return {
        "segments": len(plan.segments),
        "bytes": sum(d.nbytes for d in plan.segments),
        "fallback": list(plan.fallback),
    }
