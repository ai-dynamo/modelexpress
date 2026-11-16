# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile trainer source geometry and bounded schedules without transport resources."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cached_property

import torch
from modelexpress_rl.refit.reshard.geometry import _validate_physical_destination
from modelexpress_rl.refit.reshard.slice_plan import _row_major_strides
from modelexpress_rl.refit.reshard.transfer_plan import (
    LogicalTransferPlan,
    SourceInfo,
    plan_transfer,
    replay_scratch_bytes,
)
from modelexpress_rl.refit.reshard.types import (
    CaptureResult,
    IncompleteRefit,
    UnsupportedReshard,
    summarize_unsupported,
)
from modelexpress_rl.version import MeshIdentity
from .plan import StreamingSettings


@dataclass(frozen=True)
class TransferBatch:
    """One logical replay batch and its compiled arena byte ranges."""

    transfer_plan: LogicalTransferPlan
    layout: dict[str, tuple[int, int, tuple[int, ...], torch.dtype]]
    nbytes: int


@dataclass(frozen=True)
class TrainerTransferPlan:
    """Compile and validate trainer geometry without transport resources.

    Own one capture's logical reads, optional bounded schedule and replay scratch
    analysis. Full-copy and bounded preparation share this compilation. The transfer
    executor later binds physical replicas, addresses and receive storage.
    """

    mesh_identity: MeshIdentity
    layout_capture_snapshot: CaptureResult
    parameter_layout: Mapping[str, tuple[tuple[int, ...], torch.dtype]]
    transfer_plan: LogicalTransferPlan
    batches: tuple[TransferBatch, ...] = ()

    @classmethod
    def compile(
        cls,
        *,
        mesh_identity: MeshIdentity,
        capture: CaptureResult,
        parameter_layout: Mapping[str, tuple[tuple[int, ...], torch.dtype]],
        sources: dict[str, SourceInfo],
        source_slots: dict[str, str] | None = None,
        streaming: StreamingSettings | None = None,
        metrics: dict[str, float] | None = None,
        verify_full_sources: bool = False,
    ) -> TrainerTransferPlan:
        if metrics is None:
            metrics = {}
        budget = (
            None
            if streaming is None
            else streaming.max_staging_bytes // streaming.staging_buffers
        )
        started = time.perf_counter()
        logical = plan_transfer(
            capture,
            sources,
            source_slots=source_slots,
            verify_full_sources=verify_full_sources,
            max_staging_bytes=budget,
        )
        metrics["initial_whole_plan_s"] = time.perf_counter() - started
        started = time.perf_counter()
        cls.validate_capture(capture, parameter_layout, logical)
        metrics["initial_whole_validation_s"] = time.perf_counter() - started
        batches = (
            ()
            if streaming is None
            else _bounded_batches(
                logical,
                budget,
                total_staging_bytes=streaming.max_staging_bytes,
                staging_buffers=streaming.staging_buffers,
            )
        )
        return cls(mesh_identity, capture, parameter_layout, logical, batches)

    def validate(self) -> None:
        self.validate_capture(
            self.layout_capture_snapshot, self.parameter_layout, self.transfer_plan
        )

    @cached_property
    def arena_bytes(self) -> int:
        return max((batch.nbytes for batch in self.batches), default=0)

    @cached_property
    def replay_scratch_peak_bytes(self) -> int:
        return max(
            (replay_scratch_bytes(pull) for pull in self.transfer_plan.full_pulls),
            default=0,
        )

    @staticmethod
    def validate_capture(
        capture: CaptureResult,
        parameter_layout: Mapping[str, tuple[tuple[int, ...], torch.dtype]],
        plan: LogicalTransferPlan,
    ) -> None:
        if not capture.source_replay or not plan.source_replay:
            raise IncompleteRefit("trainer transfer requires ordered source replay")
        if plan.replay_id != capture.replay_id:
            raise IncompleteRefit("trainer transfer belongs to a different captured replay plan")
        for copy in capture.copies:
            shape, dtype = parameter_layout.get(copy.param_name, ((), None))
            owner_shape, owner_stride = capture.destination_layouts.get(
                copy.param_name, (shape, tuple(_row_major_strides(shape)))
            )
            try:
                if dtype != copy.dest_dtype or tuple(owner_shape) != tuple(shape):
                    raise UnsupportedReshard(
                        "recorded destination owner layout differs"
                    )
                _validate_physical_destination(
                    copy.dest_shape,
                    copy.dest_stride,
                    copy.dest_offset,
                    owner_shape,
                    owner_stride,
                )
            except UnsupportedReshard as error:
                raise IncompleteRefit(
                    f"destination {copy.param_name!r} write exceeds physical owner bounds: {error}"
                ) from error
        written = {copy.param_name for copy in capture.copies}
        missing = sorted(set(parameter_layout) - written)
        unsupported = list(capture.unsupported)
        if missing or unsupported or capture.unattributed or plan.fallback:
            causes = summarize_unsupported(capture.unsupported_reasons)
            raise IncompleteRefit(
                "full-tensor refit must cover every engine parameter; "
                f"missing={len(missing)}, unsupported={len(unsupported)}, "
                f"unattributed={capture.unattributed}, fallback={len(plan.fallback)}, "
                f"causes={causes}, missing_names={missing[:10]}"
            )


def _bounded_batches(
    complete: LogicalTransferPlan,
    max_staging_bytes: int,
    *,
    total_staging_bytes: int,
    staging_buffers: int,
) -> tuple[TransferBatch, ...]:
    """Partition a validated plan into consecutive writes that fit one arena."""
    if (
        isinstance(max_staging_bytes, bool)
        or not isinstance(max_staging_bytes, int)
        or max_staging_bytes <= 0
    ):
        raise ValueError("max_staging_bytes must be a positive integer")
    batches, current, layout = [], [], {}
    current_bytes = 0

    def append_batch():
        plan = replace(complete, full_pulls=tuple(current))
        batches.append(TransferBatch(plan, layout, current_bytes))

    for pull in complete.full_pulls:
        raw_bytes = math.prod(pull.global_shape) * pull.dtype.itemsize
        nbytes = ((raw_bytes + 255) // 256) * 256
        if nbytes > max_staging_bytes:
            raise IncompleteRefit(
                f"recorded source write requires {nbytes} staging bytes; "
                f"max_staging_bytes={total_staging_bytes or max_staging_bytes}, "
                f"staging_buffers={staging_buffers} gives {max_staging_bytes} bytes per arena"
            )
        if current and current_bytes + nbytes > max_staging_bytes:
            append_batch()
            current, layout, current_bytes = [], {}, 0
        layout[pull.src_name] = (
            current_bytes, current_bytes + raw_bytes,
            tuple(pull.global_shape), pull.dtype,
        )
        current.append(pull)
        current_bytes += nbytes
    if current:
        append_batch()
    if not batches:
        raise IncompleteRefit("bounded refit has no recorded writes")
    return tuple(batches)
