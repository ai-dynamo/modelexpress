# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured, normalized timing for ModelExpress refit cycles.

The recorder is deliberately independent of torch and vLLM.  A cycle owner
activates it with :func:`use_refit_timing`; lower layers then contribute spans
without adding timing parameters to their public APIs.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


MX_REFIT_TIMING_PREFIX = "MX_REFIT_TIMING"
REFIT_TIMING_STAGES = (
    "control_discovery",
    "source_preparation",
    "setup_registration",
    "transfer_planning",
    "wire_transfer",
    "receive_sync",
    "transformation",
    "installation",
    "post_install",
    "rollout_readiness",
)
_STAGE_SET = frozenset(REFIT_TIMING_STAGES)
_current_recorder: contextvars.ContextVar["RefitTimingRecorder | None"] = (
    contextvars.ContextVar("mx_refit_timing_recorder", default=None)
)


@dataclass
class _Stage:
    duration_s: float = 0.0
    count: int = 0
    statuses: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class RefitTimingRecorder:
    """Collect repeated normalized spans and emit one stable JSON record."""

    def __init__(
        self,
        *,
        backend: str,
        version: int | str,
        rank: int | None = None,
        tp_rank: int | None = None,
        tp_size: int | None = None,
        ep_rank: int | None = None,
        ep_size: int | None = None,
        cold: bool | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.backend = backend
        self.version = version
        self.rank = rank
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.cold = cold
        self.bytes = 0
        self._clock = clock
        self._started_at = clock()
        self._finished_at: float | None = None
        self._stages = {name: _Stage() for name in REFIT_TIMING_STAGES}
        self._emitted = False

    @contextlib.contextmanager
    def span(
        self,
        stage: str,
        *,
        status: str = "ok",
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        """Measure a stage; failed spans are retained and re-raised."""
        self._validate_stage(stage)
        started = self._clock()
        try:
            yield
        except BaseException:
            self.add_duration(
                stage,
                self._clock() - started,
                status="error",
                metadata=metadata,
            )
            raise
        else:
            self.add_duration(
                stage,
                self._clock() - started,
                status=status,
                metadata=metadata,
            )

    def add_duration(
        self,
        stage: str,
        duration_s: float,
        *,
        status: str = "ok",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add an externally measured duration to a normalized stage."""
        self._validate_stage(stage)
        if duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        item = self._stages[stage]
        item.duration_s += float(duration_s)
        item.count += 1
        if status not in item.statuses:
            item.statuses.append(status)
        if metadata:
            item.metadata.update(metadata)

    def mark_not_applicable(
        self,
        stage: str,
        *,
        combined_with: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Explicitly mark a stage absent from this backend's live path."""
        self._validate_stage(stage)
        metadata = {}
        if combined_with is not None:
            self._validate_stage(combined_with)
            metadata["combined_with"] = combined_with
        if reason is not None:
            metadata["reason"] = reason
        item = self._stages[stage]
        if item.count:
            raise ValueError(f"stage {stage!r} already has measured spans")
        item.statuses = ["combined" if combined_with else "not_applicable"]
        item.metadata.update(metadata)

    def add_bytes(self, count: int) -> None:
        if count < 0:
            raise ValueError("byte count must be non-negative")
        self.bytes += int(count)

    def set_cold(self, cold: bool) -> None:
        self.cold = bool(cold)

    def has_measurements(self, stage: str) -> bool:
        """Return whether ``stage`` contains at least one measured span."""
        self._validate_stage(stage)
        return self._stages[stage].count > 0

    def finish(self) -> None:
        if self._finished_at is None:
            self._finished_at = self._clock()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record in normalized stage order."""
        end = self._finished_at if self._finished_at is not None else self._clock()
        e2e_s = max(0.0, end - self._started_at)
        measured_s = sum(item.duration_s for item in self._stages.values())
        stages: dict[str, dict[str, Any]] = {}
        for name in REFIT_TIMING_STAGES:
            item = self._stages[name]
            if not item.statuses:
                status = "not_recorded"
            elif len(item.statuses) == 1:
                status = item.statuses[0]
            else:
                status = "mixed"
            value: dict[str, Any] = {
                "status": status,
                "duration_ms": round(item.duration_s * 1000.0, 3),
                "count": item.count,
            }
            if len(item.statuses) > 1:
                value["statuses"] = list(item.statuses)
            if item.metadata:
                value["metadata"] = dict(item.metadata)
            stages[name] = value
        return {
            "backend": self.backend,
            "version": self.version,
            "rank": self.rank,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "ep_rank": self.ep_rank,
            "ep_size": self.ep_size,
            "bytes": self.bytes,
            "cold_warm": (
                "unknown" if self.cold is None else ("cold" if self.cold else "warm")
            ),
            "stages": stages,
            "e2e_ms": round(e2e_s * 1000.0, 3),
            "unattributed_ms": round(max(0.0, e2e_s - measured_s) * 1000.0, 3),
        }

    def emit(self, logger: logging.Logger) -> dict[str, Any]:
        """Log the completed record once and return its dictionary form.

        The record is logged at INFO. Because some deployments (e.g. the Dynamo
        rollout) filter INFO from worker log capture, the marker line is also
        written to stdout with an explicit flush when ``MX_REFIT_TIMING_STDOUT``
        is set to a non-``0`` value, so benchmark harnesses can reliably grep the
        per-stage record regardless of logger configuration.
        """
        self.finish()
        payload = self.as_dict()
        if not self._emitted:
            line = "%s %s" % (
                MX_REFIT_TIMING_PREFIX,
                json.dumps(
                    payload,
                    separators=(",", ":"),
                    sort_keys=False,
                    default=str,
                ),
            )
            logger.info("%s", line)
            if os.environ.get("MX_REFIT_TIMING_STDOUT", "0") != "0":
                print(line, flush=True, file=sys.stdout)
            self._emitted = True
        return payload

    @staticmethod
    def _validate_stage(stage: str) -> None:
        if stage not in _STAGE_SET:
            raise ValueError(
                f"unknown refit timing stage {stage!r}; expected one of "
                f"{REFIT_TIMING_STAGES}"
            )


def current_refit_timing() -> RefitTimingRecorder | None:
    """Return the recorder active in this context, if any."""
    return _current_recorder.get()


@contextlib.contextmanager
def use_refit_timing(recorder: RefitTimingRecorder) -> Iterator[RefitTimingRecorder]:
    """Make ``recorder`` visible to nested ModelExpress layers."""
    token = _current_recorder.set(recorder)
    try:
        yield recorder
    finally:
        _current_recorder.reset(token)


@contextlib.contextmanager
def refit_span(
    stage: str,
    *,
    status: str = "ok",
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Record a span when a cycle is active; otherwise act as a no-op."""
    recorder = current_refit_timing()
    if recorder is None:
        yield
        return
    with recorder.span(stage, status=status, metadata=metadata):
        yield


def add_refit_bytes(count: int) -> None:
    recorder = current_refit_timing()
    if recorder is not None:
        recorder.add_bytes(count)


__all__ = [
    "MX_REFIT_TIMING_PREFIX",
    "REFIT_TIMING_STAGES",
    "RefitTimingRecorder",
    "add_refit_bytes",
    "current_refit_timing",
    "refit_span",
    "use_refit_timing",
]
