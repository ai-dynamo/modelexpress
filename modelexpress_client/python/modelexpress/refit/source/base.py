# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared invariants for bounded canonical tensor capture."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch

CanonicalTensor: TypeAlias = tuple[str, torch.Tensor]
CanonicalBucket: TypeAlias = tuple[CanonicalTensor, ...]
CanonicalBucketConsumer: TypeAlias = Callable[[CanonicalBucket], None]
CanonicalCapturePlan: TypeAlias = tuple[tuple[str, str, tuple[int, ...], bool], ...]


class CanonicalSourceError(RuntimeError):
    """Canonical capture failed consistently across the trainer ranks."""


class CollectiveDeadline:
    """Absolute capture deadline with an integration-owned collective abort hook."""

    def __init__(
        self,
        deadline_monotonic: float | None,
        abort_collectives: Callable[[], None] | None,
    ) -> None:
        distributed = _distributed_is_initialized()
        if distributed and (
            deadline_monotonic is None or not callable(abort_collectives)
        ):
            raise CanonicalSourceError(
                "distributed canonical capture requires an absolute collective "
                "deadline and abort_collectives callback"
            )
        if deadline_monotonic is not None and (
            isinstance(deadline_monotonic, bool)
            or not isinstance(deadline_monotonic, (int, float))
            or not math.isfinite(deadline_monotonic)
        ):
            raise ValueError("deadline_monotonic must be a finite absolute time")
        if abort_collectives is not None and not callable(abort_collectives):
            raise TypeError("abort_collectives must be callable")
        self._deadline = (
            float(deadline_monotonic) if deadline_monotonic is not None else None
        )
        self._abort = abort_collectives
        self._done = threading.Event()
        self._expired = threading.Event()
        self._abort_error: str | None = None
        self._thread: threading.Thread | None = None
        if self._deadline is not None:
            self._thread = threading.Thread(
                target=self._watch,
                name="mx-canonical-collective-deadline",
                daemon=True,
            )
            self._thread.start()

    def _watch(self) -> None:
        assert self._deadline is not None
        remaining = max(0.0, self._deadline - time.monotonic())
        if self._done.wait(remaining):
            return
        self._expired.set()
        if self._abort is not None:
            try:
                self._abort()
            except Exception as exc:
                self._abort_error = str(exc) or type(exc).__name__

    def check(self, boundary: str) -> None:
        if self._deadline is None:
            return
        if time.monotonic() >= self._deadline:
            self._expired.set()
        if self._expired.is_set():
            detail = ""
            if self._abort_error is not None:
                detail = f"; collective abort failed: {self._abort_error}"
            raise CanonicalSourceError(
                f"canonical source absolute collective deadline expired at "
                f"{boundary}{detail}"
            )

    def close(self) -> None:
        self._done.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()


@dataclass(frozen=True)
class CanonicalTensorSpec:
    """One authoritative tensor in the complete ordered canonical HF schema."""

    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", canonical_tensor_name(self.name))
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("canonical HF schema dtype must be a torch.dtype")
        try:
            shape = tuple(self.shape)
        except TypeError as exc:
            raise TypeError("canonical HF schema shape must be iterable") from exc
        if any(
            not isinstance(size, int) or isinstance(size, bool) or size < 0
            for size in shape
        ):
            raise ValueError(
                "canonical HF schema dimensions must be non-negative integers"
            )
        object.__setattr__(self, "shape", shape)

    @property
    def nbytes(self) -> int:
        return (
            torch.Size(self.shape).numel()
            * torch.empty((), dtype=self.dtype).element_size()
        )


def canonical_schema_plan(
    schema: Sequence[CanonicalTensorSpec] | None,
    bucket_bytes: int,
) -> tuple[CanonicalCapturePlan, tuple[tuple[str, int], ...]]:
    """Validate the complete external HF schema before any tensor materialization."""
    if schema is None or not schema:
        raise ValueError("an authoritative canonical HF schema is required")
    plan = []
    named_sizes = []
    names: set[str] = set()
    for spec in schema:
        if not isinstance(spec, CanonicalTensorSpec):
            raise TypeError("canonical HF schema entries must be CanonicalTensorSpec")
        if spec.name in names:
            raise ValueError(f"canonical HF schema duplicates tensor {spec.name!r}")
        if spec.nbytes > bucket_bytes:
            raise ValueError(
                f"canonical tensor {spec.name!r} size {spec.nbytes} exceeds "
                f"bucket_bytes={bucket_bytes}"
            )
        names.add(spec.name)
        plan.append((spec.name, str(spec.dtype), spec.shape, True))
        named_sizes.append((spec.name, spec.nbytes))
    return tuple(plan), tuple(named_sizes)


def canonical_tensor_name(name: str) -> str:
    """Remove wrapper-only prefixes without changing the HF tensor identity."""
    if not isinstance(name, str) or not name:
        raise ValueError("canonical tensor names must be non-empty strings")
    previous = None
    while name != previous:
        previous = name
        while name.startswith("module."):
            name = name[len("module.") :]
        while name.startswith("_orig_mod."):
            name = name[len("_orig_mod.") :]
        name = name.replace(".base_layer.", ".")
    if not name:
        raise ValueError(
            "canonical tensor names must remain non-empty after normalization"
        )
    return name


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _distributed_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def synchronize_errors(
    local_error: str | None,
    group: Any = None,
    *,
    content_digest: str | None = None,
) -> None:
    """Agree on errors and canonical content after every rank drains its source."""
    if not _distributed_is_initialized():
        if local_error is not None:
            raise CanonicalSourceError(local_error)
        return

    contribution = (local_error, content_digest)
    gathered: list[tuple[str | None, str | None] | None] = [
        None
    ] * torch.distributed.get_world_size(group)
    try:
        torch.distributed.all_gather_object(gathered, contribution, group=group)
    except Exception as exc:
        raise CanonicalSourceError(
            f"canonical source error synchronization failed: {exc}"
        ) from exc

    failures = [
        f"rank {rank}: {item[0]}"
        for rank, item in enumerate(gathered)
        if item is not None and item[0]
    ]
    if failures:
        raise CanonicalSourceError("; ".join(failures))
    if any(item is None or item[1] != content_digest for item in gathered):
        raise CanonicalSourceError("canonical content differs across trainer ranks")


def synchronize_preflight(
    local_error: str | None,
    plan: CanonicalCapturePlan | None,
    group: Any = None,
    *,
    representation: object = None,
) -> None:
    """Agree on preflight success and collective schedule in exactly one gather."""
    if not _distributed_is_initialized():
        if local_error is not None:
            raise CanonicalSourceError(local_error)
        return

    contribution = (local_error, representation, plan)
    gathered: list[tuple[str | None, object, CanonicalCapturePlan | None] | None] = [
        None
    ] * torch.distributed.get_world_size(group)
    try:
        torch.distributed.all_gather_object(gathered, contribution, group=group)
    except Exception as exc:
        raise CanonicalSourceError(
            f"canonical source preflight synchronization failed: {exc}"
        ) from exc

    failures = [
        f"rank {rank}: {item[0]}"
        for rank, item in enumerate(gathered)
        if item is not None and item[0]
    ]
    if failures:
        raise CanonicalSourceError("; ".join(failures))
    if any(item is None or item[1:] != (representation, plan) for item in gathered):
        raise CanonicalSourceError(
            "canonical capture plan differs across trainer ranks"
        )


def finish_collectives(group: Any = None) -> None:
    """Do not return capture success until the final trainer barrier succeeds."""
    if not _distributed_is_initialized():
        return
    try:
        torch.distributed.barrier(group=group)
    except Exception as exc:
        raise CanonicalSourceError(
            f"canonical source final barrier failed: {exc}"
        ) from exc


class BoundedBucketBuilder:
    """Validate a complete ordered stream while bounding rank-0 CPU storage."""

    def __init__(
        self,
        *,
        bucket_bytes: int,
        emit: bool,
        consume_bucket: CanonicalBucketConsumer,
        capture_units: Sequence[Sequence[str]] | None = None,
        capture_unit_sizes: Sequence[int] | None = None,
    ) -> None:
        self._bucket_bytes = bucket_bytes
        self._emit = emit
        self._consume_bucket = consume_bucket
        self._bucket: list[CanonicalTensor] = []
        self._bucket_size = 0
        self._seen: set[str] = set()
        self._capture_units = (
            tuple(tuple(unit) for unit in capture_units)
            if capture_units is not None
            else None
        )
        self._capture_unit_sizes = (
            tuple(capture_unit_sizes) if capture_unit_sizes is not None else None
        )
        if (self._capture_units is None) != (self._capture_unit_sizes is None):
            raise ValueError("capture units and sizes must be configured together")
        if self._capture_units is not None:
            if len(self._capture_units) != len(self._capture_unit_sizes):
                raise ValueError("capture unit sizes do not match capture units")
            if any(not unit for unit in self._capture_units):
                raise ValueError("capture units must be non-empty")
            if any(
                not isinstance(size, int) or isinstance(size, bool) or size < 0
                for size in self._capture_unit_sizes
            ):
                raise ValueError("capture unit sizes must be non-negative integers")
            self._expected_names = tuple(
                name for unit in self._capture_units for name in unit
            )
            self._unit_starts: dict[int, tuple[int, int]] = {}
            offset = 0
            for unit, size in zip(
                self._capture_units, self._capture_unit_sizes, strict=True
            ):
                self._unit_starts[offset] = (len(unit), size)
                offset += len(unit)
        else:
            self._expected_names = ()
            self._unit_starts = {}
        self._next_expected = 0
        self._content = hashlib.sha256(b"mx.canonical.source.content.v1\0")
        self._tensor_digests: dict[str, str] = {}
        self.error: str | None = None

    def record(self, name: str, tensor: object) -> None:
        """Record one stream element, retaining no tensors after the callback."""
        try:
            canonical_name = canonical_tensor_name(name)
            if canonical_name in self._seen:
                raise ValueError(
                    f"canonical tensor {canonical_name!r} appeared more than once"
                )
            self._seen.add(canonical_name)
            if self._capture_units is not None:
                if self._next_expected >= len(self._expected_names):
                    raise ValueError(f"unexpected canonical tensor {canonical_name!r}")
                expected_name = self._expected_names[self._next_expected]
                if canonical_name != expected_name:
                    raise ValueError(
                        f"canonical tensor order changed; expected {expected_name!r}, "
                        f"got {canonical_name!r}"
                    )
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"canonical tensor {canonical_name!r} is not a torch.Tensor"
                )
            size = tensor_nbytes(tensor)
            if size > self._bucket_bytes:
                raise ValueError(
                    f"canonical tensor {canonical_name!r} size {size} exceeds "
                    f"bucket_bytes={self._bucket_bytes}"
                )
            unit = self._unit_starts.get(self._next_expected)
            if (
                unit is not None
                and self._bucket
                and self._bucket_size + unit[1] > self._bucket_bytes
            ):
                self.flush()
            cpu_tensor = tensor.detach().to(device="cpu").contiguous()
            self._feed_content(canonical_name.encode("utf-8"))
            self._feed_content(str(cpu_tensor.dtype).encode("ascii"))
            self._feed_content(
                json.dumps(tuple(cpu_tensor.shape), separators=(",", ":")).encode(
                    "ascii"
                )
            )
            data = cpu_tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            self._feed_content(data)
            self._tensor_digests[canonical_name] = (
                f"sha256:{hashlib.sha256(data).hexdigest()}"
            )
            if self._bucket and self._bucket_size + size > self._bucket_bytes:
                self.flush()
            if self._emit and self.error is None:
                self._bucket.append((canonical_name, cpu_tensor))
                self._bucket_size += size
            self._next_expected += 1
        except Exception as exc:
            self._record_error(exc)

    def flush(self) -> None:
        bucket = tuple(self._bucket)
        self._bucket.clear()
        self._bucket_size = 0
        if not bucket or self.error is not None:
            return
        try:
            self._consume_bucket(bucket)
        except Exception as exc:
            self._record_error(exc)

    def flush_completed_units(self) -> None:
        """Expose consumer failures before the next source collective is entered."""
        if self._capture_units is None:
            return
        if self._next_expected == len(self._expected_names) or (
            self._next_expected in self._unit_starts
        ):
            self.flush()

    def finish(self) -> None:
        if self._capture_units is not None and self._next_expected != len(
            self._expected_names
        ):
            self._record_error(
                ValueError("canonical source ended before complete planned coverage")
            )
        self.flush()

    @property
    def content_digest(self) -> str:
        return f"sha256:{self._content.hexdigest()}"

    @property
    def tensor_digests(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._tensor_digests.items())

    def _feed_content(self, value: bytes) -> None:
        self._content.update(struct.pack(">Q", len(value)))
        self._content.update(value)

    def _record_error(self, error: Exception) -> None:
        if self.error is None:
            self.error = str(error) or type(error).__name__
        self._bucket.clear()
        self._bucket_size = 0


def configured_rank(rank: Callable[[], int] | None, group: Any = None) -> int:
    if rank is not None:
        value = rank()
    elif _distributed_is_initialized():
        value = torch.distributed.get_rank(group)
    else:
        value = 0
    if not isinstance(value, int) or value < 0:
        raise CanonicalSourceError(f"invalid trainer rank: {value!r}")
    return value


def first_error(errors: Sequence[BaseException]) -> str | None:
    if not errors:
        return None
    error = errors[0]
    return str(error) or type(error).__name__
