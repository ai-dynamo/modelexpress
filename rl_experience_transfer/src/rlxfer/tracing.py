# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free W3C trace-context propagation helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace

from .model import ExperienceBatch

_TRACE_NAMESPACE = "w3c.trace_context"
_TRACEPARENT = re.compile(
    r"^(?P<version>[0-9a-f]{2})-(?P<trace_id>[0-9a-f]{32})-"
    r"(?P<parent_id>[0-9a-f]{16})-(?P<flags>[0-9a-f]{2})$"
)
_TRACESTATE_KEY = re.compile(
    r"^(?:[a-z][a-z0-9_*/-]{0,255}|[a-z0-9][a-z0-9_*/-]{0,240}"
    r"@[a-z][a-z0-9_*/-]{0,13})$"
)


@dataclass(frozen=True, slots=True)
class TraceContext:
    """Validated W3C traceparent and optional tracestate values."""

    traceparent: str
    tracestate: str | None = None

    def __post_init__(self) -> None:
        match = (
            _TRACEPARENT.fullmatch(self.traceparent) if isinstance(self.traceparent, str) else None
        )
        if (
            match is None
            or match["version"] == "ff"
            or int(match["trace_id"], 16) == 0
            or int(match["parent_id"], 16) == 0
        ):
            raise ValueError("traceparent is not a valid W3C trace context")
        if self.tracestate is not None and not _valid_tracestate(self.tracestate):
            raise ValueError("tracestate is not a valid W3C trace state")


def with_trace_context(batch: ExperienceBatch, context: TraceContext) -> ExperienceBatch:
    """Return a shallow batch copy carrying trace context in a safe extension."""

    extensions = dict(batch.extensions)
    extensions[_TRACE_NAMESPACE] = {
        key: value
        for key, value in (
            ("traceparent", context.traceparent),
            ("tracestate", context.tracestate),
        )
        if value is not None
    }
    return replace(batch, extensions=extensions)


def trace_context_from(batch: ExperienceBatch) -> TraceContext | None:
    """Read and validate trace context from a canonical batch."""

    value = batch.extensions.get(_TRACE_NAMESPACE)
    if value is None:
        return None
    if not isinstance(value, Mapping) or not set(value).issubset({"traceparent", "tracestate"}):
        raise ValueError("trace-context extension is malformed")
    traceparent = value.get("traceparent")
    tracestate = value.get("tracestate")
    if not isinstance(traceparent, str) or (
        tracestate is not None and not isinstance(tracestate, str)
    ):
        raise ValueError("trace-context extension is malformed")
    return TraceContext(traceparent, tracestate)


def _valid_tracestate(value: object) -> bool:
    if not isinstance(value, str) or not value or len(value) > 512 or not value.isascii():
        return False
    members = value.split(",")
    if len(members) > 32:
        return False
    keys: set[str] = set()
    for member in members:
        key, separator, state = member.partition("=")
        if (
            not separator
            or key in keys
            or _TRACESTATE_KEY.fullmatch(key) is None
            or not 0 < len(state) <= 256
            or state.endswith(" ")
            or any(character in ",=" or not 0x20 <= ord(character) <= 0x7E for character in state)
        ):
            return False
        keys.add(key)
    return True
