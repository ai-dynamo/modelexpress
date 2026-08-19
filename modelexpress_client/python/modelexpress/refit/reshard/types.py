# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Shared, torch-free data types for the reshard pipeline.

Kept separate from ``geometry.py`` (which needs torch for ``LazyWeight``) so the
slice-intersection arithmetic depends only on these plain records and stays
unit-testable off-GPU."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# One link in a recorded op-chain: (op_name, positional args, frozen kwargs).
OpSpec = tuple
# A recorded op-chain: the ordered view/slice ops a loader applied to a source.
OpChain = tuple


class UnsupportedReshard(NotImplementedError):
    """A loader used an op we can't express as a slice / box, or a byte copy
    isn't valid (dtype mismatch). The receiver fails the update rather than
    applying an incomplete model version."""


class IncompleteRefit(RuntimeError):
    """A refit was planned and executed successfully, and still did not write
    enough of the engine's parameter bytes.

    Deliberately not an ``UnsupportedReshard``. That one means the geometry could
    not be expressed, and ``transfer_plan`` catches it per source to drop that
    source and continue; a refit that quietly skipped part of the model must never
    be swallowed by a handler written for the other meaning. Both derive from
    ``RuntimeError``, so a caller that catches only that is unaffected."""


def summarize_unsupported(
    reasons: dict, limit: int | None = 3
) -> list[tuple[str, int]]:
    """Group per-source capture failures by cause, most frequent first.

    Every message embeds the offending source's name and op-chain, so thousands
    of sources failing for one shared reason produce thousands of textually
    distinct strings. Cutting each message at its source-specific tail collapses
    them, which is what makes "every expert in the model" legible as one cause
    instead of 18432 unique ones.
    """
    counts: dict[str, int] = {}
    for message in reasons.values():
        cause = str(message).split(" on lazy ", 1)[0].strip()
        counts[cause] = counts.get(cause, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:limit] if limit is not None else ranked


@dataclass
class RecordedCopy:
    """One recorded scatter: read ``src_name`` sliced by ``op_chain`` and write it
    into ``param_name`` at ``as_strided(dest_shape, dest_stride, dest_offset)``.
    Offsets/shapes/strides are read off the (meta) destination view, so no real
    storage is needed. ``dest_dtype`` vs the source dtype decides raw-copy vs
    convert-via-staging downstream."""

    src_name: str
    op_chain: OpChain
    param_name: str
    dest_offset: int
    dest_shape: tuple
    dest_stride: tuple
    dest_dtype: Any


@dataclass
class CaptureResult:
    """Output of a bake: the recorded copies plus what could not be attributed.

    ``unsupported`` = source names whose loader used an unsupported op.
    ``unsupported_reasons`` = that source name -> the op that defeated capture.
    Without it a rejected refit reports only how many sources failed, which is
    not enough to tell an unexpressible fused layout apart from a loader that
    merely touched one op outside the allowlist.
    ``unattributed`` = copy_ calls fired with no active loader stamp. Either
    condition makes the update fail closed in the current receiver; there is no
    fallback path that serves those tensors by another route."""

    copies: list = field(default_factory=list)
    unsupported: list = field(default_factory=list)
    unattributed: int = 0
    unsupported_reasons: dict = field(default_factory=dict)
