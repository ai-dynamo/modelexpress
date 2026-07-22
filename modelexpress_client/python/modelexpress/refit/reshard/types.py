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
    isn't valid (dtype mismatch); the affected tensor falls back to a full
    (non-sliced) pull rather than being captured or copied wrong."""


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
    """Output of a bake: the recorded copies plus what fell back.

    ``unsupported`` = source names whose loader used an unsupported op (full-pull
    those). ``unattributed`` = copy_ calls fired with no active loader stamp (the
    destination param can't be attributed -> also full-pull)."""

    copies: list = field(default_factory=list)
    unsupported: list = field(default_factory=list)
    unattributed: int = 0
