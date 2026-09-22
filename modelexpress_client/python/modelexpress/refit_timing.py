# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for :mod:`modelexpress.refit.timing`.

New code should import timing primitives from ``modelexpress.refit`` or
``modelexpress.refit.timing``.
"""

from .refit.timing import (
    MX_REFIT_TIMING_PREFIX,
    REFIT_TIMING_STAGES,
    RefitTimingRecorder,
    add_refit_bytes,
    current_refit_timing,
    refit_span,
    use_refit_timing,
)

__all__ = [
    "MX_REFIT_TIMING_PREFIX",
    "REFIT_TIMING_STAGES",
    "RefitTimingRecorder",
    "add_refit_bytes",
    "current_refit_timing",
    "refit_span",
    "use_refit_timing",
]
