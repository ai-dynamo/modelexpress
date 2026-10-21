# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Throughput-floor diagnostics for staged RL weight transfers."""

from __future__ import annotations

import json
import logging
import math

from modelexpress_rl import envs

# Grepped by the CI perf gate, so it is named once rather than spelled at each
# call site.
SLOW_THROUGHPUT_MARKER = "MX_REFIT_SLOW_THROUGHPUT"
SLOW_THROUGHPUT_SCHEMA = "refit-slow-throughput-v1"


def min_gbps(log: logging.Logger) -> float:
    """Per-rank floor in Gbps; 0 disables the check.

    Read at call time so a harness can set it per run. Off unless configured
    because only the operator knows what their fabric should deliver.

    A non-finite floor disables the check rather than being compared against.
    ``float("nan")`` parses happily and every comparison against NaN is False, so
    a fat-fingered value would pass the "is it configured" test and then fail the
    "is the rate acceptable" test, warning on every single refit and turning the
    guard into noise. Infinity parses too and would condemn every refit outright.

    A negative floor disables it too, but says so. ``warn_if_below_floor`` treats
    anything at or below zero as off, so a stray minus sign would otherwise take
    a configured gate out of service in silence - and the caller who wrote -50
    believes a floor is in force. Only ``0`` is documented as the off switch, so
    every other disabling value has to announce itself.
    """
    floor = envs.MX_RESHARD_MIN_GBPS
    if not math.isfinite(floor):
        log.warning(
            "MX_RESHARD_MIN_GBPS=%r is not a finite number; the throughput floor "
            "is disabled for this run",
            floor,
        )
        return 0.0
    if floor < 0.0:
        log.warning(
            "MX_RESHARD_MIN_GBPS=%r is negative; the throughput floor is disabled "
            "for this run (set 0 to disable it deliberately)",
            floor,
        )
        return 0.0
    return floor


def warn_if_below_floor(
    *,
    wire_bytes: int,
    wire_seconds: float,
    log: logging.Logger,
    context: dict | None = None,
) -> None:
    """Report a wire rate far below what the fabric should deliver.

    Warn rather than abort a correct refit. The structured record lets CI gate
    throughput regressions without interrupting a running training job.

    ``context`` is merged into the record so each receiver can name whoever saw
    it. The incident had one fast rank and three slow ones in the same run, so a
    record that cannot be attributed is a record that cannot describe it.

    Logs through the caller's logger so the report surfaces under the module that
    owns the transfer, next to that receiver's own cost record.
    """
    floor = min_gbps(log)
    if floor <= 0 or wire_bytes <= 0 or wire_seconds <= 0:
        return
    implied_gbps = wire_bytes * 8 / wire_seconds / 1e9
    if implied_gbps >= floor:
        return
    record = {
        "schema": SLOW_THROUGHPUT_SCHEMA,
        **(context or {}),
        "wire_bytes": wire_bytes,
        "wire_s": round(wire_seconds, 6),
        "implied_gbps": round(implied_gbps, 1),
        "floor_gbps": floor,
        # How far under, because "slightly below a conservative floor" and "an
        # order of magnitude below" are different incidents and a gate may
        # reasonably treat them differently.
        "shortfall_x": round(floor / implied_gbps, 2) if implied_gbps > 0 else None,
    }
    log.warning("%s %s", SLOW_THROUGHPUT_MARKER, json.dumps(record, sort_keys=True))
