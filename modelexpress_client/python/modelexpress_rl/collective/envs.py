# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment policy for the NCCL M2N collective refit path.

Every timeout here is a deadline rather than a hint. Group formation being
bounded is not enough on its own: READY only means the group formed, and the
communicator setup and the transfer that follow can each block indefinitely on
their own. A path whose failure story is "turn hangs into attributable
failures" has to bound what happens after READY too.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    MX_NCCL_REFIT_NUM_STREAMS: int
    MX_NCCL_REFIT_GROUP_TIMEOUT_S: float
    MX_NCCL_REFIT_POLL_INTERVAL_S: float
    MX_NCCL_REFIT_COMM_INIT_TIMEOUT_S: float
    MX_NCCL_REFIT_TRANSFER_TIMEOUT_S: float
    MX_NCCL_REFIT_REGISTRATION_TTL_S: int
    MX_NCCL_REFIT_STACK_BYTES: int


def _int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except ValueError as error:
        raise ValueError(f"invalid {name}: {os.environ.get(name)!r}") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except ValueError as error:
        raise ValueError(f"invalid {name}: {os.environ.get(name)!r}") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _nonnegative_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except ValueError as error:
        raise ValueError(f"invalid {name}: {os.environ.get(name)!r}") from error
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


environment_variables: dict[str, Callable[[], Any]] = {
    "MX_NCCL_REFIT_NUM_STREAMS": lambda: _int("MX_NCCL_REFIT_NUM_STREAMS", 2),
    "MX_NCCL_REFIT_GROUP_TIMEOUT_S": lambda: _float(
        "MX_NCCL_REFIT_GROUP_TIMEOUT_S", 600.0
    ),
    "MX_NCCL_REFIT_POLL_INTERVAL_S": lambda: _float(
        "MX_NCCL_REFIT_POLL_INTERVAL_S", 0.25
    ),
    "MX_NCCL_REFIT_COMM_INIT_TIMEOUT_S": lambda: _float(
        "MX_NCCL_REFIT_COMM_INIT_TIMEOUT_S", 300.0
    ),
    "MX_NCCL_REFIT_TRANSFER_TIMEOUT_S": lambda: _float(
        "MX_NCCL_REFIT_TRANSFER_TIMEOUT_S", 600.0
    ),
    "MX_NCCL_REFIT_REGISTRATION_TTL_S": lambda: _int(
        "MX_NCCL_REFIT_REGISTRATION_TTL_S",
        _int("MX_HEARTBEAT_INTERVAL_SECS", 30) * 3,
    ),
    # Equal-geometry stacking budget, in bytes of one stacked call's area.
    # Zero (the default) keeps one reshard call per tensor. It is read by the
    # trainer that builds the plan and travels in the plan, never read
    # independently by the receivers.
    "MX_NCCL_REFIT_STACK_BYTES": lambda: _nonnegative_int(
        "MX_NCCL_REFIT_STACK_BYTES", 0
    ),
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(environment_variables)


# Native PACK staging bucket sizing (NCCL_RESHARD_PACK_BUFFSIZES), mirrored
# read-only from the nccl-extensions M2N library (m2n_config.cc) so the budget
# of one reshard call (a stacked call included) is checked against the pool
# the transfers actually run against. The native value is
# "size[:slots],size[:slots],...": sizes are decimal bytes with an optional
# binary K or M suffix (there is no G), a bare size gets one slot, at most 8
# buckets sum to at most 64 slots, duplicate sizes are rejected, and the
# largest bucket must hold at least 2 KiB. Unset -- or anything the native
# library cannot parse, which it ignores with a warning -- leaves the built-in
# one-bucket 2-GiB profile in effect, so both cases report the default here
# rather than raising.
_PACK_STAGING_DEFAULT_BUCKET_BYTES = 2048 * 1024 * 1024
_PACK_STAGING_MAX_BUCKETS = 8
_PACK_STAGING_MAX_TOTAL_SLOTS = 64  # MAX_SPLIT_CONCURRENCY in the native pool
_PACK_STAGING_MIN_BUCKET_BYTES = 2048
_UINT64_MAX = (1 << 64) - 1  # strtoull ERANGE bound of the native size parse
_INT64_MAX = (1 << 63) - 1  # strtol ERANGE bound of the native slot parse

# ASCII-only classes mirror the native isM2nEnvDigit/isM2nEnvSpace exactly:
# \d and \s on str would also admit Unicode digits and whitespace, which the
# native parser rejects (and Python's int() would happily convert).
_PACK_BUCKET_ITEM = re.compile(
    r"[ \t\n\r\f\v]*([0-9]+)([kKmM]?)[ \t\n\r\f\v]*(?::[ \t\n\r\f\v]*([0-9]+))?[ \t\n\r\f\v]*"
)


def _parse_pack_buffsizes(value: str) -> tuple[int, tuple[int, ...]] | None:
    """``(total slots, bucket sizes in bytes)`` of a PACK_BUFFSIZES value.

    None when the value is invalid. The sizes keep the listed order.
    """
    if not value:
        return None
    sizes: list[int] = []
    total = 0
    for item in value.split(","):
        if len(sizes) >= _PACK_STAGING_MAX_BUCKETS:
            return None
        match = _PACK_BUCKET_ITEM.fullmatch(item)
        if match is None:
            return None
        size = int(match.group(1))
        if size == 0 or size > _UINT64_MAX:
            return None
        multiplier = {"": 1, "k": 1024, "m": 1024 * 1024}[match.group(2).lower()]
        if size > _UINT64_MAX // multiplier:
            return None
        size *= multiplier
        slots = 1
        if match.group(3) is not None:
            slots = int(match.group(3))
            if slots > _INT64_MAX:
                return None
        if slots <= 0 or slots > _PACK_STAGING_MAX_TOTAL_SLOTS:
            return None
        total += slots
        if total > _PACK_STAGING_MAX_TOTAL_SLOTS:
            return None
        sizes.append(size)
    if len(set(sizes)) != len(sizes):
        return None
    if max(sizes) < _PACK_STAGING_MIN_BUCKET_BYTES:
        return None
    return total, tuple(sizes)


def pack_staging_largest_bucket_bytes() -> int:
    """Bytes of the largest PACK staging bucket this process will run with.

    One native PACK request must fit the largest bucket, so this bounds the
    area of a single reshard call (a stacked call included). Unset, or a value
    the native library cannot parse and therefore ignores, means the built-in
    profile, whose one bucket is 2 GiB.
    """
    raw = os.environ.get("NCCL_RESHARD_PACK_BUFFSIZES")
    parsed = None if raw is None else _parse_pack_buffsizes(raw)
    if parsed is None:
        return _PACK_STAGING_DEFAULT_BUCKET_BYTES
    return max(parsed[1])
