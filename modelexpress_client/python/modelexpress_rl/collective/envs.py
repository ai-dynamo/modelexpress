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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    MX_NCCL_REFIT_NUM_STREAMS: int
    MX_NCCL_REFIT_GROUP_TIMEOUT_S: float
    MX_NCCL_REFIT_POLL_INTERVAL_S: float
    MX_NCCL_REFIT_COMM_INIT_TIMEOUT_S: float
    MX_NCCL_REFIT_TRANSFER_TIMEOUT_S: float
    MX_NCCL_REFIT_MISC_CHUNK_BYTES: int
    MX_NCCL_REFIT_REGISTRATION_TTL_S: int


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


environment_variables: dict[str, Callable[[], Any]] = {
    "MX_NCCL_REFIT_NUM_STREAMS": lambda: _int("MX_NCCL_REFIT_NUM_STREAMS", 2),
    "MX_NCCL_REFIT_GROUP_TIMEOUT_S": lambda: _float("MX_NCCL_REFIT_GROUP_TIMEOUT_S", 600.0),
    "MX_NCCL_REFIT_POLL_INTERVAL_S": lambda: _float("MX_NCCL_REFIT_POLL_INTERVAL_S", 0.25),
    "MX_NCCL_REFIT_COMM_INIT_TIMEOUT_S": lambda: _float(
        "MX_NCCL_REFIT_COMM_INIT_TIMEOUT_S", 300.0
    ),
    "MX_NCCL_REFIT_TRANSFER_TIMEOUT_S": lambda: _float(
        "MX_NCCL_REFIT_TRANSFER_TIMEOUT_S", 600.0
    ),
    "MX_NCCL_REFIT_MISC_CHUNK_BYTES": lambda: _int(
        "MX_NCCL_REFIT_MISC_CHUNK_BYTES", 268435456
    ),
    "MX_NCCL_REFIT_REGISTRATION_TTL_S": lambda: _int(
        "MX_NCCL_REFIT_REGISTRATION_TTL_S",
        _int("MX_HEARTBEAT_INTERVAL_SECS", 30) * 3,
    ),
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(environment_variables)
