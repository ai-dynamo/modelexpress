# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RL-specific deployment policy.

Model identity, server connectivity, worker endpoints, NIXL ports, and
heartbeat timing use the shared :mod:`modelexpress.envs` configuration.
Rank-local identity and endpoints are derived from the initialized engine.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    MX_TRAINER_ENGINE: str
    MX_TRAINER_STAGING_MODE: str
    MX_WEIGHT_PAYLOAD_FORMAT: str


environment_variables: dict[str, Callable[[], Any]] = {
    "MX_TRAINER_ENGINE": lambda: os.environ.get("MX_TRAINER_ENGINE", "MEGATRON")
    .strip()
    .upper(),
    "MX_TRAINER_STAGING_MODE": lambda: os.environ.get(
        "MX_TRAINER_STAGING_MODE", "IN_PLACE"
    )
    .strip()
    .upper(),
    "MX_WEIGHT_PAYLOAD_FORMAT": lambda: os.environ.get(
        "MX_WEIGHT_PAYLOAD_FORMAT", "FULL_TENSOR"
    )
    .strip()
    .upper(),
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(environment_variables)
