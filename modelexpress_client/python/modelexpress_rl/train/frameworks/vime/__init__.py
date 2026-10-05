# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vime trainer integration for ModelExpress RL."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .modelexpress import UpdateWeightFromModelExpress

__all__ = ["UpdateWeightFromModelExpress"]


def __getattr__(name: str) -> Any:
    if name == "UpdateWeightFromModelExpress":
        from .modelexpress import UpdateWeightFromModelExpress

        return UpdateWeightFromModelExpress
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
