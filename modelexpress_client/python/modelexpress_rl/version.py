# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared weight-version references for ModelExpress RL clients."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WeightVersionRef:
    """Opaque reference to one global WeightVersion created by the orchestrator."""

    version_id: str

    def __post_init__(self) -> None:
        if not self.version_id.strip():
            raise ValueError("version.version_id is required")


__all__ = ["WeightVersionRef"]
