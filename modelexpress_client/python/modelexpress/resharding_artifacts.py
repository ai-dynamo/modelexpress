# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable JSON artifact helpers for resharding plans and simulators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .resharding_types import (
    CompetitiveSimulationResult,
    SegmentPlan,
    SimulationResult,
)


def segment_plans_to_json(plans: Sequence[SegmentPlan]) -> str:
    """Serialize segment plans as stable JSON."""

    return json.dumps(
        [plan.to_dict() for plan in plans],
        indent=2,
        sort_keys=True,
    )


def segment_plans_from_json(payload: str) -> list[SegmentPlan]:
    """Deserialize segment plans from JSON."""

    return [SegmentPlan.from_dict(item) for item in json.loads(payload)]


def write_json_artifact(
    data: (
        SegmentPlan
        | SimulationResult
        | CompetitiveSimulationResult
        | Sequence[SegmentPlan]
    ),
    path: str | Path,
) -> None:
    """Write a stable JSON artifact for planner or simulator output."""

    if isinstance(data, (SimulationResult, CompetitiveSimulationResult)):
        payload: Any = data.to_dict()
    elif isinstance(data, SegmentPlan):
        payload = data.to_dict()
    else:
        payload = [item.to_dict() for item in data]

    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
