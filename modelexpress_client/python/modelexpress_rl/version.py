# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared weight-version references for ModelExpress RL clients."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainerTensorsMetadata:
    """Compact reference to immutable worker-local logical tensor coverage."""

    logical_shard_id: str
    metadata_endpoint: str


@dataclass(frozen=True)
class TrainerMeshRef:
    """Opaque reference to one generation of a trainer mesh."""

    mesh_id: str
    generation: int

    def __post_init__(self) -> None:
        if not isinstance(self.mesh_id, str):
            raise TypeError("mesh_id must be a string")
        if not self.mesh_id.strip():
            raise ValueError("mesh_id is required")
        if type(self.generation) is not int or not 0 < self.generation < 2**64:
            raise ValueError("generation must be a positive uint64 integer")


@dataclass(frozen=True)
class WeightVersionRef:
    """Opaque reference to one global WeightVersion created by the orchestrator."""

    version_id: str

    def __post_init__(self) -> None:
        if not self.version_id.strip():
            raise ValueError("version.version_id is required")


__all__ = ["TrainerMeshRef", "TrainerTensorsMetadata", "WeightVersionRef"]
