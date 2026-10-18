# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generator-engine boundary for ModelExpress RL refit installation."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass


class GeneratorEngineContext(ABC):
    """Typed rank-local inputs used to construct one engine adapter."""


@dataclass(frozen=True)
class TrainerSourceShard:
    """Immutable manifest and worker selection for one trainer slot."""

    source_slot_id: str
    worker_id: str
    manifest_digest: str
    manifest_endpoint: str
    manifest: bytes
    structural_digest: str


__all__ = ["GeneratorEngineContext", "TrainerSourceShard"]
