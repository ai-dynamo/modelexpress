# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generator-engine boundary for ModelExpress RL refit installation."""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from ..train.manifest import ShardChecksumKey


class GeneratorEngineContext(ABC):
    """Typed rank-local inputs used to construct one engine adapter."""


@dataclass(frozen=True)
class TrainerSourceShard:
    """Immutable manifest and worker selection for one trainer slot."""

    source_slot_id: str
    worker_id: str
    stable_metadata_digest: str | None
    metadata_endpoint: str
    metadata: bytes | None
    checksum_keys: Mapping[tuple[str, int], ShardChecksumKey] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.checksum_keys, MappingProxyType):
            object.__setattr__(self, "checksum_keys", MappingProxyType(dict(self.checksum_keys)))


__all__ = ["GeneratorEngineContext", "TrainerSourceShard"]
