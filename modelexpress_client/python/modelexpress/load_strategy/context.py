# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared context objects for ModelExpress load strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import torch
import torch.nn as nn

from .. import p2p_pb2
from ..client import MxClientBase

if TYPE_CHECKING:
    from ..adapter import EngineAdapter
    from ..nixl_transfer import NixlTransferManager
    from vllm.config import ModelConfig, VllmConfig
    from vllm.config.load import LoadConfig


T = TypeVar("T")


@dataclass
class LoadResult(Generic[T]):
    """Stable envelope passed through strategies and adapter hooks."""

    value: T
    model: nn.Module | None = None
    publishable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def model_for_publish(self) -> nn.Module | None:
        return self.model if self.publishable else None


@dataclass
class LoadContext:
    """Shared state passed to all loading strategies."""

    vllm_config: VllmConfig
    model_config: ModelConfig
    load_config: LoadConfig
    target_device: torch.device
    global_rank: int
    device_id: int
    identity: p2p_pb2.SourceIdentity
    mx_client: MxClientBase
    worker_id: str
    adapter: EngineAdapter | None = None
    worker_rank: int | None = None
    nixl_manager: NixlTransferManager | None = None
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.worker_rank is None:
            self.worker_rank = self.global_rank
