# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared context objects for ModelExpress load strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar

import torch
import torch.nn as nn

from .. import p2p_pb2
from ..client import MxClientBase
from ..accelerators import AcceleratorBackend, CudaAcceleratorBackend

if TYPE_CHECKING:
    from ..adapter import EngineAdapter
    from ..nixl_transfer import NixlTransferManager
    from ..vmm import VmmArena
    from sglang.srt.configs.load_config import LoadConfig as SglangLoadConfig
    from sglang.srt.configs.model_config import ModelConfig as SglangModelConfig
    from vllm.config import ModelConfig as VllmModelConfig
    from vllm.config.load import LoadConfig as VllmLoadConfig

    EngineModelConfig: TypeAlias = VllmModelConfig | SglangModelConfig
    EngineLoadConfig: TypeAlias = VllmLoadConfig | SglangLoadConfig
else:
    EngineModelConfig = object
    EngineLoadConfig = object


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

    model_config: EngineModelConfig
    load_config: EngineLoadConfig
    target_device: torch.device
    global_rank: int
    worker_rank: int
    device_id: int
    identity: p2p_pb2.SourceIdentity
    mx_client: MxClientBase
    worker_id: str
    adapter: EngineAdapter | None = None
    accelerator_backend: AcceleratorBackend = field(default_factory=CudaAcceleratorBackend)
    nixl_manager: NixlTransferManager | None = None
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    # When MX_VMM_ARENA=1, maybe_enter_vmm_arena populates this with the
    # active VmmArena. Strategies that register tensors with NIXL can use
    # the arena's (base, used_bytes) range as a single registration via
    # cuMemGetHandleForAddressRange + ibv_reg_dmabuf_mr, collapsing
    # O(plugin_calls) MRs to 1.
    vmm_arena: VmmArena | None = None
