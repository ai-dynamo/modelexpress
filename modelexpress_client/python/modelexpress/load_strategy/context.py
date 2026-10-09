# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared context objects for ModelExpress load strategies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar

import torch
from torch import nn

from .. import p2p_pb2
from ..accelerators import AcceleratorBackend, CudaAcceleratorBackend
from ..client import MxClientBase

if TYPE_CHECKING:
    from sglang.srt.configs.load_config import LoadConfig as SglangLoadConfig
    from sglang.srt.configs.model_config import ModelConfig as SglangModelConfig
    from vllm.config import ModelConfig as VllmModelConfig
    from vllm.config.load import LoadConfig as VllmLoadConfig

    from ..adapter import EngineAdapter
    from ..nixl_transfer import NixlTransferManager
    from ..vmm import VmmArena

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
    # Unique process rank in the engine's distributed world.
    global_rank: int
    # Model-shard key used to match compatible MX source and target workers.
    worker_rank: int
    # Process rank within its node; local rank 0 coordinates node-local state.
    local_rank: int
    # Local accelerator ordinal; topology must not be inferred from this value.
    device_id: int
    identity: p2p_pb2.SourceIdentity
    mx_client: MxClientBase
    worker_id: str
    mx_server_url: str | None = None
    node_rank: int = 0
    # Head address from the engine's own distributed config (for vLLM,
    # parallel_config.master_addr). Non-head nodes rewrite a pod-local
    # readiness URL onto this host, since loopback serves nothing there.
    head_addr: str | None = None
    adapter: EngineAdapter | None = None
    # Which engine's loader built this context. The strategy chain is otherwise
    # engine-agnostic, and its metrics have to carry the same `engine` label the
    # load and phase timings do or they cannot be read against each other. The
    # default is what a bare test fixture gets; the three real builders all set
    # it explicitly.
    engine: str = "other"
    accelerator_backend: AcceleratorBackend = field(
        default_factory=CudaAcceleratorBackend
    )
    # False keeps a secondary in-process load (the MTP drafter) out of P2P.
    p2p_enabled: bool = True
    # RL cold start snapshots this once after distributed rank agreement.
    desired_version_uid: str | None = None
    # Optional engine-level gate checked before weight metadata is advertised.
    source_ready_fn: Callable[[], bool] | None = None
    nixl_manager: NixlTransferManager | None = None
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    # When MX_VMM_ARENA=1, maybe_enter_vmm_arena populates this with the
    # active VmmArena. Strategies that register tensors with NIXL can use
    # the arena's (base, used_bytes) range as a single registration via
    # cuMemGetHandleForAddressRange + ibv_reg_dmabuf_mr, collapsing
    # O(plugin_calls) MRs to 1.
    vmm_arena: VmmArena | None = None
