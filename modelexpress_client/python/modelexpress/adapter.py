# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine adapter contract for ModelExpress loading strategies."""

from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING, Iterator

import torch

from . import p2p_pb2

if TYPE_CHECKING:
    from .load_strategy.context import LoadResult


class UnsupportedCapability(NotImplementedError):
    """Raised by default bodies for adapter capabilities."""


class StrategyFailed(RuntimeError):
    """A strategy failed but the chain may try the next strategy."""

    def __init__(self, message: str, *, mutated: bool = False):
        super().__init__(message)
        self.mutated = mutated


def gated_capability(method):
    """Create an optional adapter method that engines must override to support it.

    Strategies compare their required EngineAdapter methods by object identity.
    If an engine inherits this default method, the strategy is not eligible; if
    the engine overrides it, the strategy may call the engine implementation.
    Calling the inherited default still raises UnsupportedCapability.
    """

    @functools.wraps(method)
    def default(self, *args, **kwargs):
        raise UnsupportedCapability(method.__name__)

    default.__gated_capability__ = True
    return default


class EngineAdapter:
    """Optional-method integration contract implemented by each engine."""

    @gated_capability
    def build_identity(self) -> p2p_pb2.SourceIdentity:
        ...

    @gated_capability
    def get_worker_rank(self) -> int:
        ...

    @gated_capability
    def get_device_id(self) -> int:
        ...

    @gated_capability
    def discover_tensors(self, result: LoadResult) -> dict[str, torch.Tensor]:
        ...

    @gated_capability
    def apply_weight_iter(
        self,
        result: LoadResult,
        weights_iter: Iterator[tuple[str, torch.Tensor]],
    ) -> LoadResult:
        ...

    @gated_capability
    def load_via_native(self, result: LoadResult) -> LoadResult:
        ...

    @gated_capability
    def reinit_for_retry(self, result: LoadResult) -> LoadResult:
        ...

    def get_unique_id(self) -> str:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return f"rank-{torch.distributed.get_rank()}"
        return f"pid-{os.getpid()}"

    def get_target_device(self) -> torch.device:
        return torch.device(f"cuda:{self.get_device_id()}")

    def prepare_rdma_target(self, result: LoadResult) -> LoadResult:
        return result

    def before_rdma_receive(self, result: LoadResult) -> LoadResult:
        return result

    def after_rdma_receive(self, result: LoadResult) -> LoadResult:
        return result

    def after_weight_iter_load(self, result: LoadResult) -> LoadResult:
        return result

    def after_native_load(self, result: LoadResult) -> LoadResult:
        return result

    def unpublish(self, result: LoadResult) -> None:
        return None

    def cleanup(self, result: LoadResult) -> None:
        return None
