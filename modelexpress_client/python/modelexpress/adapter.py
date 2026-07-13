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
    """Expected strategy miss that lets the chain try the next strategy.

    Set mutated=True when the current strategy may have changed model weights
    or model structure before failing. The chain will run rollback() for
    strategy-owned cleanup, then ask the adapter to re-initialize the model.
    """

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
    """Engine-specific boundary used by shared loading strategies.

    Load strategies own ModelExpress policy: fallback order, metadata
    publishing, RDMA transfer, and retry decisions. Engine adapters own the
    operations that depend on a framework's model object, device mapping, and
    post-load processing rules.

    Methods decorated with @gated_capability are optional capabilities. A
    strategy lists the exact adapter methods it needs in its `requires` tuple;
    the strategy is eligible only when the concrete engine overrides those
    methods. Plain lifecycle hooks below are no-op by default because they are
    safe extension points around an already-selected strategy.
    """

    @gated_capability
    def build_identity(self) -> p2p_pb2.SourceIdentity:
        """Return the stable identity used to match compatible source workers."""
        ...

    @gated_capability
    def get_worker_rank(self) -> int:
        """Return the model-shard key used to pair source and target workers.

        Data-parallel replicas should normally return the same key because
        their weights are interchangeable. Tensor, pipeline, and expert
        parallel shards should return distinct keys when they own distinct
        model weights.
        """
        ...

    @gated_capability
    def get_global_rank(self) -> int:
        """Return the engine's global distributed rank for logging and metadata."""
        ...

    @gated_capability
    def get_device_id(self) -> int:
        """Return the local CUDA device id owned by this adapter instance."""
        ...

    @gated_capability
    def discover_tensors(self, result: LoadResult) -> dict[str, torch.Tensor]:
        """Return publishable tensors from the loaded engine model.

        Strategies call this after weights are ready. Implementations may run
        engine-specific tensor adoption or normalization before collecting the
        tensors, but should not change model weights.
        """
        ...

    @gated_capability
    def apply_weight_iter(
        self,
        result: LoadResult,
        weights_iter: Iterator[tuple[str, torch.Tensor]],
    ) -> LoadResult:
        """Apply a stream of named tensors to the engine model.

        This hook may mutate model weights. If a strategy catches an exception
        from this method, it should raise StrategyFailed(mutated=True) so the
        chain reinitializes the model before trying another strategy.
        """
        ...

    @gated_capability
    def build_model_streamer_weight_iter(
        self,
        model_uri: str,
        model: torch.nn.Module | None = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Return the engine-native ModelStreamer weight iterator.

        The shared strategy owns fallback and registration policy. Engines own
        the concrete ModelStreamer integration because native loader APIs and
        distributed-device selection are framework-specific. Some engine-native
        loaders need the initialized model to discover secondary weight sources.
        """
        ...

    @gated_capability
    def load_via_native(self, result: LoadResult) -> LoadResult:
        """Load the model using the engine's native disk/checkpoint loader."""
        ...

    @gated_capability
    def reinit_for_retry(self, result: LoadResult) -> LoadResult:
        """Replace a possibly-mutated model with a fresh engine model instance."""
        ...

    def get_unique_id(self) -> str:
        """Return a best-effort unique id for engines without custom identity."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return f"rank-{torch.distributed.get_rank()}"
        return f"pid-{os.getpid()}"

    def get_target_device(self) -> torch.device:
        """Return the torch device where target model state should live."""
        return torch.device(f"cuda:{self.get_device_id()}")

    def is_cuda_alike(self) -> bool:
        """Return whether this engine is running on a CUDA-like platform."""
        return False

    def prepare_rdma_target(self, result: LoadResult) -> LoadResult:
        """Prepare target-side model storage before receiving RDMA weights."""
        return result

    def before_rdma_receive(self, result: LoadResult) -> LoadResult:
        """Run engine post-processing needed before RDMA writes into tensors."""
        return result

    def prepare_rdma_target_from_manifest(
        self,
        result: LoadResult,
        source_tensors,
    ) -> LoadResult:
        """Adjust target tensor layout based on the source manifest."""
        return result

    def after_rdma_receive(self, result: LoadResult) -> LoadResult:
        """Run engine post-processing after RDMA weights have been received."""
        return result

    def after_weight_iter_load(self, result: LoadResult) -> LoadResult:
        """Run engine post-processing after apply_weight_iter() succeeds."""
        return result

    def after_native_load(self, result: LoadResult) -> LoadResult:
        """Run engine post-processing after load_via_native() succeeds."""
        return result

    def release_failed_load(self, result: LoadResult) -> LoadResult:
        """Release engine-owned state before retrying after a failed strategy."""
        return result
