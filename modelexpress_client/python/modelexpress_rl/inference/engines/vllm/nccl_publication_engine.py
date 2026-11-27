# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM NCCL weight transfer that keeps each worker a ModelExpress P2P source.

The trainer still pushes weights over vLLM's native NCCL backend. Around each
update, this engine withdraws the worker's published tensors, drains peer
readers before NCCL mutates them, and republishes them afterwards, so a
restarted replica can copy the live weights from a healthy peer instead of
reloading them from storage.
"""

from __future__ import annotations

import logging
from typing import Protocol

import torch
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

from modelexpress.engines.vllm.loader import get_model_loader
from modelexpress.load_strategy import (
    drain_tensor_readers,
    publish_metadata,
    register_tensors,
    unpublish_metadata,
)
from modelexpress.metadata.worker_server import _tensor_read_lease_timeout_seconds
from modelexpress_rl import envs as rl_envs

logger = logging.getLogger(__name__)

_DRAIN_TIMEOUT_MARGIN_S = 30


class _RuntimePublication(Protocol):
    def unpublish_runtime_tensors(self, drain_timeout: float) -> None: ...

    def publish_runtime_tensors(self, version_id: str) -> None: ...


class _DummyBootPublication:
    """Publishes a dummy-loaded worker's tensors the way MxModelLoader would."""

    def __init__(self, ctx) -> None:
        self._ctx = ctx

    def unpublish_runtime_tensors(self, drain_timeout: float) -> None:
        drain_tensor_readers(self._ctx, timeout=drain_timeout)
        unpublish_metadata(self._ctx)

    def publish_runtime_tensors(self, version_id: str) -> None:
        self._ctx.identity.revision = version_id
        publish_metadata(self._ctx)


def _register_dummy_boot_tensors(vllm_config, model) -> _DummyBootPublication | None:
    """Register a dummy-loaded model with NIXL so later updates can publish it.

    Runs before torch.compile and CUDA graph capture, which the tensor walker
    cannot traverse. Never raises: the worker just never becomes a P2P source.
    """
    from modelexpress.engines.vllm.adapter import build_vllm_load_context

    try:
        ctx = build_vllm_load_context(vllm_config, vllm_config.model_config)
        register_tensors(model, ctx)
    except Exception:
        logger.exception(
            "Failed to register dummy-loaded tensors with ModelExpress; "
            "this worker will not be a P2P source"
        )
        return None
    if ctx.nixl_manager is None:
        logger.warning(
            "ModelExpress did not register dummy-loaded tensors; "
            "this worker will not be a P2P source"
        )
        return None
    logger.info("Registered %d dummy-loaded tensors with ModelExpress", len(ctx.tensors))
    return _DummyBootPublication(ctx)


def runtime_tensor_drain_timeout_seconds() -> int:
    """Outlast every read lease granted before the drain began.

    Leases cannot be renewed, so a drain this long never times out on a
    legitimate reader.
    """
    return _tensor_read_lease_timeout_seconds() + _DRAIN_TIMEOUT_MARGIN_S


class ModelExpressNCCLWeightTransferEngine(NCCLWeightTransferEngine):
    """vLLM's NCCL engine plus ModelExpress publication of the target model.

    Workers loaded through ModelExpress publish through their loader. Workers
    started with ``--load-format dummy`` register their tensors here. Draft
    model updates pass through to NCCL without touching publication.
    """

    def __init__(self, config, vllm_config, device: torch.device, model) -> None:
        super().__init__(config, vllm_config, device, model)
        self._revision = rl_envs.MX_RUNTIME_PUBLISH_REVISION
        if self._revision is None:
            raise ValueError(
                "The modelexpress_nccl weight-transfer backend requires "
                "MX_RUNTIME_PUBLISH_REVISION"
            )
        self._dummy_boot: _DummyBootPublication | None = None
        if vllm_config.load_config.load_format == "dummy":
            self._dummy_boot = _register_dummy_boot_tensors(vllm_config, model)

    def _publication(self) -> _RuntimePublication | None:
        if self.model is not self._default_model:
            return None
        loader = get_model_loader(self.device.index)
        if loader is not None:
            return loader
        return self._dummy_boot

    def start_weight_update(self) -> None:
        publication = self._publication()
        if publication is not None:
            # Raises if readers do not drain, so NCCL never mutates read tensors.
            publication.unpublish_runtime_tensors(
                drain_timeout=runtime_tensor_drain_timeout_seconds()
            )
        super().start_weight_update()

    def finish_weight_update(self) -> None:
        super().finish_weight_update()
        publication = self._publication()
        if publication is None:
            return
        try:
            publication.publish_runtime_tensors(self._revision)
        except Exception:
            logger.exception(
                "Failed to republish updated weights as ModelExpress revision %s; "
                "continuing without this worker as a P2P source",
                self._revision,
            )
