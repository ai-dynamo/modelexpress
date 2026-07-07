# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier-3: ModelExpress backend for vLLM's native weight-transfer API.

Thin adapter conforming to vLLM's ``WeightTransferEngine`` ABC. All the
vLLM-specific logic — geometry discovery, receive/translate, buffer
registration, EP filter, verify — lives in the tier-2 weight-update layer
(:class:`modelexpress.engines.vllm.weight_update.MxVllmWeightUpdater`), which in
turn uses the generic tier-1 :class:`modelexpress.MxV2RefitReceiver`. This
backend just maps the ABC's 4-phase protocol onto the tier-2 lifecycle:

  * ``init_transfer_engine``  -> ``initialize_weight_update_setup``
  * ``receive_weights``       -> ``start_weight_update`` + ``update_weights`` + ``finish_weight_update``
  * ``trainer_send_weights``  -> not used (MX publishes via MxV2TrainingPublisher)
  * ``shutdown``.

A framework that wants finer control can drive the tier-2 layer directly rather
than through this ABC (same base APIs).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterator

import torch

from vllm.distributed.weight_transfer.base import WeightTransferEngine

# Tier-2 layer + shared info objects (which subclass vLLM's ABC info bases).
# Re-exported here for back-compat with callers importing them from this module.
from .weight_update import (  # noqa: F401
    MxInitInfo,
    MxTrainerSendArgs,
    MxUpdateInfo,
    MxVllmWeightUpdater,
    WeightSubset,
)

logger = logging.getLogger("modelexpress.engines.vllm.weight_transfer")

_MX_ENGINE_NAME = "mx"


class MxWeightTransferEngine(WeightTransferEngine[MxInitInfo, MxUpdateInfo]):
    """ModelExpress / NIXL RDMA backend for vLLM weight transfer (tier-3 adapter)."""

    init_info_cls = MxInitInfo
    update_info_cls = MxUpdateInfo

    def __init__(self, config, parallel_config) -> None:
        super().__init__(config, parallel_config)
        self._updater = MxVllmWeightUpdater()

    def init_transfer_engine(self, init_info: MxInitInfo) -> None:
        self._updater.initialize_weight_update_setup(init_info)

    def receive_weights(
        self,
        update_info: MxUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        version = int(getattr(update_info, "version", 0))
        self._updater.start_weight_update(version)
        self._updater.update_weights(update_info, load_weights)
        self._updater.finish_weight_update(version)

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | Any,
    ) -> None:
        raise NotImplementedError(
            "MX trainer send is driven by MxV2TrainingPublisher in the NeMo-RL "
            "Megatron policy worker (stream_weights_via_mx), not this hook."
        )

    def shutdown(self) -> None:
        self._updater.shutdown()


def register() -> None:
    """Register the MX backend with vLLM's WeightTransferEngineFactory."""
    from vllm.distributed.weight_transfer import WeightTransferEngineFactory
    WeightTransferEngineFactory.register_engine(_MX_ENGINE_NAME, MxWeightTransferEngine)


# Best-effort auto-register on import (safe: no-op if vLLM factory absent).
try:
    register()
except Exception:  # noqa: BLE001
    pass
