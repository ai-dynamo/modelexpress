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
import os
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

    def __init__(self, config, parallel_config, model=None) -> None:
        super().__init__(config, parallel_config, model)
        self._vllm_config = config
        self._model = model
        self._updater = MxVllmWeightUpdater()
        self._mdl = None

    def init_transfer_engine(self, init_info: MxInitInfo) -> None:
        # One init request is fanned out to every vLLM worker. Derive local
        # identity inside the worker rather than requiring the orchestrator to
        # construct a different request per TP/DP rank.
        init_info.device_id = (
            torch.cuda.current_device() if torch.cuda.is_available() else 0
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            init_info.worker_rank = torch.distributed.get_rank()
        if not init_info.model_name:
            model_config = getattr(self._vllm_config, "model_config", None)
            init_info.model_name = (
                getattr(model_config, "model", None)
                or getattr(model_config, "served_model_name", None)
                or ""
            )
        self._updater.initialize_weight_update_setup(init_info)

    def receive_weights(
        self,
        update_info: MxUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        try:
            from vllm.distributed.parallel_state import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            update_info.tp_world_size = int(
                get_tensor_model_parallel_world_size()
            )
            update_info.tp_rank = int(get_tensor_model_parallel_rank())
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Unable to introspect vLLM TP layout; using request values "
                "tp=%d rank=%d: %s",
                update_info.tp_world_size,
                update_info.tp_rank,
                exc,
            )
        if not 0 <= update_info.tp_rank < update_info.tp_world_size:
            raise ValueError(
                f"Invalid target TP identity: rank={update_info.tp_rank}, "
                f"world_size={update_info.tp_world_size}"
            )
        model = self._model or getattr(load_weights, "__self__", None)
        if model is not None and update_info.moe_expert_filter:
            try:
                for module in model.modules():
                    if hasattr(module, "expert_map") and hasattr(
                        module, "global_num_experts"
                    ):
                        update_info.ep_world_size = int(
                            getattr(module, "ep_size", 1) or 1
                        )
                        update_info.ep_rank = int(
                            getattr(module, "ep_rank", 0) or 0
                        )
                        update_info.num_experts = int(
                            getattr(module, "global_num_experts", 0) or 0
                        )
                        break
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Unable to introspect live vLLM EP layout; using request "
                    "values ep=%d rank=%d experts=%d: %s",
                    update_info.ep_world_size,
                    update_info.ep_rank,
                    update_info.num_experts,
                    exc,
                )
        load_callback = load_weights
        if os.environ.get("MX_LOAD_MODE", "stock").lower() == "direct":
            if model is None:
                logger.warning(
                    "MX_LOAD_MODE=direct requested, but load_weights is not a "
                    "bound model method; using stock loader"
                )
            else:
                if self._mdl is None:
                    from .mdl import MdlLoader

                    self._mdl = MdlLoader(model)
                load_callback = self._mdl.load_weights
        version = int(getattr(update_info, "version", 0))
        self._updater.start_weight_update(version)
        self._updater.update_weights(update_info, load_callback)
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
