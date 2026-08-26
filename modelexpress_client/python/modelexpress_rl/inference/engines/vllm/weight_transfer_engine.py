# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM weight-transfer backend backed by ModelExpress WeightVersions."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from vllm.distributed.weight_transfer import WeightTransferEngine
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)

from modelexpress_rl.inference.client import (
    ModelExpressGeneratorClient,
    ModelExpressGeneratorConfig,
    StagedWeightHandle,
)
from modelexpress_rl.version import WeightVersionRef

from .context import VllmGeneratorContext

logger = logging.getLogger(__name__)


@dataclass
class ModelExpressWeightTransferInitInfo(WeightTransferInitInfo):
    """ModelExpress initializes from the vLLM worker's live model context."""


@dataclass
class ModelExpressWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Reference to one immutable READY ModelExpress WeightVersion."""

    version_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.version_id, str) or not self.version_id.strip():
            raise ValueError("version_id is required")


class ModelExpressWeightTransferEngine(WeightTransferEngine):
    """Install an exact ModelExpress WeightVersion at vLLM's safe point."""

    init_info_cls = ModelExpressWeightTransferInitInfo
    update_info_cls = ModelExpressWeightTransferUpdateInfo
    supports_draft_weight_update = False

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[Any],
        trainer_args: dict[str, Any] | Any,
    ) -> None:
        """Ignore vLLM's trainer transport; ModelExpress publishes versions."""
        logger.warning(
            "vLLM trainer transport is ignored; publish WeightVersions with "
            "ModelExpressTrainerClient instead"
        )

    def __init__(self, config, vllm_config, device, model) -> None:
        super().__init__(config, vllm_config, device, model)
        self._generator_config = ModelExpressGeneratorConfig(
            engine_context=VllmGeneratorContext(
                model=model,
                vllm_config=vllm_config,
            ),
            model_name=getattr(vllm_config.model_config, "model", None),
        )
        self._client: ModelExpressGeneratorClient | None = None
        self._update_active = False
        self._staged: StagedWeightHandle | None = None
        self._closed = False

    def init_transfer_engine(
        self, _init_info: ModelExpressWeightTransferInitInfo
    ) -> None:
        """Initialize ModelExpress from the rank-local vLLM model context."""
        if self._closed:
            logger.warning("weight transfer engine is shut down")
            return
        if self._client is not None:
            logger.warning("weight transfer engine is already initialized")
            return
        self._client = ModelExpressGeneratorClient.initialize(self._generator_config)

    def start_weight_update(self) -> None:
        if self._closed:
            logger.warning("weight transfer engine is shut down")
            return
        if self._client is None:
            logger.warning("weight transfer engine is not initialized")
            return
        if self._update_active:
            logger.warning("weight update is already active")
            return
        self._update_active = True
        self._staged = None

    def receive_weights(
        self, update_info: ModelExpressWeightTransferUpdateInfo
    ) -> None:
        if not self._update_active:
            logger.warning("weight update has not been started")
            return
        if self._staged is not None:
            logger.warning("weight update already received a version")
            return

        client = self._client
        if client is None:
            logger.warning("weight transfer engine is not initialized")
            self._update_active = False
            return

        staged: StagedWeightHandle | None = None
        try:
            staged = client.stage_weight(
                version=WeightVersionRef(update_info.version_id)
            )
            client.apply_weight(staged)
            self._staged = staged
        except BaseException as error:
            if staged is not None:
                try:
                    staged.release()
                except BaseException:
                    logger.warning(
                        "failed to release version %s while handling %s",
                        staged.version_id,
                        type(error).__name__,
                        exc_info=True,
                    )
            self._update_active = False
            self._staged = None
            raise

    def finish_weight_update(self) -> None:
        if not self._update_active:
            logger.warning("weight update has not been started")
            return
        if self._staged is None:
            logger.warning("weight update has not received a version")
            self._update_active = False
            return
        staged = self._staged
        try:
            staged.release()
        finally:
            self._staged = None
            self._update_active = False

    def shutdown(self) -> None:
        if self._closed:
            return
        try:
            if self._staged is not None:
                self._staged.release()
        finally:
            self._staged = None
            self._update_active = False
            if self._client is not None:
                self._client.close()
                self._client = None
            self._closed = True


__all__ = [
    "ModelExpressWeightTransferEngine",
    "ModelExpressWeightTransferInitInfo",
    "ModelExpressWeightTransferUpdateInfo",
]
