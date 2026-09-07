# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM weight-transfer backend backed by ModelExpress WeightVersions."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, replace
from time import perf_counter
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
from modelexpress_rl.inference.receiver import (
    DEFAULT_REFIT_CHECKPOINT_MAX_SIZE_GB,
    ObjectStorageGeneratorConfig,
)
from modelexpress_rl.object_storage import ObjectStorageType
from modelexpress_rl.version import WeightVersionRef

from .context import VllmGeneratorContext

logger = logging.getLogger(__name__)


@dataclass
class ModelExpressWeightTransferInitInfo(WeightTransferInitInfo):
    """Optional ModelExpress connection and object-storage settings."""

    model_name: str | None = None
    initial_base_version_id: str | None = None
    seed_checkpoint_path: str | None = None
    refit_checkpoint_dir: str | None = None
    refit_checkpoint_max_size_gb: int | None = DEFAULT_REFIT_CHECKPOINT_MAX_SIZE_GB
    server_url: str | None = None
    object_storage_type: str | None = None
    object_storage_endpoint_url: str | None = None
    object_storage_region_name: str | None = None
    registration_ttl_seconds: int | None = None
    lease_ttl_seconds: int | None = None
    max_transfer_attempts: int = 3
    max_replay_chain_length: int = 64
    rpc_timeout_seconds: float = 30.0


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
        self._active_version_id: str | None = None
        self._staged: StagedWeightHandle | None = None
        self._closed = False

    def init_transfer_engine(
        self, init_info: ModelExpressWeightTransferInitInfo
    ) -> None:
        """Initialize ModelExpress from the rank-local vLLM model context."""
        if self._closed:
            logger.warning("weight transfer engine is shut down")
            return
        if self._client is not None:
            logger.warning("weight transfer engine is already initialized")
            return

        object_storage_values = (
            init_info.object_storage_type,
            init_info.initial_base_version_id,
            init_info.seed_checkpoint_path,
            init_info.refit_checkpoint_dir,
            init_info.object_storage_endpoint_url,
            init_info.object_storage_region_name,
        )
        object_storage = None
        if any(value is not None for value in object_storage_values):
            if (
                init_info.object_storage_type is None
                or init_info.initial_base_version_id is None
                or init_info.seed_checkpoint_path is None
                or init_info.refit_checkpoint_dir is None
            ):
                raise ValueError(
                    "object storage requires object_storage_type, "
                    "initial_base_version_id, seed_checkpoint_path, and "
                    "refit_checkpoint_dir"
                )
            try:
                storage_type = ObjectStorageType(init_info.object_storage_type)
            except ValueError as error:
                raise ValueError(
                    f"unsupported object_storage_type="
                    f"{init_info.object_storage_type!r}"
                ) from error
            object_storage = ObjectStorageGeneratorConfig(
                storage_type=storage_type,
                initial_base_version_id=init_info.initial_base_version_id,
                seed_checkpoint_path=init_info.seed_checkpoint_path,
                refit_checkpoint_dir=init_info.refit_checkpoint_dir,
                refit_checkpoint_max_size_gb=(
                    init_info.refit_checkpoint_max_size_gb
                ),
                endpoint_url=init_info.object_storage_endpoint_url,
                region_name=init_info.object_storage_region_name,
            )

        model_name = (
            init_info.model_name
            if init_info.model_name is not None
            else self._generator_config.model_name
        )
        self._client = ModelExpressGeneratorClient.initialize(
            replace(
                self._generator_config,
                model_name=model_name,
                server_url=init_info.server_url,
                registration_ttl_seconds=init_info.registration_ttl_seconds,
                lease_ttl_seconds=init_info.lease_ttl_seconds,
                max_transfer_attempts=init_info.max_transfer_attempts,
                max_replay_chain_length=init_info.max_replay_chain_length,
                rpc_timeout_seconds=init_info.rpc_timeout_seconds,
                object_storage=object_storage,
            )
        )
        logger.info("ModelExpress weight transfer initialized model=%s", model_name)

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
        self._active_version_id = None
        self._staged = None
        logger.info("ModelExpress weight update started")

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
            logger.info(
                "ModelExpress weight update receiving version=%s",
                update_info.version_id,
            )
            stage_started = perf_counter()
            staged = client.stage_weight(
                version=WeightVersionRef(update_info.version_id)
            )
            stage_weight_time = perf_counter() - stage_started
            metrics = staged.metrics
            apply_metrics = client.apply_weight(staged)
            if isinstance(apply_metrics, dict):
                metrics.update(apply_metrics)
            metrics["perf/mx_receive_stage_weight_time"] = stage_weight_time
            for key, value in sorted(metrics.items()):
                if key.startswith("perf/") and isinstance(value, (int, float)):
                    logger.info("ModelExpress receiver metric %s=%s", key, value)
            self._staged = staged
            self._active_version_id = update_info.version_id
            logger.info(
                "ModelExpress weight update applied version=%s",
                update_info.version_id,
            )
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
            self._active_version_id = None
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
        version_id = self._active_version_id
        try:
            staged.release()
            logger.info(
                "ModelExpress weight update finished version=%s",
                version_id,
            )
        finally:
            self._staged = None
            self._active_version_id = None
            self._update_active = False

    def shutdown(self) -> None:
        if self._closed:
            return
        try:
            if self._staged is not None:
                self._staged.release()
        finally:
            self._staged = None
            self._active_version_id = None
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
