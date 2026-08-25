# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native vLLM bridge for canonical ModelExpress S3 refit."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter

import torch

from modelexpress_rl.inference.client import (
    ModelExpressGeneratorClient,
    ModelExpressGeneratorConfig,
)
from modelexpress_rl.inference.engines.vllm.context import VllmGeneratorContext
from modelexpress_rl.inference.receiver import S3GeneratorConfig
from modelexpress_rl.train import WeightPayloadFormat
from modelexpress_rl.version import WeightVersionRef

from vllm.config import VllmConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)


logger = logging.getLogger(__name__)


@dataclass
class MxRefitInitInfo(WeightTransferInitInfo):
    """Configuration for one rank-local canonical S3 receiver."""

    model_name: str
    initial_base_version_id: str
    launch_checkpoint: str
    preparation_cache_dir: str
    server_url: str | None = None
    s3_endpoint_url: str | None = None
    s3_region_name: str | None = None
    registration_ttl_seconds: int | None = None
    lease_ttl_seconds: int | None = None
    max_transfer_attempts: int = 3
    rpc_timeout_seconds: float = 30.0


@dataclass
class MxRefitUpdateInfo(WeightTransferUpdateInfo):
    """Opaque ModelExpress version to stage and install."""

    version_id: str


class MxRefitWeightTransferEngine(
    WeightTransferEngine[MxRefitInitInfo, MxRefitUpdateInfo]
):
    """Drive the current ModelExpress generator API through vLLM hooks."""

    init_info_cls = MxRefitInitInfo
    update_info_cls = MxRefitUpdateInfo
    supports_draft_weight_update = False

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: VllmConfig,
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        self._client: ModelExpressGeneratorClient | None = None

    def init_transfer_engine(self, init_info: MxRefitInitInfo) -> None:
        self.shutdown()
        self._client = ModelExpressGeneratorClient.initialize(
            ModelExpressGeneratorConfig(
                engine_context=VllmGeneratorContext(
                    model=self.model,
                    vllm_config=self.vllm_config,
                ),
                model_name=init_info.model_name,
                payload_format=WeightPayloadFormat.XOR_DELTA,
                server_url=init_info.server_url,
                registration_ttl_seconds=init_info.registration_ttl_seconds,
                lease_ttl_seconds=init_info.lease_ttl_seconds,
                max_transfer_attempts=init_info.max_transfer_attempts,
                rpc_timeout_seconds=init_info.rpc_timeout_seconds,
                s3=S3GeneratorConfig(
                    initial_base_version_id=init_info.initial_base_version_id,
                    launch_checkpoint=init_info.launch_checkpoint,
                    preparation_cache_dir=init_info.preparation_cache_dir,
                    endpoint_url=init_info.s3_endpoint_url,
                    region_name=init_info.s3_region_name,
                ),
            )
        )

    def start_weight_update(self) -> None:
        """The generator adapter owns its complete installation window."""

    def receive_weights(self, update_info: MxRefitUpdateInfo) -> None:
        assert self._client is not None
        stage_started = perf_counter()
        staged = self._client.stage_weight(
            version=WeightVersionRef(update_info.version_id)
        )
        stage_weight_time = perf_counter() - stage_started
        try:
            metrics = staged.metrics
            apply_metrics = self._client.apply_weight(staged)
            if isinstance(apply_metrics, dict):
                metrics.update(apply_metrics)
            metrics["perf/mx_receive_stage_weight_time"] = stage_weight_time
            for key, value in sorted(metrics.items()):
                if key.startswith("perf/") and isinstance(value, (int, float)):
                    logger.info("ModelExpress receiver metric %s=%s", key, value)
        finally:
            staged.release()

    def finish_weight_update(self) -> None:
        """The generator adapter completes installation in ``apply_weight``."""

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    @staticmethod
    def trainer_send_weights(iterator, trainer_args) -> None:
        del iterator, trainer_args
        raise NotImplementedError(
            "ModelExpress canonical S3 revisions are receiver-pulled"
        )


__all__ = [
    "MxRefitInitInfo",
    "MxRefitUpdateInfo",
    "MxRefitWeightTransferEngine",
]
