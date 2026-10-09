# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang generator integration for ModelExpress RL refit."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ...adapter import GeneratorEngineContext
from ...runtime import EngineRuntime
from .context import SglangGeneratorContext

if TYPE_CHECKING:
    from ...client import ModelExpressGeneratorClient


def get_modelexpress_generator(model_runner: Any) -> ModelExpressGeneratorClient:
    """Create or reuse the runner's S3 refit client from modelexpress_config.

    The client is cached on model_runner.modelexpress_generator; SGLang closes
    it during scheduler shutdown.
    """
    generator = getattr(model_runner, "modelexpress_generator", None)
    if generator is not None:
        return generator

    from ....object_storage import ObjectStorageType
    from ...client import ModelExpressGeneratorClient, ModelExpressGeneratorConfig
    from ...receiver import (
        DEFAULT_REFIT_CHECKPOINT_MAX_SIZE_GB,
        ObjectStorageGeneratorConfig,
    )

    config = model_runner.server_args.modelexpress_config
    if isinstance(config, str):
        config = json.loads(config)
    config = config or {}
    checkpoint = config.get("seed_checkpoint_path")
    if not checkpoint:
        checkpoint, _, _ = model_runner.loader._prepare_weights(
            model_runner.model_config.model_path,
            model_runner.model_config.revision,
            False,
        )
    generator = ModelExpressGeneratorClient.initialize(
        ModelExpressGeneratorConfig(
            engine_context=SglangGeneratorContext(model_runner),
            model_name=config.get("model_name"),
            server_url=config.get("server_url", config.get("url")),
            registration_ttl_seconds=config.get("registration_ttl_seconds"),
            lease_ttl_seconds=config.get("lease_ttl_seconds"),
            max_transfer_attempts=config.get("max_transfer_attempts", 3),
            max_replay_chain_length=config.get("max_replay_chain_length", 64),
            rpc_timeout_seconds=config.get("rpc_timeout_seconds", 30.0),
            object_storage=ObjectStorageGeneratorConfig(
                storage_type=ObjectStorageType.S3,
                initial_base_version_id=config["initial_base_version_id"],
                seed_checkpoint_path=checkpoint,
                refit_checkpoint_dir=config["refit_checkpoint_dir"],
                refit_checkpoint_max_size_gb=config.get(
                    "refit_checkpoint_max_size_gb",
                    DEFAULT_REFIT_CHECKPOINT_MAX_SIZE_GB,
                ),
                endpoint_url=config.get("object_storage_endpoint_url"),
                region_name=config.get("object_storage_region_name"),
            ),
        )
    )
    model_runner.modelexpress_generator = generator
    return generator


def _create_sglang_engine_runtime(
    engine_context: GeneratorEngineContext,
) -> EngineRuntime:
    if not isinstance(engine_context, SglangGeneratorContext):
        raise TypeError("SGLang requires a SglangGeneratorContext")
    from .installer import _SglangInstaller

    runner = engine_context.model_runner
    return EngineRuntime(
        model_name=runner.model_config.model_path,
        installer=_SglangInstaller(runner),
    )


__all__ = ["SglangGeneratorContext", "get_modelexpress_generator"]
