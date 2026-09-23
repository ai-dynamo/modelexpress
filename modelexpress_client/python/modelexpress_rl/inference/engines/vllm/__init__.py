# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM generator integration."""

from __future__ import annotations

from modelexpress import envs

from ...adapter import GeneratorEngineContext
from ...runtime import EngineRuntime, FullTensorEngineCapability
from .context import VllmGeneratorContext


def _create_vllm_engine_runtime(
    engine_context: GeneratorEngineContext,
) -> EngineRuntime:
    if not isinstance(engine_context, VllmGeneratorContext):
        raise TypeError("VLLM requires a VllmGeneratorContext")

    from torch.nn import Module
    from vllm.config import ModelConfig, VllmConfig

    from modelexpress.engines.vllm.adapter import VllmAdapter
    from modelexpress.engines.vllm.loader import get_model_loader

    from .installer import _VllmInstaller

    model = engine_context.model
    vllm_config = engine_context.vllm_config
    model_config = vllm_config.model_config
    if not isinstance(model, Module):
        raise TypeError("vLLM engine context model must be a torch Module")
    if not isinstance(vllm_config, VllmConfig):
        raise TypeError("vLLM engine context vllm_config must be a VllmConfig")
    if not isinstance(model_config, ModelConfig):
        raise TypeError("vLLM engine context model_config must be a ModelConfig")
    engine = VllmAdapter(vllm_config, model_config)
    loader = get_model_loader(engine.get_device_id())
    runtime_tensors = (
        loader.tensors
        if loader is not None
        and loader.tensors
        and loader.nixl_manager is not None
        else None
    )

    installer = _VllmInstaller(
        model=model,
        vllm_config=vllm_config,
        model_config=model_config,
        device=engine.get_target_device(),
        convert_native_to_hf=engine_context.convert_native_to_hf,
        runtime_tensors=runtime_tensors,
    )

    def build_identity(version_id: str):
        identity = engine.build_identity()
        identity.revision = version_id
        return identity

    def unpublish_runtime_tensors() -> None:
        if loader is not None:
            loader.unpublish_runtime_tensors()

    def publish_runtime_tensors(version_id: str) -> None:
        if loader is not None:
            loader.publish_runtime_tensors(version_id)

    return EngineRuntime(
        model_name=envs.MX_MODEL_NAME_OVERRIDE or vllm_config.model_config.model,
        installer=installer,
        full_tensor=FullTensorEngineCapability(
            device_id=engine.get_device_id(),
            device=engine.get_target_device(),
            worker_rank=engine.get_worker_rank(),
            capture_layout=installer.capture,
            runtime_tensors=runtime_tensors,
            source_worker_id=(
                loader.worker_id if runtime_tensors is not None else None
            ),
            unpublish_runtime_tensors=unpublish_runtime_tensors,
            publish_runtime_tensors=publish_runtime_tensors,
            build_identity=build_identity,
            nixl_manager=loader.nixl_manager if loader is not None else None,
        ),
    )


__all__ = ["VllmGeneratorContext"]
