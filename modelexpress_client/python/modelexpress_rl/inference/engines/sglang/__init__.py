# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang generator integration for ModelExpress RL refit."""

from ...adapter import GeneratorEngineContext
from ...runtime import EngineRuntime, FullTensorEngineCapability
from .context import SglangGeneratorContext


def _create_sglang_engine_runtime(
    engine_context: GeneratorEngineContext,
) -> EngineRuntime:
    if not isinstance(engine_context, SglangGeneratorContext):
        raise TypeError("SGLang requires a SglangGeneratorContext")
    from .installer import _SglangInstaller

    runner = engine_context.model_runner
    layout = None
    capability = None
    if engine_context.enable_full_tensor:
        from modelexpress.engines.sglang.adapter import SglangAdapter
        from sglang.srt.configs.device_config import DeviceConfig
        from .layout import SglangFullTensorLayout

        layout = SglangFullTensorLayout(runner)
        engine = SglangAdapter(
            runner.load_config,
            runner.model_config,
            DeviceConfig(device=runner.device, gpu_id=runner.gpu_id),
        )

        def build_identity(version_id):
            identity = engine.build_identity()
            identity.revision = version_id
            return identity

        capability = FullTensorEngineCapability(
            device_id=engine.get_device_id(),
            device=engine.get_target_device(),
            worker_rank=engine.get_worker_rank(),
            accelerator=engine.accelerator_backend.name,
            capture_layout=layout.capture,
            parameter_layout=layout.parameter_layout,
            build_identity=build_identity,
        )
    return EngineRuntime(
        model_name=runner.model_config.model_path,
        installer=_SglangInstaller(runner, layout=layout),
        full_tensor=capability,
    )


__all__ = ["SglangGeneratorContext"]
