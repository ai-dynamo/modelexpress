# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM implementation of the generator-engine refit contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from modelexpress.engines.vllm.adapter import VllmAdapter

from modelexpress_rl.inference.adapter import (
    GeneratorEngineAdapter,
    GeneratorTransferInputs,
)
from modelexpress_rl.inference.nixl_staged_transfer import (
    _NixlStagedTransfer,
    _PreparedNixlTransfer,
    _StagedNixlWeights,
)
from modelexpress_rl.train import WeightPayloadFormat

from .installer import _VllmInstaller

if TYPE_CHECKING:
    from torch.nn import Module
    from vllm.config import ModelConfig, VllmConfig

class VllmGeneratorAdapter(GeneratorEngineAdapter):
    """Compose exact-version NIXL staging with graph-safe vLLM installation."""

    def __init__(
        self,
        *,
        model: Module,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        worker_id: str,
    ) -> None:
        engine = VllmAdapter(vllm_config, model_config)
        device_id = engine.get_device_id()
        device = engine.get_target_device()
        self._installer = _VllmInstaller(
            model=model,
            vllm_config=vllm_config,
            model_config=model_config,
            device=device,
        )
        self._transfer = _NixlStagedTransfer(
            agent_name=f"mx-refit-{worker_id}",
            device_id=device_id,
            device=device,
        )
        self._active_plan: _PreparedNixlTransfer | None = None
        self._active_fingerprint: tuple | None = None
        self._active_staged: _StagedNixlWeights | None = None

    @property
    def supported_payload_formats(self) -> frozenset[WeightPayloadFormat]:
        return frozenset({WeightPayloadFormat.FULL_TENSOR})

    def create_transfer_plan(
        self, inputs: GeneratorTransferInputs
    ) -> _PreparedNixlTransfer:
        if self._active_staged is not None:
            raise RuntimeError(
                "release staged weight before replacing its transfer plan"
            )
        if inputs.payload_format not in self.supported_payload_formats:
            raise ValueError(
                f"VllmGeneratorAdapter does not support "
                f"{inputs.payload_format.value} payloads"
            )
        if any(source.transport.upper() != "NIXL" for source in inputs.sources):
            raise ValueError(
                "VllmGeneratorAdapter currently supports NIXL sources only"
            )
        plan = self._transfer.prepare(
            manifests=[source.manifest for source in inputs.sources],
            capture_layout=self._installer.capture,
        )
        self._active_plan = plan
        self._active_fingerprint = inputs.physical_fingerprint
        return plan

    def validate_transfer_plan(
        self,
        plan: object,
        inputs: GeneratorTransferInputs,
    ) -> bool:
        return (
            plan is self._active_plan
            and self._active_fingerprint == inputs.physical_fingerprint
        )

    def stage_weight(self, plan: object) -> _StagedNixlWeights:
        if plan is not self._active_plan:
            raise RuntimeError("vLLM transfer plan is no longer active")
        staged = self._transfer.stage(cast(_PreparedNixlTransfer, plan))
        self._active_staged = staged
        return staged

    def apply_weight(self, staged: object) -> dict[str, Any]:
        active_staged = self._active_staged
        if (
            active_staged is None
            or staged is not active_staged
            or self._active_plan is None
            or active_staged.plan_revision != self._active_plan.plan_revision
        ):
            raise RuntimeError("vLLM staged weight is no longer active")
        self._installer.install(active_staged.tensors)
        return active_staged.metrics

    def release_staged_weight(self, staged: object) -> None:
        active_staged = self._active_staged
        if (
            active_staged is None
            or staged is not active_staged
            or self._active_plan is None
            or active_staged.plan_revision != self._active_plan.plan_revision
        ):
            raise RuntimeError("vLLM staged weight is no longer active")
        # Registered buffers are reusable plan workspace; releasing a version
        # invalidates only this staged handle.
        self._active_staged = None

    def close(self) -> None:
        """Release the rank-local NIXL agent."""
        self._active_staged = None
        self._active_plan = None
        self._active_fingerprint = None
        self._transfer.close()


__all__ = ["VllmGeneratorAdapter"]
