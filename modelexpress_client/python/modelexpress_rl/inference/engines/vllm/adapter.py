# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM implementation of the generator-engine refit contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from modelexpress.engines.vllm.adapter import VllmAdapter

from modelexpress_rl.inference.adapter import (
    GeneratorTransferInputs,
    NixlGeneratorSource,
)
from modelexpress_rl.inference.nixl_staged_transfer import (
    _NixlStagedTransfer,
    _PreparedNixlTransfer,
    _StagedNixlWeights,
)
from modelexpress_rl.inference.receiver import (
    CanonicalS3GeneratorAdapter,
    PreparedCheckpoint,
    S3GeneratorConfig,
)
from modelexpress_rl.train import WeightPayloadFormat

from .installer import _VllmInstaller

if TYPE_CHECKING:
    from torch.nn import Module
    from vllm.config import ModelConfig, VllmConfig


class VllmGeneratorAdapter(CanonicalS3GeneratorAdapter):
    """Install NIXL full tensors or canonical S3 deltas into vLLM."""

    def __init__(
        self,
        *,
        model: Module,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        worker_id: str,
        model_name: str | None = None,
        s3: S3GeneratorConfig | None = None,
    ) -> None:
        self._uses_s3 = s3 is not None
        if self._uses_s3 and not model_name:
            raise ValueError("model_name is required for canonical S3")

        engine = VllmAdapter(vllm_config, model_config)
        device_id = engine.get_device_id()
        device = engine.get_target_device()
        self._installer = _VllmInstaller(
            model=model,
            vllm_config=vllm_config,
            model_config=model_config,
            device=device,
        )
        if s3 is not None:
            super().__init__(model_name=cast(str, model_name), config=s3)
            return

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
        if self._uses_s3:
            return super().supported_payload_formats
        return frozenset({WeightPayloadFormat.FULL_TENSOR})

    def stage_weight(
        self, inputs: GeneratorTransferInputs
    ) -> PreparedCheckpoint | _StagedNixlWeights:
        if self._uses_s3:
            return super().stage_weight(inputs)
        if self._active_staged is not None:
            raise RuntimeError(
                "release staged weight before replacing its transfer plan"
            )
        if inputs.payload_format not in self.supported_payload_formats:
            raise ValueError(
                f"VllmGeneratorAdapter does not support "
                f"{inputs.payload_format.value} payloads"
            )
        if any(
            not isinstance(source.transport, NixlGeneratorSource)
            for source in inputs.sources
        ):
            raise ValueError(
                "VllmGeneratorAdapter currently supports NIXL sources only"
            )
        if (
            self._active_plan is None
            or self._active_fingerprint != inputs.physical_fingerprint
        ):
            self._active_plan = self._transfer.prepare(
                manifests=[source.transport.manifest for source in inputs.sources],
                capture_layout=self._installer.capture,
            )
            self._active_fingerprint = inputs.physical_fingerprint
        try:
            staged = self._transfer.stage(
                cast(_PreparedNixlTransfer, self._active_plan)
            )
        except Exception:
            self._active_plan = None
            self._active_fingerprint = None
            raise
        self._active_staged = staged
        return staged

    def apply_weight(self, staged: object) -> dict[str, Any]:
        if self._uses_s3:
            return super().apply_weight(staged)
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

    def install_prepared_checkpoint(self, prepared: PreparedCheckpoint) -> None:
        self._installer.install_checkpoint(prepared.path)

    def release_staged_weight(self, staged: object) -> None:
        if self._uses_s3:
            super().release_staged_weight(staged)
            return
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
        """Release the selected rank-local transport."""
        if self._uses_s3:
            super().close()
            return
        self._active_staged = None
        self._active_plan = None
        self._active_fingerprint = None
        self._transfer.close()


__all__ = ["VllmGeneratorAdapter"]
