# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM implementation of the generator-engine refit contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from modelexpress import p2p_pb2
from modelexpress.client import MxClientBase
from modelexpress.engines.vllm.adapter import VllmAdapter
from modelexpress_rl import envs as rl_envs
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
    ObjectStorageGeneratorConfig,
    PreparedCheckpoint,
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
        object_storage: ObjectStorageGeneratorConfig | None = None,
    ) -> None:
        self._uses_s3 = object_storage is not None

        engine = VllmAdapter(vllm_config, model_config)
        self._engine = engine
        device_id = engine.get_device_id()
        device = engine.get_target_device()
        self._installer = _VllmInstaller(
            model=model,
            vllm_config=vllm_config,
            model_config=model_config,
            device=device,
        )
        if object_storage is not None:
            super().__init__(
                model_name=vllm_config.model_config.model,
                config=object_storage,
            )
            return

        self._transfer = _NixlStagedTransfer(
            agent_name=f"mx-refit-{worker_id}",
            device_id=device_id,
            device=device,
            listen_port=rl_envs.MX_REFIT_METADATA_PORT + device_id,
        )
        self._active_plan: _PreparedNixlTransfer | None = None
        self._active_fingerprint: tuple | None = None
        self._active_staged: _StagedNixlWeights | None = None

    @property
    def worker_rank(self) -> int:
        return self._engine.get_worker_rank()

    def build_p2p_identity(self, version_id: str) -> p2p_pb2.SourceIdentity:
        identity = self._engine.build_identity()
        identity.revision = version_id
        return identity

    def stage_peer_weight(self, source: p2p_pb2.WorkerMetadata) -> _StagedNixlWeights:
        self._active_plan = None
        self._active_fingerprint = None
        staged = self._transfer.stage_peer(
            source=source,
            parameter_layout=self._installer.parameter_layout(),
        )
        self._active_staged = staged
        return staged

    def publish_weight_version(
        self,
        *,
        version_id: str,
        staged: object,
        p2p_client: MxClientBase,
        worker_id: str,
    ) -> None:
        active_staged = self._active_staged
        if active_staged is None or staged is not active_staged:
            raise RuntimeError("vLLM staged weight is no longer active")
        self._transfer.publish_peer(
            staged=active_staged,
            identity=self.build_p2p_identity(version_id),
            p2p_client=p2p_client,
            worker_rank=self.worker_rank,
            worker_id=worker_id,
            accelerator=self._engine.accelerator_backend.name,
        )

    @property
    def supported_payload_formats(self) -> frozenset[WeightPayloadFormat]:
        if self._uses_s3:
            return super().supported_payload_formats
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
        if any(
            not isinstance(source.transport, NixlGeneratorSource)
            for source in inputs.sources
        ):
            raise ValueError(
                "VllmGeneratorAdapter currently supports NIXL sources only"
            )
        plan = self._transfer.prepare(
            manifests=[source.transport.manifest for source in inputs.sources],
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

    def stage_weight(self, plan: object) -> PreparedCheckpoint | _StagedNixlWeights:
        if self._uses_s3:
            return super().stage_weight(cast(GeneratorTransferInputs, plan))
        if plan is not self._active_plan:
            raise RuntimeError("vLLM transfer plan is no longer active")
        self._transfer.unpublish_peer()
        staged = self._transfer.stage(cast(_PreparedNixlTransfer, plan))
        self._active_staged = staged
        return staged

    def apply_weight(self, staged: object) -> dict[str, Any]:
        if self._uses_s3:
            return super().apply_weight(staged)
        active_staged = self._active_staged
        if active_staged is None or staged is not active_staged:
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
        if active_staged is None or staged is not active_staged:
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
