# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-tensor preparation over NIXL."""

from __future__ import annotations

from collections.abc import Callable

from modelexpress import p2p_pb2
from modelexpress.client import MxClientBase

from ...train import WeightPayloadFormat
from ..adapter import NixlGeneratorSource
from ..nixl_staged_transfer import (
    _NixlStagedTransfer,
    _PreparedNixlTransfer,
    _StagedNixlWeights,
)
from ..plan import (
    MethodCapabilities,
    GeneratorPeerUpdateSource,
    PreparedArtifact,
    PreparedEngineTensors,
    ResolvedSource,
    TrainerUpdateSource,
    WeightSource,
    UpdateMethod,
)


class FullTensorNixlUpdateMethod(UpdateMethod):
    """Prepare engine-layout tensors from trainer or generator NIXL sources."""

    def __init__(
        self,
        *,
        transfer: _NixlStagedTransfer,
        capture_layout: Callable,
        parameter_layout: Callable,
        build_identity: Callable[[str], p2p_pb2.SourceIdentity],
        worker_rank: int,
        worker_id: str,
        accelerator: str,
        p2p_client: MxClientBase,
    ) -> None:
        self._transfer = transfer
        self._capture_layout = capture_layout
        self._parameter_layout = parameter_layout
        self._build_identity = build_identity
        self._worker_rank = worker_rank
        self._worker_id = worker_id
        self._accelerator = accelerator
        self._p2p_client = p2p_client
        self._active_plan: _PreparedNixlTransfer | None = None
        self._active_fingerprint: tuple | None = None
        self._active_staged: _StagedNixlWeights | None = None

    @property
    def capabilities(self) -> MethodCapabilities:
        return MethodCapabilities(
            payload_formats=frozenset({WeightPayloadFormat.FULL_TENSOR}),
            sources=frozenset({WeightSource.TRAINER, WeightSource.GENERATOR}),
            artifact_type=PreparedEngineTensors,
        )

    def prepare(
        self,
        *,
        version,
        source: ResolvedSource,
    ) -> PreparedArtifact:
        del version
        if self._active_staged is not None:
            raise RuntimeError("release staged weight before staging another version")
        self._transfer.unpublish_peer()
        if isinstance(source, GeneratorPeerUpdateSource):
            staged = self._transfer.stage_peer(
                source=source.worker,
                parameter_layout=self._parameter_layout(),
            )
            self._active_plan = None
            self._active_fingerprint = None
        elif isinstance(source, TrainerUpdateSource):
            inputs = source.inputs
            if any(
                not isinstance(item.transport, NixlGeneratorSource)
                for item in inputs.sources
            ):
                raise ValueError("full-tensor method requires NIXL sources")
            reusable = (
                self._active_plan is not None
                and self._active_fingerprint == inputs.physical_fingerprint
            )
            if not reusable:
                self._active_plan = self._transfer.prepare(
                    manifests=[item.transport.manifest for item in inputs.sources],
                    capture_layout=self._capture_layout,
                )
                self._active_fingerprint = inputs.physical_fingerprint
            staged = self._transfer.stage(self._active_plan)
        else:
            raise TypeError("full-tensor method received an unsupported source")
        self._active_staged = staged
        return PreparedEngineTensors(staged=staged)

    def release(self, prepared: PreparedArtifact) -> None:
        if not isinstance(prepared, PreparedEngineTensors):
            raise TypeError("full-tensor method requires staged engine tensors")
        if prepared.staged is not self._active_staged:
            raise RuntimeError("full-tensor staged weight is no longer active")
        self._active_staged = None

    def publish_applied(self, *, version_id: str, prepared: PreparedArtifact) -> None:
        if not isinstance(prepared, PreparedEngineTensors):
            raise TypeError("full-tensor method requires staged engine tensors")
        if prepared.staged is not self._active_staged:
            raise RuntimeError("full-tensor staged weight is no longer active")
        self._transfer.publish_peer(
            staged=self._active_staged,
            identity=self._build_identity(version_id),
            p2p_client=self._p2p_client,
            worker_rank=self._worker_rank,
            worker_id=self._worker_id,
            accelerator=self._accelerator,
        )

    def close(self) -> None:
        self._active_staged = None
        self._active_plan = None
        self._active_fingerprint = None
        self._transfer.close()


__all__ = ["FullTensorNixlUpdateMethod"]
