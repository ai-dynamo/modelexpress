# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-load runtime tensor transfer over NIXL."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from modelexpress.metadata.worker_server import TensorReadLease
from modelexpress.refit.timing import add_refit_bytes, add_refit_duration

from ...train import WeightPayloadFormat
from ..nixl_staged_transfer import _NixlStagedTransfer
from ..plan import (
    GeneratorPeerUpdateSource,
    MethodCapabilities,
    PreparedArtifact,
    PreparedRuntimeTensors,
    ResolvedSource,
    UpdateMethod,
    WeightSource,
)


@dataclass
class _PreparedRuntimeTensorRead:
    """Peer selection retained until the caller reaches its safe point."""

    tensor_read: TensorReadLease
    tensors: dict[str, torch.Tensor]
    metrics: dict[str, Any] = field(default_factory=dict)
    transfer_started: bool = False

    def mark_transfer_started(self) -> None:
        self.transfer_started = True


class RuntimeTensorNixlUpdateMethod(UpdateMethod):
    """Receive a generator peer directly into live post-PWAL tensors."""

    def __init__(
        self,
        *,
        transfer: _NixlStagedTransfer,
        runtime_tensors: dict[str, torch.Tensor],
    ) -> None:
        self._transfer = transfer
        self._runtime_tensors = runtime_tensors
        self._active_read: _PreparedRuntimeTensorRead | None = None

    @property
    def capabilities(self) -> MethodCapabilities:
        return MethodCapabilities(
            payload_formats=frozenset({WeightPayloadFormat.FULL_TENSOR}),
            sources=frozenset({WeightSource.GENERATOR}),
            artifact_type=PreparedRuntimeTensors,
        )

    def prepare(self, *, version, source: ResolvedSource) -> PreparedArtifact:
        del version
        if self._active_read is not None:
            raise RuntimeError("release staged weight before staging another version")
        if not isinstance(source, GeneratorPeerUpdateSource):
            raise TypeError("runtime tensor method requires a generator source")

        tensor_read = self._transfer.prepare_peer_read(
            source=source.worker,
            mx_source_id=source.mx_source_id,
            worker_id=source.worker_id,
            destination_tensors=self._runtime_tensors,
        )
        self._active_read = _PreparedRuntimeTensorRead(
            tensor_read=tensor_read,
            tensors=self._runtime_tensors,
        )
        return PreparedRuntimeTensors(staged=self._active_read)

    @contextmanager
    def installation_context(self, prepared: PreparedArtifact):
        if not isinstance(prepared, PreparedRuntimeTensors):
            raise TypeError("runtime tensor method requires a prepared peer read")
        read = prepared.staged
        if read is not self._active_read:
            raise RuntimeError("runtime tensor read is no longer active")
        try:
            read.metrics.update(
                self._transfer.receive_peer(
                    tensor_read=read.tensor_read,
                    destination_tensors=read.tensors,
                    on_transfer_start=read.mark_transfer_started,
                )
            )
        finally:
            read.tensor_read.close()
        _attribute_transfer(read.metrics)
        yield

    def mutated_during_installation_context(
        self,
        prepared: PreparedArtifact,
    ) -> bool:
        if not isinstance(prepared, PreparedRuntimeTensors):
            return False
        read = prepared.staged
        return read is self._active_read and read.transfer_started

    def release(self, prepared: PreparedArtifact) -> None:
        if not isinstance(prepared, PreparedRuntimeTensors):
            raise TypeError("runtime tensor method requires a prepared peer read")
        if prepared.staged is not self._active_read:
            raise RuntimeError("runtime tensor read is no longer active")
        try:
            self._active_read.tensor_read.close()
        finally:
            self._active_read = None

    def close(self) -> None:
        try:
            if self._active_read is not None:
                self._active_read.tensor_read.close()
        finally:
            self._active_read = None
            self._transfer.close()


def _attribute_transfer(metrics: dict[str, float]) -> None:
    add_refit_bytes(metrics.get("bytes_received", 0))
    if "wire_s" in metrics:
        add_refit_duration("wire_transfer", metrics["wire_s"])
    if "reconstruct_s" in metrics:
        add_refit_duration("receive_sync", metrics["reconstruct_s"])


__all__ = ["RuntimeTensorNixlUpdateMethod"]
