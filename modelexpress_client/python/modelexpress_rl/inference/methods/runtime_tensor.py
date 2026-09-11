# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-load runtime tensor transfer over NIXL."""

from __future__ import annotations

import torch
from modelexpress.refit.timing import add_refit_bytes, add_refit_duration

from ...train import WeightPayloadFormat
from ..nixl_staged_transfer import _NixlStagedTransfer, _StagedNixlWeights
from ..plan import (
    GeneratorPeerUpdateSource,
    MethodCapabilities,
    PreparedArtifact,
    PreparedRuntimeTensors,
    ResolvedSource,
    UpdateMethod,
    WeightSource,
)


class RuntimeTensorNixlUpdateMethod(UpdateMethod):
    """Stage a generator peer's processed runtime tensor representation."""

    def __init__(
        self,
        *,
        transfer: _NixlStagedTransfer,
        runtime_tensors: dict[str, torch.Tensor],
    ) -> None:
        self._transfer = transfer
        self._runtime_tensors = runtime_tensors
        self._active_staged: _StagedNixlWeights | None = None

    @property
    def capabilities(self) -> MethodCapabilities:
        return MethodCapabilities(
            payload_formats=frozenset({WeightPayloadFormat.FULL_TENSOR}),
            sources=frozenset({WeightSource.GENERATOR}),
            artifact_type=PreparedRuntimeTensors,
        )

    def prepare(self, *, version, source: ResolvedSource) -> PreparedArtifact:
        del version
        if self._active_staged is not None:
            raise RuntimeError("release staged weight before staging another version")
        if not isinstance(source, GeneratorPeerUpdateSource):
            raise TypeError("runtime tensor method requires a generator source")
        layout = {
            name: (tuple(tensor.shape), tensor.dtype)
            for name, tensor in self._runtime_tensors.items()
        }
        self._active_staged = self._transfer.stage_peer(
            source=source.worker,
            parameter_layout=layout,
        )
        _attribute_transfer(self._active_staged.metrics)
        return PreparedRuntimeTensors(staged=self._active_staged)

    def release(self, prepared: PreparedArtifact) -> None:
        if not isinstance(prepared, PreparedRuntimeTensors):
            raise TypeError("runtime tensor method requires staged runtime tensors")
        if prepared.staged is not self._active_staged:
            raise RuntimeError("runtime staged weight is no longer active")
        self._active_staged = None

    def close(self) -> None:
        self._active_staged = None
        self._transfer.close()

def _attribute_transfer(metrics: dict[str, float]) -> None:
    add_refit_bytes(metrics.get("bytes_received", 0))
    if "wire_s" in metrics:
        add_refit_duration("wire_transfer", metrics["wire_s"])
    if "reconstruct_s" in metrics:
        add_refit_duration("receive_sync", metrics["reconstruct_s"])


__all__ = ["RuntimeTensorNixlUpdateMethod"]
