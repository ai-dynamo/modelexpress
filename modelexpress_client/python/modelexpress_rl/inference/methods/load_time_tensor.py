# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load-time tensor preparation over NIXL."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from modelexpress.refit.timing import (
    add_refit_bytes,
    add_refit_duration,
    refit_span,
    set_refit_cold,
)

from ...train import WeightPayloadFormat
from ..nixl_staged_transfer import (
    _NixlStagedTransfer,
    _StagedNixlWeights,
)
from ..plan import (
    MethodCapabilities,
    PreparedArtifact,
    PreparedEngineTensors,
    PreparedStreamingTensors,
    ResolvedSource,
    ResolvedTrainerSource,
    UpdateMethod,
    WeightSource,
)


class LoadTimeTensorNixlUpdateMethod(UpdateMethod):
    """Prepare trainer tensors in the engine's load-time layout."""

    def __init__(
        self,
        *,
        transfer: _NixlStagedTransfer,
    ) -> None:
        if transfer is None:
            raise ValueError("transfer is required")
        self._transfer = transfer
        self._active_staged: _StagedNixlWeights | None = None
        self._active_streamed: PreparedStreamingTensors | None = None

    @property
    def capabilities(self) -> MethodCapabilities:
        return MethodCapabilities(
            payload_formats=frozenset({WeightPayloadFormat.FULL_TENSOR}),
            sources=frozenset({WeightSource.TRAINER}),
            artifact_type=PreparedEngineTensors,
        )

    def prepare(self, *, version, source: ResolvedSource) -> PreparedArtifact:
        if self._active_staged is not None or self._active_streamed is not None:
            raise RuntimeError("release the active update before preparing another")
        if not isinstance(source, ResolvedTrainerSource):
            raise TypeError("load-time tensor method requires a trainer source")
        manifests = [item.manifest for item in source.snapshot.shards]
        is_streaming = self._transfer.bounded_staging
        with refit_span("transfer_planning", accumulate_metadata=True) as counters:
            prepared = self._transfer.prepare(
                manifests=manifests,
                trainer_snapshot=source.snapshot,
                checksums=source.checksums,
            )
            counters.update(prepared.metrics)
        set_refit_cold(not bool(prepared.metrics.get("plan_cache_hits")))
        if is_streaming:
            metrics = dict(prepared.metrics)
            streamed = PreparedStreamingTensors(
                batches=lambda: self._transfer.iter_bounded(prepared, metrics),
                parameter_names=frozenset(
                    name for batch in prepared.batches for name in batch.layouts[0]
                ),
                transfer_metrics=metrics,
            )
            self._active_streamed = streamed
            return streamed
        self._active_staged = self._transfer.stage(prepared)
        _attribute_transfer(self._active_staged.metrics)
        return PreparedEngineTensors(staged=self._active_staged)

    @contextmanager
    def installation_context(self, prepared: PreparedArtifact) -> Iterator[None]:
        if isinstance(prepared, PreparedStreamingTensors):
            if prepared is not self._active_streamed:
                raise RuntimeError("streaming update does not own the active source")
            if prepared.ownership.release_blocked:
                raise RuntimeError("streaming cleanup is unproven; reset the process")
        try:
            yield
        finally:
            if isinstance(prepared, PreparedStreamingTensors):
                _attribute_transfer(prepared.transfer_metrics)

    def release(self, prepared: PreparedArtifact) -> None:
        if isinstance(prepared, PreparedStreamingTensors):
            if prepared is not self._active_streamed:
                raise RuntimeError("streaming update is no longer active")
            if prepared.ownership.release_blocked:
                raise RuntimeError(
                    "streaming cleanup is unproven; reset the process before release"
                )
            self._active_streamed = None
            return
        if not isinstance(prepared, PreparedEngineTensors):
            raise TypeError("load-time tensor method requires staged engine tensors")
        if prepared.staged is not self._active_staged:
            raise RuntimeError("load-time staged weight is no longer active")
        self._active_staged = None

    def validate_close(self) -> None:
        if (
            self._active_streamed is not None
            and self._active_streamed.ownership.release_blocked
        ):
            raise RuntimeError(
                "streaming cleanup is unproven; retain resources and reset the process"
            )

    def close(self) -> None:
        self.validate_close()
        self._active_streamed = None
        self._active_staged = None
        self._transfer.close()


def _attribute_transfer(metrics: dict[str, float]) -> None:
    add_refit_bytes(metrics.get("bytes_received", 0))
    if "wire_s" in metrics:
        add_refit_duration("wire_transfer", metrics["wire_s"])
    if "reconstruct_s" in metrics:
        add_refit_duration("receive_sync", metrics["reconstruct_s"])


__all__ = ["LoadTimeTensorNixlUpdateMethod"]
