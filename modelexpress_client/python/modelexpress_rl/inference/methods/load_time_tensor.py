# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load-time tensor preparation over NIXL."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from modelexpress_rl.refit.timing import (
    add_refit_bytes,
    add_refit_duration,
    refit_span,
    set_refit_cold,
)

from ...train import WeightPayloadFormat
from ..nixl_staged_transfer import (
    _NixlStagedTransfer,
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
    """Prepare trainer source writes for engine installation.

    Own one active trainer artifact, staging a full copy eagerly or deferring bounded
    reads until installation. Delegate receive resources to the transfer executor and
    batch lifetime to the streaming artifact. Determine release safety from its
    cleanup state before allowing resource or version-lease release.
    """

    def __init__(
        self,
        *,
        transfer: _NixlStagedTransfer,
    ) -> None:
        if transfer is None:
            raise ValueError("transfer is required")
        self._transfer = transfer
        self._active: PreparedEngineTensors | PreparedStreamingTensors | None = None

    @property
    def capabilities(self) -> MethodCapabilities:
        return self.capabilities_for(self._transfer.bounded_staging)

    @staticmethod
    def capabilities_for(bounded: bool) -> MethodCapabilities:
        """Describe configured output before receive resources are allocated."""
        return MethodCapabilities(
            payload_formats=frozenset({WeightPayloadFormat.FULL_TENSOR}),
            sources=frozenset({WeightSource.TRAINER}),
            artifact_type=PreparedStreamingTensors if bounded else PreparedEngineTensors,
        )

    def prepare(self, *, version, source: ResolvedSource) -> PreparedArtifact:
        if self._active is not None:
            raise RuntimeError("release the active update before preparing another")
        if not isinstance(source, ResolvedTrainerSource):
            raise TypeError("load-time tensor method requires a trainer source")
        is_streaming = self._transfer.bounded_staging
        with refit_span("transfer_planning", accumulate_metadata=True) as counters:
            prepared = self._transfer.prepare(
                trainer_snapshot=source.snapshot,
                checksums=source.checksums,
            )
            counters.update(prepared.metrics)
        set_refit_cold(not bool(prepared.metrics.get("plan_cache_hits")))
        if is_streaming:
            metrics = dict(prepared.metrics)
            streamed = PreparedStreamingTensors(
                _batch_factory=lambda: self._transfer.iter_bounded(prepared, metrics),
                replay_id=prepared.replay_id,
                transfer_metrics=metrics,
            )
            self._active = streamed
            return streamed
        staged = self._transfer.stage(prepared)
        _attribute_transfer(staged.metrics)
        self._active = PreparedEngineTensors(staged=staged)
        return self._active

    def recover_preparation(self, error: BaseException) -> None:
        if not self._transfer.bounded_staging:
            super().recover_preparation(error)

    @contextmanager
    def installation_context(self, prepared: PreparedArtifact) -> Iterator[None]:
        if prepared is not self._active:
            raise RuntimeError("update does not own the active source")
        self.validate_release(prepared)
        try:
            yield
        finally:
            if isinstance(prepared, PreparedStreamingTensors):
                _attribute_transfer(prepared.transfer_metrics)

    def can_release(self, prepared: PreparedArtifact) -> bool:
        return (
            not isinstance(prepared, PreparedStreamingTensors)
            or not prepared.release_blocked
        )

    def release(self, prepared: PreparedArtifact) -> None:
        if prepared is not self._active:
            raise RuntimeError("update is no longer active")
        self.validate_release(prepared)
        self._active = None

    def validate_close(self) -> None:
        if self._active is not None:
            self.validate_release(self._active)

    def close(self) -> None:
        self.validate_close()
        self._active = None
        self._transfer.close()


def _attribute_transfer(metrics: dict[str, float]) -> None:
    add_refit_bytes(metrics.get("bytes_received", 0))
    if "wire_s" in metrics:
        add_refit_duration("wire_transfer", metrics["wire_s"])
    if "reconstruct_s" in metrics:
        add_refit_duration("receive_sync", metrics["reconstruct_s"])


__all__ = ["LoadTimeTensorNixlUpdateMethod"]
