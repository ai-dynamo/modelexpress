# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load-time tensor preparation over NIXL."""

from __future__ import annotations

from collections.abc import Callable

from modelexpress.refit.timing import (
    add_refit_bytes,
    add_refit_duration,
    refit_span,
    set_refit_cold,
)

from ...train import WeightPayloadFormat
from ..adapter import NixlGeneratorSource
from ..nixl_staged_transfer import (
    _NixlStagedTransfer,
    _PreparedNixlTransfer,
    _StagedNixlWeights,
)
from ..plan import (
    MethodCapabilities,
    PreparedArtifact,
    PreparedEngineTensors,
    PreparedStreamingTensors,
    ResolvedSource,
    TrainerUpdateSource,
    UpdateMethod,
    WeightSource,
)


class LoadTimeTensorNixlUpdateMethod(UpdateMethod):
    """Prepare trainer tensors in the engine's load-time layout."""

    def __init__(
        self,
        *,
        transfer: _NixlStagedTransfer,
        capture_layout: Callable,
    ) -> None:
        self._transfer = transfer
        self._capture_layout = capture_layout
        self._active_plan: _PreparedNixlTransfer | None = None
        self._active_fingerprint: tuple | None = None
        self._active_manifest_digests: tuple[str, ...] = ()
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
        del version
        if self._active_staged is not None or self._active_streamed is not None:
            raise RuntimeError("release staged weight before staging another version")
        if not isinstance(source, TrainerUpdateSource):
            raise TypeError("load-time tensor method requires a trainer source")
        inputs = source.inputs
        if any(
            not isinstance(item.transport, NixlGeneratorSource)
            for item in inputs.sources
        ):
            raise ValueError("load-time tensor method requires NIXL sources")
        reusable = (
            self._active_plan is not None
            and self._active_fingerprint == inputs.physical_fingerprint
        )
        set_refit_cold(not reusable)
        manifests = [item.transport.manifest for item in inputs.sources]
        manifest_digests = tuple(item.manifest_digest for item in inputs.sources)
        with refit_span(
            "transfer_planning",
            metadata={
                "plan_cache_hits": int(reusable),
                "plan_cache_misses": int(not reusable),
            },
            accumulate_metadata=True,
        ):
            if not reusable:
                self._active_plan = self._transfer.prepare(
                    manifests=manifests,
                    capture_layout=self._capture_layout,
                )
                self._active_fingerprint = inputs.physical_fingerprint
        if reusable and manifest_digests != self._active_manifest_digests:
            assert self._active_plan is not None
            with refit_span(
                "source_preparation",
                metadata={"manifest_refreshes": 1},
                accumulate_metadata=True,
                duration_key="manifest_refresh_s",
            ):
                self._transfer.refresh_sources(self._active_plan, manifests)
        self._active_manifest_digests = manifest_digests
        self._active_staged = self._transfer.stage(self._active_plan)
        _attribute_transfer(self._active_staged.metrics)
        return PreparedEngineTensors(staged=self._active_staged)

    def prepare_streaming(
        self,
        *,
        version,
        source: ResolvedSource,
        max_staging_bytes: int,
    ) -> PreparedStreamingTensors:
        """Prepare trainer metadata without transferring a full weight copy."""
        del version
        if self._active_staged is not None or self._active_streamed is not None:
            raise RuntimeError("release the active update before preparing another")
        if not isinstance(source, TrainerUpdateSource) or any(
            not isinstance(item.transport, NixlGeneratorSource)
            for item in source.inputs.sources
        ):
            raise ValueError("bounded staging requires NIXL trainer sources")
        # Streaming replaces the full-copy destinations, including any cached
        # descriptors into them. Invalidate before a possibly failing switch.
        self._active_plan = None
        self._active_fingerprint = None
        self._active_manifest_digests = ()
        try:
            prepared = self._transfer.prepare(
                manifests=[item.transport.manifest for item in source.inputs.sources],
                capture_layout=self._capture_layout,
                max_staging_bytes=max_staging_bytes,
            )
        except Exception:
            try:
                self._transfer.reset_workspace()
            except Exception as cleanup_error:
                raise ValueError(
                    "failed to reset streaming preparation; restart the generator engine"
                ) from cleanup_error
            raise
        metrics = dict(prepared.metrics)
        self._active_streamed = PreparedStreamingTensors(
            batches=lambda: self._transfer.iter_bounded(prepared, metrics),
            parameter_names=frozenset(
                name for batch in prepared.batches for name in batch.layouts[0]
            ),
            transfer_metrics=metrics,
        )
        return self._active_streamed

    def release(self, prepared: PreparedArtifact) -> None:
        if isinstance(prepared, PreparedStreamingTensors):
            if prepared is not self._active_streamed:
                raise RuntimeError("streaming update is no longer active")
            self._active_streamed = None
            return
        if not isinstance(prepared, PreparedEngineTensors):
            raise TypeError("load-time tensor method requires staged engine tensors")
        if prepared.staged is not self._active_staged:
            raise RuntimeError("load-time staged weight is no longer active")
        self._active_staged = None

    def close(self) -> None:
        self._active_streamed = None
        self._active_staged = None
        self._active_plan = None
        self._active_fingerprint = None
        self._active_manifest_digests = ()
        self._transfer.close()

def _attribute_transfer(metrics: dict[str, float]) -> None:
    add_refit_bytes(metrics.get("bytes_received", 0))
    if "wire_s" in metrics:
        add_refit_duration("wire_transfer", metrics["wire_s"])
    if "reconstruct_s" in metrics:
        add_refit_duration("receive_sync", metrics["reconstruct_s"])


__all__ = ["LoadTimeTensorNixlUpdateMethod"]
