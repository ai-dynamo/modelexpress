# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FSDP/DTensor implementation of the trainer-engine adapter contract.

Setup is one-time; the per-step source geometry is re-read from the state_dict
the client passes each stage, so a trainer that re-materializes its state_dict
(CPU offload, gathered state dict) still publishes the latest weights:

- IN_PLACE (recommended for synchronous updates): registers DTensor local storage
  directly (contiguous, so RDMA-registerable) and serves it with no copy. Its
  premise is stable storage: the registered address must not change, so each
  stage asserts the source still sits where it was registered and fails toward
  COPY_TO_HOST otherwise. Sources must remain immutable until version retirement
  and already match the wire dtype; trainer-side conversion is incompatible.
- COPY_TO_HOST (recommended when IN_PLACE is unavailable): allocates persistent
  pinned CPU buffers in the wire dtype and registers them once. Each stage copies
  the current shards into those buffers; publish_ready fences CUDA copies.
- COPY_TO_DEVICE: retains an additional wire-format copy on the source device.
  This can reduce latency compared with host staging, but its VRAM cost makes it
  an explicit choice for exceptional cases such as small models.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist

from modelexpress import envs as mx_envs
from modelexpress.refit.reshard.cuda_pool import classic_cuda_alloc
from modelexpress.refit.timing import (
    add_refit_duration,
    add_refit_metadata,
    refit_span,
)
from modelexpress_rl.train.adapter import (
    CompletionFence,
    NixlMetadataProvider,
    StagedWeightVersionShardData,
    TrainerEngineAdapter,
    TrainerStagingMode,
    WeightPayloadFormat,
    WeightVersionShardManifest,
)

from .publisher import (
    WIRE_DTYPE,
    LocalTensorShard,
    build_fsdp_reshard_manifest,
    capture_local_shards,
)


class FSDPTrainerAdapter(TrainerEngineAdapter):
    """Expose FSDP/DTensor state-dict shards through the trainer contract.

    ``initialize`` fixes the shard layout and registers the source buffers once;
    ``stage_shard`` re-reads the rank-local views each step and either snapshots
    them into the persistent arenas (COPY) or serves them in place (IN_PLACE).
    """

    def __init__(
        self,
        *,
        manager: NixlMetadataProvider,
        nixl_metadata_endpoint: str,
        wire_dtype_overrides: Mapping[str, torch.dtype] | None = None,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("FSDP distributed process group is not initialized")
        self._manager = manager
        self._wire_dtype_overrides = dict(wire_dtype_overrides or {})
        for name, dtype in self._wire_dtype_overrides.items():
            if not isinstance(name, str) or not name:
                raise ValueError("wire dtype overrides require non-empty tensor names")
            if dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError(f"unsupported wire dtype {dtype!r} for {name!r}")
        self._nixl_metadata_endpoint = nixl_metadata_endpoint
        self._source_slot_id = f"publisher:global-rank:{dist.get_rank()}"
        self._initialized = False
        self._staging_mode: TrainerStagingMode | None = None
        # name -> (global_shape, shard_offset, local_shape) fixed at initialize().
        self._expected_layout: dict[
            str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
        ] = {}
        self._expected_source_dtypes: dict[str, torch.dtype] = {}
        self._arenas: dict[str, torch.Tensor] = {}  # COPY: name -> registered arena
        # name -> the address we registered (the arena for COPY, the live source
        # for IN_PLACE). The served buffer must keep sitting here.
        self._registered_addrs: dict[str, int] = {}
        self._manifest: WeightVersionShardManifest | None = None

    @property
    def source_slot_id(self) -> str:
        return self._source_slot_id

    def bind_tensors(self, tensors: Any) -> str:
        """Validate the local state dict and bind it to this global-rank slot."""
        self._capture(tensors)
        return self.source_slot_id

    @property
    def supported_staging_modes(self) -> frozenset[TrainerStagingMode]:
        return frozenset(
            {
                TrainerStagingMode.IN_PLACE,
                TrainerStagingMode.COPY_TO_HOST,
                TrainerStagingMode.COPY_TO_DEVICE,
            }
        )

    @property
    def supported_payload_formats(self) -> frozenset[WeightPayloadFormat]:
        # Sharding lives in the manifest; the payload is the (sharded) full tensor.
        return frozenset({WeightPayloadFormat.FULL_TENSOR})

    def initialize(
        self, *, shards: list[LocalTensorShard], staging_mode: TrainerStagingMode
    ) -> None:
        """Fix the shard layout and register the source buffers (idempotent)."""
        if self._initialized:
            return
        if staging_mode not in self.supported_staging_modes:
            raise NotImplementedError(
                f"FSDPTrainerAdapter does not support {staging_mode.value} staging"
            )
        names = frozenset(s.name for s in shards)
        if len(names) != len(shards):
            raise ValueError("FSDP shard names are not unique within this rank")
        self._expected_layout = {
            s.name: (s.global_shape, s.shard_offset, s.local_shape) for s in shards
        }
        self._expected_source_dtypes = {s.name: s.source_tensor.dtype for s in shards}

        if staging_mode is TrainerStagingMode.IN_PLACE:
            self._register_sources_in_place(shards)
        else:
            self._allocate_and_register_arenas(shards, staging_mode)

        self._staging_mode = staging_mode
        self._initialized = True

    def _allocate_and_register_arenas(
        self, shards: list[LocalTensorShard], staging_mode: TrainerStagingMode
    ) -> None:
        """Allocate one persistent arena per shard in its selected wire dtype."""
        host = staging_mode is TrainerStagingMode.COPY_TO_HOST
        if host and not torch.cuda.is_available():
            # Unpinned host memory would still be registered as DRAM and read
            # over RDMA, so decline rather than silently serve pageable pages.
            # A trainer with no CUDA has no device shards to stage from either.
            raise RuntimeError("COPY_TO_HOST staging requires CUDA for pinned memory")
        with contextlib.nullcontext() if host else classic_cuda_alloc():
            self._arenas = {
                s.name: torch.empty(
                    s.local_shape,
                    dtype=self._wire_dtype(s),
                    device="cpu" if host else s.source_tensor.device,
                    pin_memory=host,
                )
                for s in shards
            }
        self._manager.register_tensors(
            {
                self._register_key(i, s.name): self._arenas[s.name]
                for i, s in enumerate(shards)
            }
        )
        self._registered_addrs = {
            name: arena.data_ptr() for name, arena in self._arenas.items()
        }

    def _register_sources_in_place(self, shards: list[LocalTensorShard]) -> None:
        """Register the live local storage as the served buffer (no copy)."""
        for shard in shards:
            self._require_in_place_servable(shard)
        self._manager.register_tensors(
            {
                self._register_key(i, s.name): s.source_tensor
                for i, s in enumerate(shards)
            }
        )
        self._registered_addrs = {s.name: s.source_tensor.data_ptr() for s in shards}

    def stage_shard(
        self,
        *,
        tensors: Any,
        staging_mode: TrainerStagingMode,
        payload_format: WeightPayloadFormat,
    ) -> StagedWeightVersionShardData:
        """Capture one immutable, rank-local FSDP version shard."""
        if staging_mode not in self.supported_staging_modes:
            raise NotImplementedError(
                f"FSDPTrainerAdapter does not support {staging_mode.value} staging"
            )
        if payload_format not in self.supported_payload_formats:
            raise NotImplementedError(
                f"FSDPTrainerAdapter does not support {payload_format.value} payloads"
            )

        # Re-read the rank-local views from THIS step's state_dict so a
        # re-materialized source still publishes the latest weights; these same
        # shards seed the one-time setup on the first stage (single capture).
        with refit_span(
            "source_preparation",
            metadata={"shard_captures": 1},
            accumulate_metadata=True,
            duration_key="shard_capture_s",
        ):
            shards = self._capture(tensors)
        # Charged only on the first stage. Later calls are a no-op, and a stage
        # reporting near-zero registration would read as though registering
        # were free rather than already done.
        registration = (
            contextlib.nullcontext()
            if self._initialized
            else refit_span(
                "setup_registration",
                metadata={"trainer_registrations": 1},
                accumulate_metadata=True,
                duration_key="trainer_registration_s",
            )
        )
        with registration:
            self.initialize(shards=shards, staging_mode=staging_mode)
        if staging_mode is not self._staging_mode:
            raise ValueError(
                f"FSDPTrainerAdapter initialized for {self._staging_mode.value} "
                f"staging; cannot stage {staging_mode.value}"
            )
        with refit_span(
            "source_preparation",
            metadata={"layout_validations": 1},
            accumulate_metadata=True,
            duration_key="layout_validation_s",
        ):
            self._require_same_layout(shards)

        if staging_mode is not TrainerStagingMode.IN_PLACE:
            with refit_span(
                "source_preparation",
                metadata={"staging_copy_enqueues": 1},
                accumulate_metadata=True,
                duration_key="staging_copy_enqueue_s",
            ):
                publish_ready = self._snapshot_into_arenas(shards)
        else:  # IN_PLACE serves live storage; nothing to copy.
            self._require_sources_pinned(shards)
            publish_ready = CompletionFence(lambda: None)

        if mx_envs.MX_RESHARD_PUBLISH_DIGEST:
            publish_ready.wait()
        return self._staged(shards, publish_ready)

    def _snapshot_into_arenas(self, shards: list[LocalTensorShard]) -> CompletionFence:
        """Copy each rank-local source into its persistent registered arena.

        ``copy_`` casts to the selected wire dtype when the source dtype differs. The arena is
        the served buffer, so point each shard at it.
        """
        devices = set()
        for shard in shards:
            arena = self._arenas[shard.name]
            arena.copy_(shard.source_tensor, non_blocking=True)
            shard.staging_tensor = arena
            devices.update(t.device for t in (arena, shard.source_tensor) if t.is_cuda)
        events = []
        for device in sorted(devices, key=str):
            done = torch.cuda.Event()
            done.record(torch.cuda.current_stream(device))
            events.append(done)

        def wait() -> None:
            for event in events:
                event.synchronize()

        return CompletionFence(wait)

    def _require_sources_pinned(self, shards: list[LocalTensorShard]) -> None:
        """Fail unless every source still sits where it was registered.

        IN_PLACE publishes the registered address, so a moved source would
        advertise stale (freed or reused) memory. Recommend COPY_TO_HOST.
        """
        for shard in shards:
            self._require_in_place_servable(shard)
            if shard.source_tensor.data_ptr() != self._registered_addrs[shard.name]:
                raise NotImplementedError(
                    f"{shard.name}: source storage moved since registration; "
                    "IN_PLACE requires stable storage; use COPY_TO_HOST"
                )

    def _capture(self, tensors: Any) -> list[LocalTensorShard]:
        if not isinstance(tensors, dict):
            raise TypeError("tensors must be an FSDP state_dict (dict[str, Tensor])")
        unknown = self._wire_dtype_overrides.keys() - tensors.keys()
        if unknown:
            raise ValueError(
                f"wire dtype override names absent from state_dict: {sorted(unknown)}"
            )
        for name in self._wire_dtype_overrides:
            if not tensors[name].is_floating_point():
                raise ValueError(
                    f"wire dtype override requires a floating state_dict tensor: {name}"
                )
        shards = capture_local_shards(tensors)
        if not shards:
            raise ValueError("no local FSDP shards to publish")
        return shards

    def _require_same_layout(self, shards: list[LocalTensorShard]) -> None:
        expected_names = frozenset(self._expected_layout)
        names = frozenset(s.name for s in shards)
        if names != expected_names:
            missing = sorted(expected_names - names)
            extra = sorted(names - expected_names)
            raise ValueError(
                "FSDP tensor set changed since initialize "
                f"(missing={missing[:5]} extra={extra[:5]})"
            )
        for shard in shards:
            if shard.source_tensor.dtype != self._expected_source_dtypes[shard.name]:
                raise ValueError(f"{shard.name}: source dtype changed since initialize")
            layout = (shard.global_shape, shard.shard_offset, shard.local_shape)
            if layout != self._expected_layout[shard.name]:
                raise ValueError(
                    f"{shard.name}: shard geometry changed since initialize "
                    f"(was {self._expected_layout[shard.name]}, now {layout}); "
                    "a trainer must keep a fixed shard layout across steps"
                )

    @staticmethod
    def _register_key(index: int, name: str) -> str:
        return f"__pub__{index}__{name}"

    def _wire_dtype(self, shard: LocalTensorShard) -> torch.dtype:
        return self._wire_dtype_overrides.get(shard.name, WIRE_DTYPE)

    def _require_in_place_servable(self, shard: LocalTensorShard) -> None:
        dtype = self._wire_dtype(shard)
        if shard.source_tensor.dtype != dtype:
            raise NotImplementedError(
                f"{shard.name}: IN_PLACE serves the source dtype but wire is "
                f"{dtype}; use COPY_TO_HOST to cast"
            )
        if not shard.source_tensor.is_contiguous():
            raise NotImplementedError(
                f"{shard.name}: IN_PLACE requires a contiguous local shard; "
                "use COPY_TO_HOST for this tensor"
            )

    def _staged(
        self, shards: list[LocalTensorShard], publish_ready: CompletionFence
    ) -> StagedWeightVersionShardData:
        cache_hit = self._manifest is not None and not mx_envs.MX_RESHARD_PUBLISH_DIGEST
        if not cache_hit:
            manifest_metrics: dict[str, int | float] = {}
            blob = build_fsdp_reshard_manifest(
                manager=self._manager,
                shards=shards,
                metadata_endpoint=self._nixl_metadata_endpoint,
                metrics=manifest_metrics,
            )
            # Per-shard element size, since wire dtype overrides mean shards no
            # longer share one wire width.
            total_bytes = sum(
                math.prod(s.local_shape) * s.served_tensor.element_size()
                for s in shards
            )
            self._manifest = WeightVersionShardManifest(
                data=blob,
                tensor_count=len({shard.name for shard in shards}),
                total_bytes=total_bytes,
                transport="NIXL",
            )
            # Measured by the publisher while it built the blob, which is the
            # only place the generation and serialization halves are separable.
            for name in ("manifest_generation_s", "manifest_serialization_s"):
                add_refit_duration(
                    "source_preparation",
                    float(manifest_metrics[name]),
                    metadata={name: float(manifest_metrics[name])},
                )
        assert self._manifest is not None
        add_refit_metadata(
            "source_preparation",
            {
                "manifest_bytes": len(self._manifest.data),
                "manifest_cache_hit": cache_hit,
                "manifest_tensor_count": self._manifest.tensor_count,
            },
        )
        return StagedWeightVersionShardData(
            manifest=self._manifest,
            publish_ready=publish_ready,
            # Keep the served buffers alive while the version can be selected.
            buffer_owner=tuple(s.served_tensor for s in shards),
        )


__all__ = ["FSDPTrainerAdapter"]
