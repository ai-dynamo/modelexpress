# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FSDP/DTensor implementation of the trainer-engine adapter contract.

Setup is one-time; the per-step source geometry is re-read from the state_dict
the client passes each stage, so a trainer that re-materializes its state_dict
(CPU offload, gathered state dict) still publishes the latest weights:

- COPY_TO_DEVICE (default): ``initialize`` allocates one persistent wire-dtype
  arena per shard and registers them once. Each stage snapshots the live weights
  into those stable arenas (cast to the wire dtype only when the source differs);
  ``publish_ready`` fences the async copy. Robust to a moving source because the
  registered arena never moves.
- IN_PLACE (optimization): ``initialize`` registers the DTensor local storage
  directly (contiguous, so RDMA-registerable) and serves it with no copy. Its
  premise is stable storage: the registered address must not change, so each
  stage asserts the source still sits where it was registered and fails toward
  COPY_TO_DEVICE otherwise. The source must already be the wire dtype (no
  in-place cast).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist

from modelexpress.refit.reshard.cuda_pool import classic_cuda_alloc
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
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("FSDP distributed process group is not initialized")
        self._manager = manager
        self._nixl_metadata_endpoint = nixl_metadata_endpoint
        self._source_slot_id = f"publisher:global-rank:{dist.get_rank()}"
        self._initialized = False
        self._staging_mode: TrainerStagingMode | None = None
        # name -> (global_shape, shard_offset, local_shape) fixed at initialize().
        self._expected_layout: dict[
            str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
        ] = {}
        self._arenas: dict[str, torch.Tensor] = {}  # COPY: name -> registered arena
        # name -> the address we registered (the arena for COPY, the live source
        # for IN_PLACE). The served buffer must keep sitting here.
        self._registered_addrs: dict[str, int] = {}

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
            {TrainerStagingMode.COPY_TO_DEVICE, TrainerStagingMode.IN_PLACE}
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

        if staging_mode is TrainerStagingMode.COPY_TO_DEVICE:
            self._allocate_and_register_arenas(shards)
        else:  # IN_PLACE
            self._register_sources_in_place(shards)

        self._staging_mode = staging_mode
        self._initialized = True

    def _allocate_and_register_arenas(self, shards: list[LocalTensorShard]) -> None:
        """Allocate one persistent bf16 arena per shard and register them once."""
        # TODO(staging-followups):
        # - register via a single VmmArena + register_arena (one dmabuf MR).
        # - env var to stage into CPU pinned host memory vs GPU device memory
        #   (host staging frees device memory when the GPU is tight).
        with classic_cuda_alloc():
            self._arenas = {
                s.name: torch.empty(
                    s.local_shape, dtype=WIRE_DTYPE, device=s.source_tensor.device
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
        shards = self._capture(tensors)
        self.initialize(shards=shards, staging_mode=staging_mode)
        if staging_mode is not self._staging_mode:
            raise ValueError(
                f"FSDPTrainerAdapter initialized for {self._staging_mode.value} "
                f"staging; cannot stage {staging_mode.value}"
            )
        self._require_same_layout(shards)

        if staging_mode is TrainerStagingMode.COPY_TO_DEVICE:
            publish_ready = self._snapshot_into_arenas(shards)
        else:  # IN_PLACE serves live storage; nothing to copy.
            self._require_sources_pinned(shards)
            publish_ready = CompletionFence(lambda: None)

        return self._staged(shards, publish_ready)

    def _snapshot_into_arenas(self, shards: list[LocalTensorShard]) -> CompletionFence:
        """Copy each rank-local source into its persistent registered arena.

        ``copy_`` casts to bf16 only when the source dtype differs. The arena is
        the served buffer, so point each shard at it.
        """
        stream = torch.cuda.current_stream() if torch.cuda.is_available() else None
        for shard in shards:
            arena = self._arenas[shard.name]
            arena.copy_(shard.source_tensor)
            shard.staging_tensor = arena
        if stream is not None:
            done = torch.cuda.Event()
            done.record(stream)
            return CompletionFence(done.synchronize)
        return CompletionFence(lambda: None)

    def _require_sources_pinned(self, shards: list[LocalTensorShard]) -> None:
        """Fail unless every source still sits where it was registered.

        IN_PLACE publishes the registered address, so a moved source would
        advertise stale (freed or reused) memory. Fail toward COPY_TO_DEVICE.
        """
        for shard in shards:
            self._require_in_place_servable(shard)
            if shard.source_tensor.data_ptr() != self._registered_addrs[shard.name]:
                raise NotImplementedError(
                    f"{shard.name}: source storage moved since registration; "
                    "IN_PLACE requires stable storage, use COPY_TO_DEVICE"
                )

    def _capture(self, tensors: Any) -> list[LocalTensorShard]:
        if not isinstance(tensors, dict):
            raise TypeError("tensors must be an FSDP state_dict (dict[str, Tensor])")
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

    @staticmethod
    def _require_in_place_servable(shard: LocalTensorShard) -> None:
        if shard.source_tensor.dtype != WIRE_DTYPE:
            raise NotImplementedError(
                f"{shard.name}: IN_PLACE serves the source dtype but wire is "
                f"{WIRE_DTYPE}; use COPY_TO_DEVICE to cast"
            )
        if not shard.source_tensor.is_contiguous():
            raise NotImplementedError(
                f"{shard.name}: IN_PLACE requires a contiguous local shard; "
                "use COPY_TO_DEVICE for this tensor"
            )

    def _staged(
        self, shards: list[LocalTensorShard], publish_ready: CompletionFence
    ) -> StagedWeightVersionShardData:
        blob = build_fsdp_reshard_manifest(
            manager=self._manager,
            shards=shards,
            metadata_endpoint=self._nixl_metadata_endpoint,
        )
        wire_elsize = torch.empty((), dtype=WIRE_DTYPE).element_size()
        total_bytes = sum(math.prod(s.local_shape) * wire_elsize for s in shards)
        return StagedWeightVersionShardData(
            manifest=WeightVersionShardManifest(
                data=blob,
                tensor_count=len({s.name for s in shards}),
                total_bytes=total_bytes,
                transport="NIXL",
            ),
            publish_ready=publish_ready,
            # Keep the served buffers alive while the version can be selected.
            buffer_owner=tuple(s.served_tensor for s in shards),
        )


__all__ = ["FSDPTrainerAdapter"]
