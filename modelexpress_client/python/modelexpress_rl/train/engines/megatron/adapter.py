# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Megatron implementation of the trainer-engine adapter contract."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch.distributed as dist

from modelexpress_rl.train.adapter import (
    CompletionFence,
    NixlMetadataProvider,
    StagedWeightVersionShardData,
    TrainerEngineAdapter,
    TrainerStagingMode,
    WeightPayloadFormat,
    WeightVersionShardManifest,
)

from .aliases import MegatronTensorSpec, build_hf_aliases
from .publisher import build_megatron_reshard_manifest


class MegatronTrainerAdapter(TrainerEngineAdapter):
    """Expose existing Megatron/NIXL buffers through the trainer contract."""

    def __init__(
        self,
        *,
        manager: NixlMetadataProvider,
        nixl_metadata_endpoint: str,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Megatron distributed process group is not initialized")
        self._manager = manager
        self._nixl_metadata_endpoint = nixl_metadata_endpoint
        self._source_slot_id: str | None = None
        self._registered_addrs: dict[str, int] | None = None

    @property
    def source_slot_id(self) -> str:
        """Return the logical trainer contribution represented by this rank."""
        if self._source_slot_id is None:
            raise RuntimeError("bind_tensors() must be called before source_slot_id")
        return self._source_slot_id

    def bind_tensors(self, tensors: Any) -> str:
        """Bind one logical model partition independently of its DP replica."""
        if not isinstance(tensors, list) or not all(
            isinstance(item, MegatronTensorSpec) for item in tensors
        ):
            raise TypeError("tensors must be a list of MegatronTensorSpec")
        layout = [
            {
                "name": item.name,
                "role": item.role,
                "hf_names": item.hf_names,
                "global_shape": item.global_shape,
                "placement_kind": item.placement_kind,
                "shard_axis": item.shard_axis,
                "local_shard_range": item.local_shard_range,
                "extras": item.extras,
            }
            for item in sorted(tensors, key=lambda item: item.name)
        ]
        digest = hashlib.sha256(
            json.dumps(layout, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        source_slot_id = f"megatron:partition:{digest}"
        if self._source_slot_id is not None and self._source_slot_id != source_slot_id:
            raise RuntimeError("Megatron logical tensor partition changed after binding")
        self._source_slot_id = source_slot_id
        return source_slot_id

    @property
    def supported_staging_modes(self) -> frozenset[TrainerStagingMode]:
        return frozenset({TrainerStagingMode.IN_PLACE})

    @property
    def supported_payload_formats(self) -> frozenset[WeightPayloadFormat]:
        return frozenset({WeightPayloadFormat.FULL_TENSOR})

    def stage_shard(
        self,
        *,
        tensors: Any,
        staging_mode: TrainerStagingMode,
        payload_format: WeightPayloadFormat,
    ) -> StagedWeightVersionShardData:
        """Capture the current pre-registered Megatron source buffers."""
        if staging_mode not in self.supported_staging_modes:
            raise NotImplementedError(
                f"MegatronTrainerAdapter does not support {staging_mode.value} staging"
            )
        if payload_format not in self.supported_payload_formats:
            raise NotImplementedError(
                f"MegatronTrainerAdapter does not support {payload_format.value} payloads"
            )
        if not isinstance(tensors, list) or not all(
            isinstance(item, MegatronTensorSpec) for item in tensors
        ):
            raise TypeError("tensors must be a list of MegatronTensorSpec")
        sources = {item.name: item.tensor for item in tensors}
        if len(sources) != len(tensors):
            raise ValueError("Megatron tensor names must be unique within this rank")
        addresses = {name: tensor.data_ptr() for name, tensor in sources.items()}
        if self._registered_addrs is None:
            self._manager.register_tensors(sources)
            self._registered_addrs = addresses
        elif addresses != self._registered_addrs:
            raise RuntimeError(
                "Megatron source storage changed after NIXL registration; "
                "IN_PLACE requires stable tensor addresses"
            )
        published = build_hf_aliases(
            tensors,
            agent_name=str(self._manager.agent_name),
        )
        manifest = build_megatron_reshard_manifest(
            manager=self._manager,
            published=published,
            metadata_endpoint=self._nixl_metadata_endpoint,
        )
        total_bytes = sum(
            math.prod(shard.shape) * tensor.elsize
            for tensor in manifest.tensors
            for shard in tensor.shards
        )

        return StagedWeightVersionShardData(
            manifest=WeightVersionShardManifest(
                data=manifest.blob,
                tensor_count=len(manifest.tensors),
                total_bytes=total_bytes,
                transport="NIXL",
            ),
            # IN_PLACE performs no asynchronous copy, so the shard is ready to
            # publish as soon as its manifest has been built.
            publish_ready=CompletionFence(lambda: None),
            # IN_PLACE borrows these live tensor objects until the version is
            # retired; retaining them here makes that ownership explicit.
            buffer_owner=tuple(tensors),
        )


__all__ = ["MegatronTrainerAdapter"]
