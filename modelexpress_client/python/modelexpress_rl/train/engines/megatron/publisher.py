# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish registered native Megatron tensors for RL refit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modelexpress.refit.reshard.rendezvous import (
    MxReshardRendezvous,
    PublishedTensor,
    wrap_rendezvous_blob,
)
from modelexpress_rl.train.adapter import NixlMetadataProvider

from .aliases import MegatronTensorSpec, build_hf_aliases


@dataclass(frozen=True)
class MegatronReshardManifest:
    """Serialized manifest and the tensor records used to build it."""

    blob: bytes
    tensors: tuple[PublishedTensor, ...]


@dataclass(frozen=True)
class MegatronPublishedTensorSpec:
    """Legacy one-to-one native tensor publication descriptor."""

    name: str
    global_shape: tuple[int, ...]
    shard_axis: int | None = None
    local_shard_range: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not self.global_shape or any(int(dim) <= 0 for dim in self.global_shape):
            raise ValueError(f"{self.name}: invalid global shape {self.global_shape}")
        if (self.shard_axis is None) != (self.local_shard_range is None):
            raise ValueError(
                f"{self.name}: shard_axis and local_shard_range must be set together"
            )


def build_megatron_reshard_manifest(
    *,
    manager: NixlMetadataProvider,
    published: list[PublishedTensor],
    metadata_endpoint: str,
) -> MegatronReshardManifest:
    """Describe already-registered Megatron tensors without publishing them."""

    if not metadata_endpoint or ":" not in metadata_endpoint:
        raise ValueError(
            "metadata_endpoint must be an explicit host:port reachable by receivers"
        )
    agent_name = str(manager.agent_name)
    if not published:
        raise ValueError("published tensor list must not be empty")
    names = set()
    for tensor in published:
        if tensor.name in names:
            raise ValueError(f"duplicate published tensor {tensor.name!r}")
        names.add(tensor.name)
        if not tensor.shards:
            raise ValueError(f"{tensor.name}: no shards were published")
        for shard in tensor.shards:
            if shard.agent_name != agent_name:
                raise ValueError(
                    f"{tensor.name}: shard agent {shard.agent_name!r} does not "
                    f"match manager agent {agent_name!r}"
                )
            if shard.addr <= 0:
                raise ValueError(f"{tensor.name}: shard has invalid address")

    blob = wrap_rendezvous_blob(
        manager.nixl_metadata,
        agent_name,
        metadata_endpoint,
        published,
    )
    return MegatronReshardManifest(blob=blob, tensors=tuple(published))


def publish_megatron_reshard_view(
    *,
    manager: NixlMetadataProvider,
    rendezvous: MxReshardRendezvous,
    tensors: dict[str, Any],
    specs: list[MegatronPublishedTensorSpec],
    metadata_endpoint: str,
) -> str:
    """Publish registered Megatron tensors through the existing rendezvous."""
    by_name = {spec.name: spec for spec in specs}
    if len(by_name) != len(specs):
        raise ValueError("duplicate Megatron publish spec")
    missing = sorted(set(by_name).difference(tensors))
    extra = sorted(set(tensors).difference(by_name))
    if missing or extra:
        raise ValueError(
            f"Megatron shard table/tensor mismatch: missing={missing[:10]} "
            f"extra={extra[:10]}"
        )
    items = [
        MegatronTensorSpec(
            name=name,
            tensor=tensors[name],
            role="column",
            hf_names=(name,),
            global_shape=spec.global_shape,
            placement_kind="SHARD" if spec.shard_axis is not None else "REPLICATE",
            shard_axis=spec.shard_axis,
            local_shard_range=spec.local_shard_range,
        )
        for name, spec in sorted(by_name.items())
    ]
    published = build_hf_aliases(items, agent_name=str(manager.agent_name))
    manifest = build_megatron_reshard_manifest(
        manager=manager,
        published=published,
        metadata_endpoint=metadata_endpoint,
    )
    return rendezvous.publish(manifest.blob)


def publish_registered_shard_table(
    *,
    manager: NixlMetadataProvider,
    rendezvous: MxReshardRendezvous,
    published: list[PublishedTensor],
    metadata_endpoint: str,
) -> str:
    """Publish validated aliases through the existing rendezvous."""
    manifest = build_megatron_reshard_manifest(
        manager=manager,
        published=published,
        metadata_endpoint=metadata_endpoint,
    )
    return rendezvous.publish(manifest.blob)


__all__ = [
    "MegatronPublishedTensorSpec",
    "MegatronReshardManifest",
    "build_megatron_reshard_manifest",
    "publish_megatron_reshard_view",
    "publish_registered_shard_table",
]
