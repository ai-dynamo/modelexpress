# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish registered native Megatron tensors for RL refit."""

from __future__ import annotations

from dataclasses import dataclass, replace

from modelexpress_rl.refit.reshard.rendezvous import PublishedTensor, wrap_rendezvous_blob
from modelexpress_rl.train.adapter import NixlMetadataProvider


@dataclass(frozen=True)
class MegatronReshardManifest:
    """Serialized manifest and the tensor records used to build it."""

    blob: bytes
    tensors: tuple[PublishedTensor, ...]


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

    stable_tensors = [
        replace(tensor, shards=[replace(shard, digest=None) for shard in tensor.shards])
        for tensor in published
    ]
    blob = wrap_rendezvous_blob(
        manager.nixl_metadata,
        agent_name,
        metadata_endpoint,
        stable_tensors,
    )
    return MegatronReshardManifest(blob=blob, tensors=tuple(published))


__all__ = [
    "MegatronReshardManifest",
    "build_megatron_reshard_manifest",
]
