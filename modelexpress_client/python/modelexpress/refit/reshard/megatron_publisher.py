# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish registered native Megatron tensors through #496 rendezvous."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modelexpress.refit.reshard.rendezvous import (
    MxReshardRendezvous,
    PublishedShard,
    PublishedTensor,
    wrap_rendezvous_blob,
)
from modelexpress.refit.reshard.verify import VERIFY, tensor_digest


@dataclass(frozen=True)
class MegatronPublishedTensorSpec:
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


def publish_megatron_reshard_view(
    *,
    manager: Any,
    client: Any,
    model_name: str,
    worker_rank: int,
    worker_id: str,
    tensors: dict[str, Any],
    specs: list[MegatronPublishedTensorSpec],
    metadata_endpoint: str,
) -> str:
    """Publish a #496 shard table over an existing NIXL registration.

    The framework's normal publisher owns tensor lifetime, registration, and
    listen thread. This seam only describes those same stable addresses in the
    #496 rendezvous format, avoiding a duplicate NIXL agent or second tensor
    allocation.
    """

    if not metadata_endpoint or ":" not in metadata_endpoint:
        raise ValueError(
            "metadata_endpoint must be an explicit host:port reachable by receivers"
        )
    by_name = {spec.name: spec for spec in specs}
    missing = sorted(set(by_name).difference(tensors))
    extra = sorted(set(tensors).difference(by_name))
    if missing or extra:
        raise ValueError(
            f"Megatron shard table/tensor mismatch: missing={missing[:10]} "
            f"extra={extra[:10]}"
        )

    agent_name = str(manager.agent_name)
    published = []
    for name in sorted(by_name):
        tensor = tensors[name]
        spec = by_name[name]
        if not tensor.is_contiguous():
            raise ValueError(f"{name}: #496 publication requires contiguous storage")
        local_shape = tuple(int(dim) for dim in tensor.shape)
        if len(local_shape) != len(spec.global_shape):
            raise ValueError(
                f"{name}: local rank {len(local_shape)} != global rank "
                f"{len(spec.global_shape)}"
            )
        offset = [0] * len(local_shape)
        if spec.shard_axis is not None:
            axis = int(spec.shard_axis)
            if not 0 <= axis < len(local_shape):
                raise ValueError(f"{name}: invalid shard axis {axis}")
            assert spec.local_shard_range is not None
            lo, hi = (int(value) for value in spec.local_shard_range)
            if hi - lo != local_shape[axis] or hi > int(spec.global_shape[axis]):
                raise ValueError(
                    f"{name}: local shape {local_shape} disagrees with shard "
                    f"range {(lo, hi)} in global shape {spec.global_shape}"
                )
            offset[axis] = lo
        elif local_shape != tuple(spec.global_shape):
            raise ValueError(
                f"{name}: replicated local shape {local_shape} != "
                f"global shape {spec.global_shape}"
            )
        published.append(
            PublishedTensor(
                name=name,
                dtype=str(tensor.dtype),
                elsize=int(tensor.element_size()),
                full_shape=tuple(spec.global_shape),
                shards=[
                    PublishedShard(
                        agent_name=agent_name,
                        device_id=int(tensor.device.index or 0),
                        addr=int(tensor.data_ptr()),
                        shard_offset=tuple(offset),
                        shape=local_shape,
                        # Digest of the bytes a receiver will read from this
                        # address, so the receiver can prove the wire delivered
                        # this rank's parameters rather than merely plausible
                        # timings. Publishing it costs a reduction per tensor, so
                        # it is gated rather than always on.
                        digest=tensor_digest(tensor) if VERIFY else None,
                    )
                ],
            )
        )

    return publish_registered_shard_table(
        manager=manager,
        client=client,
        model_name=model_name,
        worker_rank=worker_rank,
        worker_id=worker_id,
        published=published,
        metadata_endpoint=metadata_endpoint,
    )


def publish_registered_shard_table(
    *,
    manager: Any,
    client: Any,
    model_name: str,
    worker_rank: int,
    worker_id: str,
    published: list[PublishedTensor],
    metadata_endpoint: str,
) -> str:
    """Publish a validated alias table over already-registered storage."""

    if not metadata_endpoint or ":" not in metadata_endpoint:
        raise ValueError(
            "metadata_endpoint must be an explicit host:port reachable by receivers"
        )
    if not published:
        raise ValueError("published shard table must not be empty")
    agent_name = str(manager.agent_name)
    for tensor in published:
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
    rendezvous = MxReshardRendezvous(
        client,
        role="trainer",
        rank=int(worker_rank),
        model_name=model_name,
        worker_id=worker_id,
    )
    return rendezvous.publish(blob)


__all__ = [
    "MegatronPublishedTensorSpec",
    "publish_megatron_reshard_view",
    "publish_registered_shard_table",
]
