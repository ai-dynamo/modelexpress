# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract FSDP/DTensor local shards and describe them as MX reshard sources.

Extracts the rank-local shards from an FSDP ``state_dict`` and emits MX's
engine-neutral manifest (``PublishedTensor`` / ``PublishedShard`` +
``wrap_rendezvous_blob``). HF-name conversion is deliberately NOT done here: the
receiver captures how these trainer-format sources land in the vLLM param layout
(see ``modelexpress_rl/inference/reshard/fsdp``).

Extraction rules (per state_dict tensor, floating point only):
- unsharded (not a DTensor): every rank holds + publishes the full tensor;
  the identical copies across ranks are deduped by box upstream of the
  transfer planner (merge_shard_tables)
- replicated DTensor: same as unsharded
- sharded DTensor: this rank serves its per-dim local box (general: FSDP dim 0,
  tensor-parallel dim 1, or 2-D meshes) via compute_local_shape_and_global_offset
- served as the wire dtype (WIRE_DTYPE), cast into the staging arena for
  COPY_TO_DEVICE only when the source differs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from modelexpress.refit.reshard.rendezvous import (
    PublishedShard,
    PublishedTensor,
    wrap_rendezvous_blob,
)
from modelexpress.refit.reshard.verify import published_digest
from modelexpress_rl.train.adapter import NixlMetadataProvider

logger = logging.getLogger("modelexpress_rl.train.engines.fsdp.publisher")

# The dtype weights are served on the wire as. The cast only actually happens
# when the source dtype differs (e.g. an fp32 master): a matching source copies
# as-is, and IN_PLACE can serve a matching source with no copy at all.
# TODO: make this configurable at client initialization; hardcoded to bf16 for now.
WIRE_DTYPE = torch.bfloat16


@dataclass
class LocalTensorShard:
    """One rank-local source shard extracted from the FSDP state_dict.

    ``source_tensor`` is the live (or detached) rank-local view. ``shard_offset``
    is the per-dim offset of this shard's box inside the global tensor (all-zero
    for unsharded/replicated). ``staging_tensor`` is set only for COPY_TO_DEVICE
    and is the WIRE_DTYPE registered arena the source is copied into (copy_
    converts only if the source dtype differs).
    """

    name: str
    global_shape: tuple[int, ...]
    shard_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    source_tensor: torch.Tensor
    staging_tensor: torch.Tensor | None = None

    @property
    def served_tensor(self) -> torch.Tensor:
        """The tensor NIXL actually registers/serves (staging if copied, else source)."""
        return self.staging_tensor if self.staging_tensor is not None else self.source_tensor


def capture_local_shards(
    state_dict: dict[str, torch.Tensor],
) -> list[LocalTensorShard]:
    """Extract each rank's local source shards from an FSDP state_dict.

    Every rank publishes what it holds, including the tensors it holds in full
    (unsharded and replicated). Those redundant full copies are identical
    across ranks; deduping them to one shard per box is handled upstream of
    the transfer planner (merge_shard_tables), not here.
    """
    shards: list[LocalTensorShard] = []
    skipped: list[str] = []
    for name, value in state_dict.items():
        if not value.is_floating_point():
            skipped.append(name)
            continue
        full_shape = tuple(value.shape)
        zero_offset = tuple(0 for _ in full_shape)

        # TODO(dedup-staging): for the full-copy tensors below (unsharded +
        # replicated), under COPY_TO_DEVICE every rank stages a redundant copy;
        # consider staging + publishing from rank 0 only in that mode.

        # Unsharded (not a DTensor): this rank holds the full tensor; publish it.
        if not isinstance(value, DTensor):
            shards.append(
                LocalTensorShard(
                    name=name,
                    global_shape=full_shape,
                    shard_offset=zero_offset,
                    local_shape=full_shape,
                    source_tensor=value.detach(),
                )
            )
            continue

        placements = value.placements
        local_shape, global_offset = compute_local_shape_and_global_offset(
            value.shape, value.device_mesh, placements
        )
        local = value.to_local().detach()
        if tuple(local.shape) != tuple(local_shape):
            local = local[tuple(slice(size) for size in local_shape)]

        # Replicated DTensor: this rank holds the full tensor; publish it.
        if all(placement.is_replicate() for placement in placements):
            shards.append(
                LocalTensorShard(
                    name=name,
                    global_shape=full_shape,
                    shard_offset=zero_offset,
                    local_shape=tuple(local_shape),
                    source_tensor=local,
                )
            )
            continue

        # Sharded DTensor: this rank's per-dim box (general — FSDP dim 0, TP dim 1,
        # or 2-D meshes). compute_local_shape_and_global_offset gives the full
        # per-dim offset directly, including uneven splits.
        if local.numel():
            shards.append(
                LocalTensorShard(
                    name=name,
                    global_shape=full_shape,
                    shard_offset=tuple(int(off) for off in global_offset),
                    local_shape=tuple(local_shape),
                    source_tensor=local,
                )
            )
    if skipped:
        logger.debug(
            "capture_local_shards: skipped %d non-floating-point entries: %s",
            len(skipped),
            sorted(skipped),
        )
    return shards


def build_fsdp_reshard_manifest(
    *,
    manager: NixlMetadataProvider,
    shards: list[LocalTensorShard],
    metadata_endpoint: str,
) -> bytes:
    """Describe already-registered FSDP source shards as an MX manifest blob.

    Assumes ``shards`` have their ``served_tensor`` registered with ``manager``
    (addr resolved via ``data_ptr``). Groups shards by name into one
    ``PublishedTensor`` each (one shard per rank; fan-in of multiple ranks'
    shards happens on the receive/plan side via the side table).
    """
    if not metadata_endpoint or ":" not in metadata_endpoint:
        raise ValueError("metadata_endpoint must be an explicit host:port")
    agent_name = str(manager.agent_name)
    if not shards:
        raise ValueError("no local shards to publish")

    by_name: dict[str, PublishedTensor] = {}
    for shard in shards:
        served = shard.served_tensor
        if not served.is_contiguous():
            raise ValueError(f"{shard.name}: served tensor must be contiguous for RDMA")
        addr = served.data_ptr()
        if addr <= 0:
            raise ValueError(f"{shard.name}: shard has invalid address")
        published_shard = PublishedShard(
            agent_name=agent_name,
            device_id=int(served.device.index or 0),
            addr=addr,
            shard_offset=tuple(shard.shard_offset),
            shape=tuple(shard.local_shape),
            digest=published_digest(served),
        )
        tensor = by_name.get(shard.name)
        if tensor is None:
            by_name[shard.name] = PublishedTensor(
                name=shard.name,
                dtype=str(served.dtype),
                elsize=served.element_size(),
                full_shape=tuple(shard.global_shape),
                shards=[published_shard],
            )
        else:
            tensor.shards.append(published_shard)

    published = list(by_name.values())
    return wrap_rendezvous_blob(
        manager.nixl_metadata,
        agent_name,
        metadata_endpoint,
        published,
    )


__all__ = ["WIRE_DTYPE", "LocalTensorShard", "capture_local_shards", "build_fsdp_reshard_manifest"]
