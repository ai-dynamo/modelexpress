# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-to-rank publishing helper — the no-allgather contract.

Wraps :class:`MxTrainingPublisher` (v1 fat client) with a thin façade
that publishes only **this rank's local shard** of each tensor, along
with the :class:`SliceOwnership` metadata the receiver-side planner
needs to compute the right cross-rank intersections.

The headline difference vs the v1 publisher's :meth:`publish_weights`
method: that method expects pre-materialized full tensors (the trainer
gathers them itself before publishing). This wrapper expects DTensor
inputs and calls ``.to_local()`` on each one — no gather happens.

For Megatron-Core (non-DTensor) inputs the wrapper accepts an
explicit ``(local_tensor, placement_descriptor)`` tuple per tensor;
the placement_descriptor carries the same axis + range info DTensor
exposes via ``.placements``.

The receiver side of the contract is in :mod:`modelexpress.rl_reshard_planner`
(planner) and :meth:`MxRefitReceiver.receive_segment` (data plane). Existing
v1 ``MxTrainingPublisher.publish_weights`` callers are unaffected — this is
a sibling surface, not a replacement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .rl_slice_descriptors import (
    PlacementKind,
    QuantizationScope,
    SliceOwnership,
)

logger = logging.getLogger("modelexpress.rank_local_publisher")


@dataclass(frozen=True)
class PlacementDescriptor:
    """Explicit placement info for tensors that aren't DTensors.

    Used by Megatron-Core publishers (which have their own sharding
    abstractions distinct from DTensor). DTensor inputs auto-derive
    this from ``tensor.placements`` and ``tensor.device_mesh`` so
    callers don't typically construct this directly.

    Args:
        placement_kind: see :data:`PlacementKind` —
            ``REPLICATE`` / ``SHARD`` / ``PARTIAL``.
        shard_axis: which axis is sharded (when ``SHARD``).
        local_shard_range: (start, end) along ``shard_axis`` this rank
            owns. End exclusive. Required for ``SHARD``.
        global_shape: full un-sharded shape of the tensor across all
            ranks. Required for ``SHARD`` (so the receiver can validate
            its requests don't exceed bounds).
    """

    placement_kind: PlacementKind
    global_shape: tuple[int, ...]
    shard_axis: int | None = None
    local_shard_range: tuple[int, int] | None = None


def _placement_from_dtensor(tensor: Any) -> PlacementDescriptor:
    """Read placement info from a DTensor.

    Import torch.distributed lazily so this module loads without torch
    (for the planner-only unit tests).
    """
    # Lazy import — we only need this when actually publishing on a GPU.
    from torch.distributed.tensor import DTensor, Replicate, Shard  # noqa: F401

    if not isinstance(tensor, DTensor):
        raise TypeError(
            "_placement_from_dtensor: expected a DTensor, got "
            f"{type(tensor).__name__}. For non-DTensor inputs pass an "
            "explicit PlacementDescriptor."
        )

    global_shape = tuple(tensor.shape)
    placements = tensor.placements

    if len(placements) != 1:
        # Multi-mesh-dim DTensor (e.g. FSDP+TP). For now we treat the
        # first sharded dimension as authoritative; receivers compute
        # the effective shard range across all mesh dims from the
        # mesh metadata. Out of scope for this prototype.
        raise NotImplementedError(
            "_placement_from_dtensor: multi-dim placements not yet "
            "supported in the prototype; got placements="
            f"{placements!r}. Pass an explicit PlacementDescriptor."
        )

    p = placements[0]
    if isinstance(p, Replicate):
        return PlacementDescriptor(
            placement_kind="REPLICATE",
            global_shape=global_shape,
        )
    if isinstance(p, Shard):
        shard_axis = int(p.dim)
        mesh = tensor.device_mesh
        rank = mesh.get_local_rank() if mesh is not None else 0
        world = mesh.size() if mesh is not None else 1
        dim_size = global_shape[shard_axis]
        # Even sharding only for now — DTensor's actual API has uneven
        # sharding support via `Shard.dim` + the partition map but for
        # the prototype the contract is "trainer mesh sizes evenly into
        # the dim". This matches the validated cases.
        shard_size = dim_size // world
        lo = rank * shard_size
        hi = (rank + 1) * shard_size if rank < world - 1 else dim_size
        return PlacementDescriptor(
            placement_kind="SHARD",
            global_shape=global_shape,
            shard_axis=shard_axis,
            local_shard_range=(lo, hi),
        )
    raise NotImplementedError(
        f"_placement_from_dtensor: unsupported placement {type(p).__name__}"
    )


class RankLocalPublisher:
    """Publishes per-rank local shards via the v1 MxTrainingPublisher.

    Holds a reference to a long-lived :class:`MxTrainingPublisher` (one
    per trainer rank), and offers two entry points the verl checkpoint
    engine + the NemoRL DTensor worker hook into:

    - :meth:`add_dtensor` — DTensor input, auto-derives placement
      from ``.placements`` and ``.device_mesh``. No gather.
    - :meth:`add_explicit_shard` — for Megatron-Core or other
      non-DTensor frameworks. Caller provides the local tensor + an
      explicit :class:`PlacementDescriptor`. No gather.

    Both methods record the ``SliceOwnership`` for receivers to plan
    against. :meth:`publish` then flushes everything to the catalog +
    NIXL-registers each local-shard buffer.

    Args:
        publisher: the v1 :class:`MxTrainingPublisher` (already
            ``initialize()``-ed and ready to accept tensors).
        model_name: identifier shared across trainer + receivers.
        worker_rank: this rank's index. Carried into every published
            :class:`SliceOwnership` so receivers can do same-rank routing.
    """

    def __init__(
        self,
        publisher: Any,  # MxTrainingPublisher (avoid circular import)
        *,
        model_name: str,
        worker_rank: int,
    ) -> None:
        self._publisher = publisher
        self._model_name = model_name
        self._worker_rank = worker_rank

        # Tensors to publish this cycle; cleared by :meth:`publish`.
        self._pending: dict[str, tuple[Any, SliceOwnership]] = {}

    # ------------------------------------------------------------------
    # Add methods (no IO; pure metadata + reference capture)
    # ------------------------------------------------------------------

    def add_dtensor(
        self,
        name: str,
        tensor: Any,
        *,
        compile_target: str = "bf16_cast",
        compile_metadata: dict[str, object] | None = None,
        quantization_scope: QuantizationScope = "absent",
    ) -> None:
        """Register a DTensor for rank-local publish.

        Internally calls ``tensor.to_local()`` (no allgather, no
        cross-rank comms) and builds the :class:`SliceOwnership` from
        the DTensor's placement metadata.

        MoE experts need no special handling here: publish the expert
        tensor as an ordinary shard whose ``placement`` shards the expert
        axis (the DTensor placement already expresses this).
        """
        placement = _placement_from_dtensor(tensor)
        local_tensor = tensor.to_local()
        self._record(
            name=name,
            local_tensor=local_tensor,
            placement=placement,
            compile_target=compile_target,
            compile_metadata=compile_metadata or {},
            quantization_scope=quantization_scope,
        )

    def add_explicit_shard(
        self,
        name: str,
        local_tensor: Any,
        placement: PlacementDescriptor,
        *,
        compile_target: str = "bf16_cast",
        compile_metadata: dict[str, object] | None = None,
        quantization_scope: QuantizationScope = "absent",
    ) -> None:
        """Register a non-DTensor local shard for publish.

        The caller asserts via ``placement`` that ``local_tensor`` is
        exactly the bytes this rank owns — no gather happens.

        This is the entry point for Megatron-Core (where parameters
        aren't DTensors but native parallel shards) and for tests
        that want to drive the publisher without a real torch.distributed
        process group.

        MoE experts are published as an ordinary shard: set ``placement``
        to shard the expert axis over this rank's owned-expert range (one
        call per contiguous run for non-contiguous ownership).
        """
        self._record(
            name=name,
            local_tensor=local_tensor,
            placement=placement,
            compile_target=compile_target,
            compile_metadata=compile_metadata or {},
            quantization_scope=quantization_scope,
        )

    def _record(
        self,
        *,
        name: str,
        local_tensor: Any,
        placement: PlacementDescriptor,
        compile_target: str,
        compile_metadata: dict[str, object],
        quantization_scope: QuantizationScope,
    ) -> None:
        # Derive byte_size + dtype from the local tensor.
        dtype = str(local_tensor.dtype) if hasattr(local_tensor, "dtype") else "unknown"
        byte_size = (
            local_tensor.numel() * local_tensor.element_size()
            if hasattr(local_tensor, "numel") and hasattr(local_tensor, "element_size")
            else 0
        )

        own = SliceOwnership(
            model_name=self._model_name,
            tensor_name=name,
            global_shape=placement.global_shape,
            dtype=dtype,
            placement_kind=placement.placement_kind,
            shard_axis=placement.shard_axis,
            local_shard_range=placement.local_shard_range,
            worker_rank=self._worker_rank,
            nixl_addr=0,  # populated at publish() time by the v1 client
            byte_size=byte_size,
            device_id=getattr(local_tensor, "device", None).index
            if hasattr(local_tensor, "device") and local_tensor.device.type == "cuda"
            else 0,
            compile_target=compile_target,
            compile_metadata=compile_metadata,
            quantization_scope=quantization_scope,
        )
        self._pending[name] = (local_tensor, own)

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def publish(self, *, step: int) -> str:
        """Flush all recorded shards via the v1 :class:`MxTrainingPublisher`.

        The v1 publisher's :meth:`publish_weights` accepts a
        ``named_tensors`` dict and a step counter; we pass our local
        shards through unchanged (the publisher does NIXL register +
        gRPC publish on those exact tensors, which is exactly what we
        want for the no-allgather contract).

        The :class:`SliceOwnership` entries we built up are exposed via
        :meth:`drain_slice_ownerships` so the verl checkpoint engine
        can write them into the catalog alongside the v1 metadata.

        Returns the ``mx_source_id`` the server assigns this publish.
        """
        if not self._pending:
            raise RuntimeError("RankLocalPublisher.publish(): nothing recorded")

        named_tensors = {name: tensor for name, (tensor, _own) in self._pending.items()}
        mx_source_id = self._publisher.publish_weights(
            named_tensors=named_tensors,
            step=step,
            worker_rank=self._worker_rank,
        )
        logger.info(
            "rank-local publish: rank=%d step=%d tensors=%d mx_source_id=%s",
            self._worker_rank,
            step,
            len(named_tensors),
            mx_source_id,
        )
        return mx_source_id

    def drain_slice_ownerships(self) -> list[SliceOwnership]:
        """Return + clear the pending :class:`SliceOwnership` entries.

        Called after :meth:`publish` to hand the descriptors to whatever
        component writes them into the MX catalog as the receiver-visible
        metadata. After this returns, the publisher's pending buffer is
        empty and ready for the next refit cycle.
        """
        ownerships = [own for _t, own in self._pending.values()]
        self._pending.clear()
        return ownerships

    def mark_ready(self) -> bool:
        """Forward to the underlying v1 publisher's mark_ready()."""
        return self._publisher.mark_ready(worker_rank=self._worker_rank)
