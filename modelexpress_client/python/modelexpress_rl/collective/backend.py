# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sender and receiver halves of the NCCL M2N shard-redistribution backend.

Both sides walk the *same* plan in the *same* order. That is not a stylistic
choice: a collective requires every participant to issue an identical sequence
of operations, so a rank that skips a parameter its peers issue hangs the whole
communicator rather than failing alone. The plan order is therefore the single
source of truth on both sides, and the two classes below differ only in which
end of each transfer they own.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

from . import envs
from .comm import CommunicatorCache, LaneCommunicator, LaneKey
from .spi import LocalParamSpec, RefitCtx, resolve_specs
from .types import ParamPlan, ReshardPlan

logger = logging.getLogger("modelexpress_rl.collective.backend")

DEFAULT_LAYER_GROUP = 0


def _reshard(
    *,
    comm: LaneCommunicator,
    entry: ParamPlan,
    src: Any,
    dst: Any,
) -> None:
    """Issue one ``nccl.m2n.reshard``.

    Co-called: the trainer passes ``dst=None`` and the generator passes
    ``src=None``, and NCCL routes the many-to-many redistribution internally
    from the two meshes. Both sides still pass *both* meshes, because each has
    to know the shape of the other end to route into it.

    The argument shape is pinned to the one NeMo RL's ``xferdtensor`` uses, so
    an MX-brokered deployment and a NeMo-RL-native one issue the identical
    call: tensors and communicator positional, meshes and placements by
    keyword, meshes nested to their shape, placements as real DTensor objects,
    and ``stream`` passed as a raw handle only when there is one.
    """
    from nccl.m2n import reshard  # noqa: PLC0415 - lazy: only this path needs nccl4py

    kwargs: dict[str, Any] = {
        "src_mesh": entry.src_mesh.nested(),
        "src_placements": [p.to_dtensor() for p in entry.src_placements],
        "dst_mesh": entry.dst_mesh.nested(),
        "dst_placements": [p.to_dtensor() for p in entry.dst_placements],
    }
    stream = comm.stream
    if stream is not None:
        kwargs["stream"] = _stream_handle(stream)

    reshard(src, dst, comm.handle, **kwargs)


def _stream_handle(stream: Any) -> int:
    """Raw CUDA stream handle for the reshard op.

    A ``torch.cuda.Stream`` carries it on ``cuda_stream``; anything already
    integral is passed through so a caller can supply a handle directly.
    """
    handle = getattr(stream, "cuda_stream", stream)
    return int(handle)


class _CollectiveHalf:
    """Shared plan walking, layer grouping and lane bookkeeping."""

    def __init__(
        self,
        *,
        plan: ReshardPlan,
        specs: dict[str, LocalParamSpec],
        group_id: str,
        epoch: int,
        cache: CommunicatorCache,
    ) -> None:
        resolve_specs(plan, specs)
        self._plan = plan
        self._specs = specs
        self._group_id = group_id
        self._epoch = epoch
        self._cache = cache
        self._groups: OrderedDict[int, list[ParamPlan]] = OrderedDict()
        self.setup_layer_groups(None)
        self._pending_misc = False

    def setup_layer_groups(self, groupings: list[list[str]] | None) -> None:
        """Partition the bulk parameters into layer groups.

        Default is one group holding everything. A caller that wants to bound
        trainer memory splits it, at the cost of more wire operations.
        """
        self._groups = OrderedDict()
        if groupings is None:
            self._groups[DEFAULT_LAYER_GROUP] = list(self._plan.bulk)
            return

        by_name = {entry.name: entry for entry in self._plan.bulk}
        seen: set[str] = set()
        for group_id, names in enumerate(groupings):
            entries: list[ParamPlan] = []
            for name in names:
                if name in seen:
                    raise ValueError(f"{name} appears in more than one layer group")
                seen.add(name)
                entry = by_name.get(name)
                if entry is None:
                    raise KeyError(f"{name} is not a bulk parameter in this plan")
                entries.append(entry)
            self._groups[group_id] = entries

        uncovered = sorted(set(by_name) - seen)
        if uncovered:
            raise ValueError(
                f"layer groups leave {len(uncovered)} bulk parameter(s) uncovered: "
                f"{', '.join(uncovered[:5])}"
            )

    @property
    def layer_group_ids(self) -> list[int]:
        return list(self._groups)

    def entries(self, layer_group_id: int) -> list[ParamPlan]:
        if layer_group_id not in self._groups:
            raise KeyError(f"no layer group {layer_group_id}")
        return self._groups[layer_group_id]

    def _lane(self, lane_id: int) -> LaneCommunicator:
        key = LaneKey(group_id=self._group_id, epoch=self._epoch, lane_id=lane_id)
        lane = self._cache.get(key)
        if lane is None:
            raise RuntimeError(
                f"lane {lane_id} of group {self._group_id} has no communicator at "
                f"epoch {self._epoch}; compute_plan must run before a transfer"
            )
        return lane

    def abort(self) -> None:
        """Give up on every lane of this group at once."""
        self._cache.abort_group(self._group_id)


class NcclM2nSender(_CollectiveHalf):
    """Trainer half: supplies each parameter's local shard to the collective."""

    def start_weight_update(self, version: str) -> None:
        self._pending_misc = True
        logger.debug("collective sender starting version %s", version)

    def publish_weights(self, layer_group_id: int) -> None:
        """Issue this group's reshards. Bulk only.

        The misc broadcast deliberately does not happen here. Its communicator
        spans every rank, so it overlaps every reshard lane; entering it while
        another layer group is still resharding is two overlapping
        communicators with operations in flight in different orders, which is
        the case that deadlocks.
        """
        for entry in self.entries(layer_group_id):
            spec = self._specs[entry.name]
            ctx = spec.enter()
            _reshard(comm=self._lane(entry.partition_id), entry=entry, src=ctx.buf, dst=None)
            spec.leave(ctx)

    def finish_weight_update(self, broadcast_lane_id: int) -> None:
        """Drain every reshard lane, then broadcast the misc parameters once."""
        if not self._pending_misc:
            return
        self._pending_misc = False
        lane = self._lane(broadcast_lane_id)
        for misc in self._plan.misc:
            spec = self._specs[misc.name]
            ctx = spec.enter()
            _broadcast(lane, ctx.buf, root=0)
            spec.leave(ctx)


class NcclM2nReceiver(_CollectiveHalf):
    """Generator half: supplies each parameter's destination to the collective."""

    def start_weight_update(self, version: str) -> None:
        self._pending_misc = True
        logger.debug("collective receiver starting version %s", version)

    def update_weights(self, layer_group_id: int) -> None:
        for entry in self.entries(layer_group_id):
            spec = self._specs[entry.name]
            ctx = spec.enter()
            _reshard(comm=self._lane(entry.partition_id), entry=entry, src=None, dst=ctx.buf)
            spec.leave(ctx)

    def finish_weight_update(self, broadcast_lane_id: int) -> None:
        if not self._pending_misc:
            return
        self._pending_misc = False
        lane = self._lane(broadcast_lane_id)
        for misc in self._plan.misc:
            spec = self._specs[misc.name]
            ctx = spec.enter()
            _broadcast(lane, ctx.buf, root=0)
            spec.leave(ctx)


def _broadcast(lane: LaneCommunicator, buf: Any, *, root: int) -> None:
    """One packed-broadcast step on the all-participants lane.

    In-place: producer and consumer pass the same buffer object, so the root's
    contents land in every other rank's. Both sides walk the misc list in the
    plan's order, which is why that order is part of the plan digest.
    """
    lane.handle.broadcast(sendbuf=buf, recvbuf=buf, root=root, stream=lane.stream)


def misc_chunk_size() -> int:
    """Bytes per packed-broadcast chunk."""
    return envs.MX_NCCL_REFIT_MISC_CHUNK_BYTES


__all__ = [
    "DEFAULT_LAYER_GROUP",
    "NcclM2nReceiver",
    "NcclM2nSender",
    "RefitCtx",
    "misc_chunk_size",
]
