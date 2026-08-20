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
import threading
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any

from . import envs
from .comm import CommunicatorCache, LaneCommunicator, LaneKey, NcclUnavailableError
from .spi import LocalParamSpec, RefitCtx, resolve_specs
from .types import ParamPlan, ReshardPlan

logger = logging.getLogger("modelexpress_rl.collective.backend")

DEFAULT_LAYER_GROUP = 0
_M2N_CALL_LOCK = threading.Lock()


def require_nccl_m2n() -> None:
    """Fail before rendezvous when the optional M2N runtime is unavailable."""
    try:
        from nccl.m2n import reshard as _  # noqa: F401
    except (ImportError, OSError) as error:  # pragma: no cover - environment dependent
        raise NcclUnavailableError(
            "the collective refit data plane requires the nccl.m2n extension "
            "in addition to nccl4py"
        ) from error


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
    require_nccl_m2n()
    from nccl.m2n import reshard

    kwargs: dict[str, Any] = {
        "src_mesh": entry.src_mesh.nested(),
        "src_placements": [p.to_dtensor() for p in entry.src_placements],
        "dst_mesh": entry.dst_mesh.nested(),
        "dst_placements": [p.to_dtensor() for p in entry.dst_placements],
    }
    stream = comm.stream
    if stream is not None:
        kwargs["stream"] = _stream_handle(stream)

    # NCCL M2N's native runtime is process-global and not host-thread-safe,
    # including across different communicator handles. Serializing the Python
    # submissions still permits device-side overlap on the supplied streams.
    with _M2N_CALL_LOCK:
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
        active_partition: int | None = None,
    ) -> None:
        if plan.source_partition_count <= 0:
            raise ValueError("source_partition_count must be positive")
        invalid_partitions = sorted(
            {entry.partition_id for entry in plan.bulk}
            - set(range(plan.source_partition_count))
        )
        if invalid_partitions:
            raise ValueError(
                "bulk parameters name partition(s) outside source_partition_count: "
                f"{invalid_partitions}"
            )
        if (
            active_partition is not None
            and not 0 <= active_partition < plan.source_partition_count
        ):
            raise ValueError(
                f"active partition {active_partition} is outside "
                f"[0, {plan.source_partition_count})"
            )

        self._plan = plan
        self._specs = specs
        self._group_id = group_id
        self._epoch = epoch
        self._cache = cache
        # The admitted digest on the current stack historically treated bulk as
        # a set. Canonicalizing here keeps the per-communicator op order stable
        # even if two engines enumerate that set differently.
        self._all_bulk = sorted(plan.bulk, key=lambda entry: entry.canonical())
        self._bulk = [
            entry
            for entry in self._all_bulk
            if active_partition is None or entry.partition_id == active_partition
        ]
        required = [entry.name for entry in self._bulk] + [
            entry.name for entry in plan.misc
        ]
        resolve_specs(plan, specs, required)
        self._groups: OrderedDict[int, list[ParamPlan]] = OrderedDict()
        self.setup_layer_groups(None)
        self._pending_misc = False
        self._active_lanes: OrderedDict[int, LaneCommunicator] = OrderedDict()
        self._pending_contexts: list[RefitCtx] = []

    def setup_layer_groups(self, groupings: list[list[str]] | None) -> None:
        """Partition the bulk parameters into layer groups.

        Default is one group holding everything. A caller that wants to bound
        trainer memory splits it, at the cost of more wire operations.
        """
        self._groups = OrderedDict()
        if groupings is None:
            self._groups[DEFAULT_LAYER_GROUP] = list(self._bulk)
            return

        by_name = {entry.name: entry for entry in self._all_bulk}
        active_names = {entry.name for entry in self._bulk}
        seen: set[str] = set()
        for group_id, names in enumerate(groupings):
            name_set = set(names)
            if len(name_set) != len(names):
                raise ValueError(
                    f"layer group {group_id} names a parameter more than once"
                )
            overlap = seen & name_set
            if overlap:
                raise ValueError(f"{min(overlap)} appears in more than one layer group")
            unknown = sorted(name_set - set(by_name))
            if unknown:
                raise KeyError(f"{unknown[0]} is not a bulk parameter in this plan")
            seen.update(name_set)
            # Caller order is not trusted as collective order. Every worker
            # executes the group's active subset in the same canonical order.
            self._groups[group_id] = [
                entry
                for entry in self._all_bulk
                if entry.name in name_set and entry.name in active_names
            ]

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
        self._active_lanes.clear()
        self._pending_contexts.clear()

    def _stream_context(self, lane: LaneCommunicator, spec: LocalParamSpec):
        stream = lane.stream
        if stream is None or (spec.pre is None and spec.post is None):
            return nullcontext()
        if callable(getattr(stream, "__enter__", None)) and callable(
            getattr(stream, "__exit__", None)
        ):
            return stream
        if not hasattr(stream, "cuda_stream"):
            if spec.pre is not None or spec.post is not None:
                raise TypeError(
                    "LocalParamSpec hooks require a CUDA stream object, not only a raw handle"
                )
            return nullcontext()

        import torch

        return torch.cuda.stream(stream)

    def _record_lane(self, lane: LaneCommunicator) -> None:
        self._active_lanes[id(lane)] = lane

    def _drain_active_lanes(self) -> None:
        lanes = list(self._active_lanes.values())
        for lane in lanes:
            lane.synchronize()
        self._active_lanes.clear()
        self._pending_contexts.clear()

    def _issue_reshard(self, entry: ParamPlan, *, src: Any, dst: Any) -> None:
        lane = self._lane(entry.partition_id)
        spec = self._specs[entry.name]
        with self._stream_context(lane, spec):
            ctx = spec.enter()
            # Retain staging buffers and hook state until the asynchronous CUDA
            # work is complete. Relying only on an allocator's stream tracking
            # is not sufficient for engine-owned or external buffers.
            self._pending_contexts.append(ctx)
            _reshard(comm=lane, entry=entry, src=src(ctx), dst=dst(ctx))
            spec.leave(ctx)
        self._record_lane(lane)

    def _finish_misc(self, broadcast_lane_id: int) -> None:
        if not self._pending_misc:
            return
        # Every rank drains every reshard stream it used before entering the
        # overlapping all-ranks communicator.
        self._drain_active_lanes()
        lane = self._lane(broadcast_lane_id)
        for misc in self._plan.misc:
            spec = self._specs[misc.name]
            with self._stream_context(lane, spec):
                ctx = spec.enter()
                self._pending_contexts.append(ctx)
                _broadcast(lane, ctx.buf, root=0)
                spec.leave(ctx)
            self._record_lane(lane)
        # Keep temporary buffers and post hooks alive until the broadcast work
        # has completed, and do not report success for merely enqueued work.
        self._drain_active_lanes()
        self._pending_misc = False


class NcclM2nSender(_CollectiveHalf):
    """Trainer half: supplies each parameter's local shard to the collective."""

    def __init__(self, *, source_partition: int | None = None, **kwargs: Any) -> None:
        super().__init__(active_partition=source_partition, **kwargs)

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
            self._issue_reshard(entry, src=lambda ctx: ctx.buf, dst=lambda ctx: None)

    def finish_weight_update(self, broadcast_lane_id: int) -> None:
        """Drain every reshard lane, then broadcast the misc parameters once."""
        self._finish_misc(broadcast_lane_id)


class NcclM2nReceiver(_CollectiveHalf):
    """Generator half: supplies each parameter's destination to the collective."""

    def start_weight_update(self, version: str) -> None:
        self._pending_misc = True
        logger.debug("collective receiver starting version %s", version)

    def update_weights(self, layer_group_id: int) -> None:
        for entry in self.entries(layer_group_id):
            self._issue_reshard(entry, src=lambda ctx: None, dst=lambda ctx: ctx.buf)
        # Loader.install runs immediately after this method. It may read or
        # release receive buffers, so the group's transfers and post hooks must
        # be complete before returning.
        self._drain_active_lanes()

    def finish_weight_update(self, broadcast_lane_id: int) -> None:
        self._finish_misc(broadcast_lane_id)


def _broadcast(lane: LaneCommunicator, buf: Any, *, root: int) -> None:
    """One packed-broadcast step on the all-participants lane.

    In-place: producer and consumer pass the same buffer object, so the root's
    contents land in every other rank's. Both sides walk the misc list in the
    plan's order, which is why that order is part of the plan digest.
    """
    stream = lane.stream
    if stream is None:
        lane.handle.broadcast(sendbuf=buf, recvbuf=buf, root=root, stream=None)
    else:
        lane.handle.broadcast(
            sendbuf=buf,
            recvbuf=buf,
            root=root,
            stream=_stream_handle(stream),
        )


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
