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

import ctypes
import functools
import logging
import math
import threading
import time
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, Callable

from . import envs
from .comm import CommunicatorCache, LaneCommunicator, LaneKey, NcclUnavailableError
from .spi import LocalParamSpec, RefitCtx, resolve_specs
from .types import MeshSpec, ParamPlan, ReshardPlan

logger = logging.getLogger("modelexpress_rl.collective.backend")

DEFAULT_LAYER_GROUP = 0
_M2N_CALL_LOCK = threading.RLock()
# Any transfer failure may follow CUDA work enqueued by a pre hook, reshard, or
# post hook. If its lane cannot be synchronized, retaining the associated
# contexts for process lifetime is safer than releasing storage the device may
# still consume.
_UNSETTLED_TRANSFER_RESOURCES: list[Any] = []

#: The reshard entry points were added in this NCCL release. An older library
#: fails inside the native call rather than at import, so importability alone
#: does not establish the runtime is usable.
MIN_NCCL = (2, 30, 7)


@functools.lru_cache(maxsize=1)
def loaded_nccl_version() -> tuple[int, int, int] | None:
    """Version of the libnccl this process actually resolves, or None.

    nccl4py's own ``get_version()`` reports the library it would load by path,
    which is not necessarily the one that wins: a CUDA image ships its own
    libnccl, and whichever is mapped first is the one the reshard runs
    against. Asking the loaded library directly is the only reading that
    tracks the failure.

    None means the handle did not resolve, which is a fact about this probe
    rather than about the library. It is deliberately distinct from a version
    below the floor: the probe failing is not evidence of an old runtime.
    """
    try:
        lib = ctypes.CDLL("libnccl.so.2")
        raw = ctypes.c_int()
        if lib.ncclGetVersion(ctypes.byref(raw)) != 0:
            return None
    except OSError:
        return None
    value = raw.value
    return (value // 10000, (value // 100) % 100, value % 100)


def require_nccl_m2n() -> None:
    """Fail before rendezvous when the optional M2N runtime is unavailable.

    Importability is necessary and not sufficient. The reshard entry points
    are resolved inside the native call, so a process that imports ``nccl.m2n``
    against an older libnccl fails mid-collective with peers already waiting.
    Checking the mapped library here turns that into a refusal before anyone
    joins a group.
    """
    try:
        from nccl.m2n import group, reshard  # noqa: F401
    except (ImportError, OSError) as error:  # pragma: no cover - environment dependent
        raise NcclUnavailableError(
            "the collective refit data plane needs nccl.m2n, which ships as "
            "the nccl-extensions distribution rather than as part of nccl4py; "
            "install nccl-extensions[cu12] or nccl-extensions[cu13] to match "
            "the host CUDA toolkit"
        ) from error
    if not callable(group):
        raise NcclUnavailableError(
            "the collective refit data plane needs nccl.m2n.group(), which is "
            "part of the supported nccl-extensions M2N runtime"
        )
    found = loaded_nccl_version()
    if found is None:
        logger.debug(
            "libnccl.so.2 did not resolve through ctypes, so the %s floor is "
            "unchecked; nccl.m2n imported, so this is a limit of the probe",
            ".".join(str(part) for part in MIN_NCCL),
        )
        return
    if found < MIN_NCCL:
        raise NcclUnavailableError(
            f"the libnccl this process loads is "
            f"{'.'.join(str(part) for part in found)}, and reshard needs "
            f"{'.'.join(str(part) for part in MIN_NCCL)}; preload the one "
            "nccl-extensions installed if the image ships an older library"
        )


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
    keyword, meshes nested to their shape, placements as DTensor objects wherever
    torch is installed,
    and ``stream`` passed as a raw handle only when there is one.
    """
    require_nccl_m2n()
    from nccl.m2n import reshard

    kwargs: dict[str, Any] = {
        "src_mesh": entry.src_mesh.nested(),
        "src_placements": [p.to_wire() for p in entry.src_placements],
        "dst_mesh": entry.dst_mesh.nested(),
        "dst_placements": [p.to_wire() for p in entry.dst_placements],
    }
    _set_or_validate_endpoint(
        kwargs,
        name=entry.name,
        side="src",
        buffer=src,
        expected_shape=_local_shape(
            entry.global_shape, entry.src_mesh, entry.src_placements
        ),
        expected_dtype=entry.dtype,
    )
    _set_or_validate_endpoint(
        kwargs,
        name=entry.name,
        side="dst",
        buffer=dst,
        expected_shape=_local_shape(
            entry.global_shape, entry.dst_mesh, entry.dst_placements
        ),
        expected_dtype=entry.dtype,
    )
    stream = comm.stream
    if stream is not None:
        kwargs["stream"] = _stream_handle(stream)

    # NCCL M2N's native runtime is process-global and not host-thread-safe,
    # including across different communicator handles. Serializing the Python
    # submissions still permits device-side overlap on the supplied streams.
    with _M2N_CALL_LOCK:
        reshard(src, dst, comm.handle, **kwargs)


def _local_shape(
    global_shape: tuple[int, ...],
    mesh: MeshSpec,
    placements: tuple[Any, ...],
) -> tuple[int, ...]:
    """Project a declared global shape onto one rank of a mesh."""
    shape = list(global_shape)
    for axis, placement in enumerate(placements):
        if placement.dim is not None:
            shape[placement.dim] //= mesh.shape[axis]
    return tuple(shape)


def _set_or_validate_endpoint(
    kwargs: dict[str, Any],
    *,
    name: str,
    side: str,
    buffer: Any,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
) -> None:
    """Describe an absent endpoint or validate a locally owned buffer."""
    if buffer is None:
        kwargs[f"{side}_local_shape"] = expected_shape
        kwargs[f"{side}_dtype"] = expected_dtype
        return

    data_ptr = getattr(buffer, "data_ptr", None)
    ordinary_shape = getattr(buffer, "shape", None)
    ordinary_dtype = getattr(buffer, "dtype", None)
    if callable(data_ptr) and ordinary_shape is not None and ordinary_dtype is not None:
        raw_shape = ordinary_shape
        raw_dtype = ordinary_dtype
    else:
        cuda_interface = getattr(buffer, "__cuda_array_interface__", None)
        raw_shape = cuda_interface.get("shape") if cuda_interface is not None else None
        raw_dtype = (
            cuda_interface.get("typestr") if cuda_interface is not None else None
        )
    if raw_shape is None or raw_dtype is None:
        raise TypeError(
            f"{name}: local {side} buffer must expose shape and dtype so its "
            "declared NCCL M2N layout can be validated"
        )

    actual_shape = tuple(int(dim) for dim in raw_shape)
    if actual_shape != expected_shape:
        raise ValueError(
            f"{name}: local {side} shape {actual_shape} does not match the "
            f"declared shape {expected_shape}"
        )
    actual_dtype = _normalize_dtype(raw_dtype)
    planned_dtype = _normalize_dtype(expected_dtype)
    if actual_dtype != planned_dtype:
        raise ValueError(
            f"{name}: local {side} dtype {actual_dtype!r} does not match the "
            f"declared dtype {planned_dtype!r}"
        )


def _normalize_dtype(dtype: Any) -> Any:
    """Use the binding's dtype normalization, with a dependency-light fallback."""
    try:
        from nccl.m2n.tensor import normalize_dtype
    except ImportError:
        normalized = str(dtype).removeprefix("torch.").lower()
        aliases = {
            "char": "int8",
            "byte": "uint8",
            "int": "int32",
            "long": "int64",
            "half": "float16",
            "float": "float32",
            "double": "float64",
            "float8_e4m3": "float8_e4m3fn",
        }
        normalized = aliases.get(normalized, normalized)
        try:
            import numpy as np

            normalized = str(np.dtype(normalized))
        except (ImportError, TypeError, ValueError):
            pass
        return aliases.get(normalized, normalized)
    return normalize_dtype(dtype)


def _stream_handle(stream: Any) -> int:
    """Raw CUDA stream handle for the reshard op.

    A ``torch.cuda.Stream`` carries it on ``cuda_stream``; anything already
    integral is passed through so a caller can supply a handle directly.
    """
    handle = getattr(stream, "cuda_stream", stream)
    return int(handle)


def _allocate_fence_buffer(device: Any) -> Any:
    """Allocate the default one-byte lane-fence buffer through Torch."""
    import torch

    device_context = torch.cuda.device(device) if device is not None else nullcontext()
    with device_context:
        return torch.zeros(
            1,
            dtype=torch.uint8,
            device=device if device is not None else "cuda",
        )


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
        required_bulk_names: list[str] | None = None,
        barrier_alloc: Callable[[Any], Any] | None = None,
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
        required = (
            [entry.name for entry in self._bulk]
            if required_bulk_names is None
            else list(required_bulk_names)
        )
        required.extend(entry.name for entry in plan.misc)
        resolve_specs(plan, specs, required)
        self._groups: OrderedDict[int, list[ParamPlan]] = OrderedDict()
        self.setup_layer_groups(None)
        self._pending_misc = False
        self._active_lanes: OrderedDict[int, LaneCommunicator] = OrderedDict()
        self._pending_contexts: list[tuple[int, RefitCtx]] = []
        self._previous_source_mesh_by_lane: dict[int, MeshSpec] = {}
        self._fence_buffers: dict[int, Any] = {}
        self._pending_fence_buffers: dict[int, Any] = {}
        self._barrier_alloc = (
            _allocate_fence_buffer if barrier_alloc is None else barrier_alloc
        )
        self._deadline: float | None = None
        self._timeout_s: float | None = None
        self._version: str | None = None
        self._install_mode = "drain"
        # Event install mode: group id -> one (lane, event, covered contexts)
        # entry per lane the group's transfer used. The contexts move out of
        # _pending_contexts at record time and are released when the event is
        # observed complete; stream order guarantees an event also covers every
        # earlier group's work on that lane.
        self._group_events: OrderedDict[
            int, list[tuple[LaneCommunicator, Any, list[RefitCtx]]]
        ] = OrderedDict()

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

    def _begin_transfer(self, version: str) -> None:
        """Arm this version's transfer deadline.

        READY only means the group formed. What follows it can block
        indefinitely on its own, so the transfer carries its own deadline and
        a version that overruns becomes an attributable failure rather than a
        hang with no owner.
        """
        self._pending_misc = True
        self._previous_source_mesh_by_lane.clear()
        self._group_events.clear()
        self._install_mode = envs.MX_NCCL_REFIT_INSTALL_MODE
        self._version = version
        self._timeout_s = transfer_timeout()
        self._deadline = time.monotonic() + self._timeout_s

    def _remaining(self) -> float:
        """Seconds left on this transfer, raising once it is spent."""
        if self._deadline is None:
            return math.inf
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            self._fail_deadline()
        return remaining

    def _fail_deadline(self) -> None:
        """Name what overran and let the owning operation abort safely.

        Callers may still own contexts used by enqueued CUDA work. The outer
        operation boundary must settle or quarantine those contexts before it
        aborts and clears the epoch state.
        """
        timeout_s = self._timeout_s
        version = self._version
        raise TimeoutError(
            f"collective refit of version {version!r} did not complete within "
            f"MX_NCCL_REFIT_TRANSFER_TIMEOUT_S ({timeout_s:.1f}s); group "
            f"{self._group_id} must abort and re-form at a new epoch"
        )

    def abort(self) -> None:
        """Give up on every lane of this group at once."""
        self._cache.abort_group(self._group_id)
        self._active_lanes.clear()
        self._pending_contexts.clear()
        self._group_events.clear()
        self._previous_source_mesh_by_lane.clear()
        self._fence_buffers.clear()
        self._pending_fence_buffers.clear()
        self._deadline = None

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

    def _retain_context(self, lane: LaneCommunicator, ctx: RefitCtx) -> None:
        self._pending_contexts.append((id(lane), ctx))

    def _wait_lane(self, lane: LaneCommunicator) -> None:
        remaining = self._remaining()
        try:
            lane.synchronize(timeout_s=None if remaining == math.inf else remaining)
        except TimeoutError:
            self._fail_deadline()

    def _drain_lane(self, lane: LaneCommunicator) -> None:
        self._wait_lane(lane)
        lane_id = id(lane)
        self._active_lanes.pop(lane_id, None)
        self._pending_contexts[:] = [
            (owner, ctx) for owner, ctx in self._pending_contexts if owner != lane_id
        ]
        # A drained lane's stream is provably complete, which also covers every
        # group event recorded on it; release those buckets' contexts too.
        for recorded in self._group_events.values():
            recorded[:] = [item for item in recorded if id(item[0]) != lane_id]

    def _fence_lane(self, lane: LaneCommunicator) -> None:
        lane_id = id(lane)
        barrier = self._fence_buffers.get(lane_id)
        if barrier is None:
            barrier = self._barrier_alloc(lane.device)
            self._fence_buffers[lane_id] = barrier
        # The probe itself is asynchronous work. Keep the lane visible to the
        # outer failure settlement if either broadcast or its wait fails.
        self._record_lane(lane)
        self._pending_fence_buffers[lane_id] = barrier
        _broadcast(lane, barrier, root=0)
        self._wait_lane(lane)
        self._pending_fence_buffers.pop(lane_id, None)

    def _drain_active_lanes(self) -> None:
        for lane in list(self._active_lanes.values()):
            self._drain_lane(lane)

    def _record_group_events(self, layer_group_id: int) -> None:
        """Record one completion event per lane this group's transfer used.

        Event install mode's whole premise: stream order makes such an event
        cover every reshard and hook the group enqueued on that lane, so the
        install side can prove *this group's* receive buffers complete without
        host-draining lanes that are already carrying the next group's work.
        Contexts retained for the lane move into the group's bucket and are
        released when the event is observed complete.
        """
        if layer_group_id in self._group_events:
            raise RuntimeError(
                f"layer group {layer_group_id} already has completion events "
                "recorded; issue each group at most once per round"
            )
        recorded: list[tuple[LaneCommunicator, Any, list[RefitCtx]]] = []
        partition_ids = sorted(
            {entry.partition_id for entry in self.entries(layer_group_id)}
        )
        for partition_id in partition_ids:
            lane = self._lane(partition_id)
            event = lane.record_event()
            lane_key = id(lane)
            covered = [
                ctx for owner, ctx in self._pending_contexts if owner == lane_key
            ]
            if covered:
                self._pending_contexts[:] = [
                    (owner, ctx)
                    for owner, ctx in self._pending_contexts
                    if owner != lane_key
                ]
            recorded.append((lane, event, covered))
        self._group_events[layer_group_id] = recorded

    def await_group(self, layer_group_id: int) -> None:
        """Wait for one issued group's transfers, releasing what they retained.

        Event install mode waits on the group's own lane events, bounded by
        the round's transfer deadline; afterwards install may read or release
        the group's receive buffers without a round-wide drain. Drain mode
        already host-drained inside update_weights, so there is nothing to
        wait on and this is a no-op.
        """
        recorded = self._group_events.get(layer_group_id)
        if recorded is None:
            if self._install_mode == "event":
                raise RuntimeError(
                    f"layer group {layer_group_id} has no recorded completion "
                    "events; update_weights must issue it before install"
                )
            return
        try:
            for lane, event, _covered in recorded:
                if event is None:
                    # The lane's stream cannot carry an event (test double or
                    # raw handle); the bounded lane synchronize is the fallback
                    # bound, still scoped to this group's lanes rather than
                    # every lane the round has touched.
                    self._wait_lane(lane)
                    continue
                remaining = self._remaining()
                try:
                    lane.wait_event(
                        event,
                        timeout_s=None if remaining == math.inf else remaining,
                    )
                except TimeoutError:
                    self._fail_deadline()
        except BaseException:
            self._settle_failed_transfer()
            self.abort()
            raise
        # Only now do the bucket's retained contexts get released; on failure
        # the entry stays so settlement can still see every async resource.
        del self._group_events[layer_group_id]

    def _fence_source_mesh_transition(
        self, entry: ParamPlan, lane: LaneCommunicator
    ) -> None:
        """Complete the previous ownership batch before changing source mesh.

        MILES and SGLang fence this handoff on both sides of the M2N call.
        Sparse source ranks must make the decision from the shared plan rather
        than local tensor ownership, otherwise a nonowner can run ahead into
        the next mesh while an owner still has work in flight. Each PP lane is
        independent, so its handoff must not drain another lane in the same
        concurrent wave.
        """
        lane_id = id(lane)
        source_mesh = entry.src_mesh
        previous_source_mesh = self._previous_source_mesh_by_lane.get(lane_id)
        if previous_source_mesh is not None and source_mesh != previous_source_mesh:
            self._drain_lane(lane)
            self._fence_lane(lane)
        self._previous_source_mesh_by_lane[lane_id] = source_mesh

    def _issue_reshard(
        self,
        entry: ParamPlan,
        *,
        spec: LocalParamSpec | None,
        src: Any,
        dst: Any,
        lane: LaneCommunicator | None = None,
        fence_transition: bool = True,
    ) -> tuple[LocalParamSpec, RefitCtx, LaneCommunicator] | None:
        self._remaining()
        lane = self._lane(entry.partition_id) if lane is None else lane
        if fence_transition:
            self._fence_source_mesh_transition(entry, lane)
        # A pre hook or sparse collective call can enqueue work before a
        # RefitCtx exists. Mark the lane active before any per-op work so every
        # transfer failure path settles it before aborting the epoch.
        self._record_lane(lane)
        if spec is None:
            _reshard(comm=lane, entry=entry, src=None, dst=None)
            return None
        with self._stream_context(lane, spec):
            ctx = spec.enter()
            # Retain staging buffers and hook state until the asynchronous CUDA
            # work is complete. Relying only on an allocator's stream tracking
            # is not sufficient for engine-owned or external buffers.
            self._retain_context(lane, ctx)
            # A pre hook may already have enqueued work on this stream. Mark
            # its returned context before validation or native recording can
            # fail so the outer failure path can retain it while settling.
            _reshard(comm=lane, entry=entry, src=src(ctx), dst=dst(ctx))
        return spec, ctx, lane

    def _issue_grouped_reshards(
        self,
        operations: list[
            tuple[
                ParamPlan,
                LocalParamSpec | None,
                Callable[[RefitCtx], Any],
                Callable[[RefitCtx], Any],
            ]
        ],
    ) -> None:
        """Submit deterministic same-lane/source-mesh runs as M2N groups."""
        require_nccl_m2n()
        from nccl.m2n import group

        by_lane: dict[
            int,
            list[
                tuple[
                    ParamPlan,
                    LocalParamSpec | None,
                    Callable[[RefitCtx], Any],
                    Callable[[RefitCtx], Any],
                ]
            ],
        ] = {}
        for operation in operations:
            by_lane.setdefault(operation[0].partition_id, []).append(operation)

        try:
            for lane_id in sorted(by_lane):
                lane = self._lane(lane_id)
                lane_operations = by_lane[lane_id]
                run_start = 0
                while run_start < len(lane_operations):
                    source_mesh = lane_operations[run_start][0].src_mesh
                    run_end = run_start + 1
                    while (
                        run_end < len(lane_operations)
                        and lane_operations[run_end][0].src_mesh == source_mesh
                    ):
                        run_end += 1

                    self._fence_source_mesh_transition(
                        lane_operations[run_start][0], lane
                    )
                    self._remaining()
                    pending_leaves: list[
                        tuple[LocalParamSpec, RefitCtx, LaneCommunicator]
                    ] = []
                    with _M2N_CALL_LOCK:
                        with group():
                            for entry, spec, src, dst in lane_operations[
                                run_start:run_end
                            ]:
                                pending = self._issue_reshard(
                                    entry,
                                    spec=spec,
                                    src=src,
                                    dst=dst,
                                    lane=lane,
                                    fence_transition=False,
                                )
                                if pending is not None:
                                    pending_leaves.append(pending)
                    # group_end has submitted the recorded transfers. Release
                    # the process-global native lock before post hooks enqueue
                    # framework work behind them on each lane's stream.
                    for pending_spec, ctx, pending_lane in pending_leaves:
                        with self._stream_context(pending_lane, pending_spec):
                            pending_spec.leave(ctx)
                    run_start = run_end
        except BaseException:
            # Pre hooks can enqueue before validation, recording, group_end, or
            # post fails. Prove every retained async resource safe before abort
            # clears ownership, or quarantine it for process lifetime.
            self._settle_failed_transfer()
            self.abort()
            raise

    def _settle_failed_transfer(self) -> None:
        """Keep every retained async resource alive until its stream is safe."""
        contexts_by_lane: OrderedDict[int, list[RefitCtx]] = OrderedDict()
        for lane_id, ctx in self._pending_contexts:
            contexts_by_lane.setdefault(lane_id, []).append(ctx)
        for recorded in self._group_events.values():
            for lane, _event, covered in recorded:
                if covered:
                    contexts_by_lane.setdefault(id(lane), []).extend(covered)

        lane_ids = OrderedDict.fromkeys(
            [
                *self._active_lanes,
                *contexts_by_lane,
                *self._pending_fence_buffers,
            ]
        )
        for lane_id in lane_ids:
            contexts = contexts_by_lane.get(lane_id, [])
            fence_buffer = self._pending_fence_buffers.get(lane_id)
            resources = [*contexts]
            if fence_buffer is not None:
                resources.append(fence_buffer)
            lane = self._active_lanes.get(lane_id)
            if lane is None:
                _UNSETTLED_TRANSFER_RESOURCES.extend(resources)
                logger.error(
                    "collective transfer failed with %d retained async resource(s) "
                    "whose owning lane is no longer active; retaining them for "
                    "process lifetime",
                    len(resources),
                )
                continue
            try:
                self._wait_lane(lane)
            except BaseException as error:  # noqa: BLE001 - preserve original error
                _UNSETTLED_TRANSFER_RESOURCES.extend(resources)
                logger.error(
                    "collective transfer failed and lane %s/%s could not be "
                    "synchronized; retaining %d async resource(s) for process "
                    "lifetime: %r",
                    lane.rank,
                    lane.world_size,
                    len(resources),
                    error,
                )
            else:
                self._active_lanes.pop(lane_id, None)
                self._pending_fence_buffers.pop(lane_id, None)
                self._pending_contexts[:] = [
                    (owner, ctx)
                    for owner, ctx in self._pending_contexts
                    if owner != lane_id
                ]

    def _finish_misc(self, broadcast_lane_id: int) -> None:
        if not self._pending_misc:
            return
        try:
            # Every rank drains every reshard stream it used before entering
            # the overlapping all-ranks communicator.
            self._drain_active_lanes()
            lane = self._lane(broadcast_lane_id)
            for misc in self._plan.misc:
                self._remaining()
                spec = self._specs[misc.name]
                self._record_lane(lane)
                with self._stream_context(lane, spec):
                    ctx = spec.enter()
                    self._retain_context(lane, ctx)
                    _broadcast(lane, ctx.buf, root=0)
                    spec.leave(ctx)
            # Keep temporary buffers and post hooks alive until the broadcast
            # work has completed, and do not report success for merely
            # enqueued work.
            self._drain_active_lanes()
            self._group_events.clear()
            self._pending_misc = False
        except BaseException:
            self._settle_failed_transfer()
            self.abort()
            raise


class NcclM2nSender(_CollectiveHalf):
    """Trainer half: supplies each parameter's local shard to the collective."""

    def __init__(
        self,
        *,
        source_partition: int | None = None,
        source_rank_in_lane: int | None = None,
        **kwargs: Any,
    ) -> None:
        if source_rank_in_lane is not None and (
            isinstance(source_rank_in_lane, bool)
            or not isinstance(source_rank_in_lane, int)
            or source_rank_in_lane < 0
        ):
            raise ValueError("source_rank_in_lane must be a non-negative integer")
        plan = kwargs["plan"]
        required_bulk_names = (
            None
            if source_rank_in_lane is None
            else [
                entry.name
                for entry in plan.bulk
                if (source_partition is None or entry.partition_id == source_partition)
                and source_rank_in_lane in entry.src_mesh.ranks()
            ]
        )
        super().__init__(
            active_partition=source_partition,
            required_bulk_names=required_bulk_names,
            **kwargs,
        )
        self._source_rank_in_lane = source_rank_in_lane

    def start_weight_update(self, version: str) -> None:
        self._begin_transfer(version)
        logger.debug("collective sender starting version %s", version)

    def publish_weights(self, layer_group_id: int) -> None:
        """Issue this group's reshards. Bulk only.

        The misc broadcast deliberately does not happen here. Its communicator
        spans every rank, so it overlaps every reshard lane; entering it while
        another layer group is still resharding is two overlapping
        communicators with operations in flight in different orders, which is
        the case that deadlocks.
        """
        operations = []
        for entry in self.entries(layer_group_id):
            owns_source = self._source_rank_in_lane is None or (
                self._source_rank_in_lane in entry.src_mesh.ranks()
            )
            operations.append(
                (
                    entry,
                    self._specs[entry.name] if owns_source else None,
                    lambda ctx: ctx.buf,
                    lambda ctx: None,
                )
            )
        self._issue_grouped_reshards(operations)

    def finish_weight_update(self, broadcast_lane_id: int) -> None:
        """Drain every reshard lane, then broadcast the misc parameters once."""
        self._finish_misc(broadcast_lane_id)


class NcclM2nReceiver(_CollectiveHalf):
    """Generator half: supplies each parameter's destination to the collective.

    Completion discipline is a deployment choice (MX_NCCL_REFIT_INSTALL_MODE):
    ``drain`` host-synchronizes every active lane inside update_weights, while
    ``event`` records per-lane completion events and lets await_group bound
    the wait to one group's own transfers, so issue can run ahead of install.
    Both modes end the round with the same full drain before the misc
    broadcast, and both settle retained async work identically on failure.
    """

    def start_weight_update(self, version: str) -> None:
        self._begin_transfer(version)
        logger.debug("collective receiver starting version %s", version)

    def update_weights(self, layer_group_id: int) -> None:
        self._issue_grouped_reshards(
            [
                (
                    entry,
                    self._specs[entry.name],
                    lambda ctx: None,
                    lambda ctx: ctx.buf,
                )
                for entry in self.entries(layer_group_id)
            ]
        )
        if self._install_mode == "event":
            # Loader.install still runs after this method, but it now proves
            # the group's own transfers complete (await_group) instead of
            # draining every lane the round has touched, which lets the next
            # group's reshards overlap this group's install.
            try:
                self._record_group_events(layer_group_id)
            except BaseException:
                self._settle_failed_transfer()
                self.abort()
                raise
            return
        # Loader.install runs immediately after this method. It may read or
        # release receive buffers, so the group's transfers and post hooks must
        # be complete before returning.
        try:
            self._drain_active_lanes()
        except BaseException:
            self._settle_failed_transfer()
            self.abort()
            raise

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


def transfer_timeout() -> float:
    """Deadline for one version's transfer, in seconds."""
    return envs.MX_NCCL_REFIT_TRANSFER_TIMEOUT_S


__all__ = [
    "DEFAULT_LAYER_GROUP",
    "NcclM2nReceiver",
    "NcclM2nSender",
    "RefitCtx",
]
