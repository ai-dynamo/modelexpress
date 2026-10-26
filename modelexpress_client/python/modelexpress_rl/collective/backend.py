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
from collections import OrderedDict, deque
from contextlib import nullcontext
from typing import Any

from . import envs
from .comm import (
    CommunicatorCache,
    CommunicatorWedgedError,
    LaneCommunicator,
    LaneKey,
    NcclUnavailableError,
    stream_handle,
)
from .plan import submission_order, validate_layer_groups
from .spi import LocalParamSpec, RefitCtx, resolve_specs
from .types import ParamPlan, ReshardPlan

logger = logging.getLogger("modelexpress_rl.collective.backend")

#: A single nccl.m2n reshard call that holds the host this long is logged.
_SLOW_RESHARD_WARN_S = 10.0

DEFAULT_LAYER_GROUP = 0
# Reentrant: a group holds it across recording, and each recorded reshard
# takes it again.
_M2N_CALL_LOCK = threading.RLock()

#: How long an abort waits for this half's in-flight work before deciding
#: whether its M2N handle can be destroyed.
_M2N_QUIESCE_S = 5.0

#: Handles that could not be destroyed safely. Held for the life of the process
#: so nothing ever finalizes them while GPU work may still reference their
#: resources; a process restart reclaims them.
_QUARANTINED_M2N_HANDLES: list[Any] = []

#: Buffers a wedged communicator's M2N kernel may still read or write: a
#: half's pending contexts and in-flight groups, and the engine that owns the
#: buffers. Held for the life of the process for the same reason as the
#: handles above; the process must exit on the wedge anyway.
_QUARANTINED_REFIT_BUFFERS: list[Any] = []


def _m2n_init() -> Any:
    """Create one explicit ``nccl.m2n`` handle.

    Bare ``nccl.m2n.reshard`` uses M2N's process-wide default handle, which
    nothing here could ever finalize, and which would keep communicator-keyed
    windows, device communicators and staging alive past the lanes they were
    built against. An explicit handle lets the half destroy them first.
    """
    require_nccl_m2n()
    import nccl.m2n as m2n

    with _M2N_CALL_LOCK:
        handle = m2n.init(m2n.Config())
    if not callable(getattr(handle, "destroy", None)):
        raise NcclUnavailableError(
            "this nccl-extensions build returns an M2N handle without destroy(); "
            "the collective refit needs an explicit, finalizable handle"
        )
    return handle


class _StorageRegion:
    __slots__ = ("name", "lane", "is_destination", "device", "start", "end")

    def __init__(self, name, lane, is_destination, device, start, end):
        self.name = name
        self.lane = lane
        self.is_destination = is_destination
        self.device = device
        self.start = start
        self.end = end

    def overlaps(self, other: _StorageRegion) -> bool:
        return (
            self.device == other.device
            and self.start < other.end
            and other.start < self.end
        )


def _storage_region(
    entry: ParamPlan, buf: Any, *, is_destination: bool
) -> _StorageRegion | None:
    """The byte span ``buf`` covers, or None for something that is not a tensor."""
    try:
        start = int(buf.data_ptr())
        itemsize = int(buf.element_size())
        shape = tuple(buf.shape)
        stride = tuple(buf.stride())
        device = str(buf.device)
    except (AttributeError, TypeError):
        return None
    if any(extent == 0 for extent in shape):
        return None
    span = 1 + sum((extent - 1) * step for extent, step in zip(shape, stride))
    return _StorageRegion(
        entry.name,
        entry.partition_id,
        is_destination,
        device,
        start,
        start + span * itemsize,
    )


def _check_storage(
    regions: list[_StorageRegion], entry: ParamPlan, src: Any, dst: Any
) -> None:
    """Refuse a call whose storage collides with one already in the group.

    M2N fuses the calls of a group with no ordering between them, so nothing
    recorded in one group may write storage another call reads or writes. The
    one overlap accepted is read-only source storage feeding different lanes,
    which the standalone M2N transport validated (PR #497). The check runs
    while recording, so a collision discards the group before anything is
    issued; it is a local refusal, not a fence, because a fence decided from
    local buffers would differ between ranks and break the per-communicator
    contract every rank must record identically.
    """
    for buf, is_destination in ((src, False), (dst, True)):
        if buf is None:
            continue
        region = _storage_region(entry, buf, is_destination=is_destination)
        if region is None:
            continue
        for other in regions:
            if not region.overlaps(other):
                continue
            if (
                not region.is_destination
                and not other.is_destination
                and region.lane != other.lane
            ):
                continue
            raise RuntimeError(
                f"nccl.m2n group storage overlap: {entry.name!r} "
                f"({'destination' if region.is_destination else 'source'}) overlaps "
                f"{other.name!r} "
                f"({'destination' if other.is_destination else 'source'}); every "
                "parameter of one layer group needs its own storage until the "
                "group's transfer completes"
            )
        regions.append(region)


def _m2n_destroy(handle: Any) -> None:
    with _M2N_CALL_LOCK:
        handle.destroy()


def _scratch_bytes(spec: LocalParamSpec, ctx: RefitCtx) -> int:
    """Bytes of the buffer ``pre`` allocated for this call, 0 for live storage."""
    if spec.pre is None or ctx.buf is spec.base:
        return 0
    try:
        return int(ctx.buf.nbytes)
    except (AttributeError, TypeError, ValueError):
        return 0


class _ScratchWindow:
    """Bounds the ``pre`` scratch buffers one per-call layer group keeps alive.

    Used on both halves: the generator's receive buffers and the trainer's
    send buffers (a ``pre`` that stacks grouped experts builds a fresh one per
    call). Without it a group retains every ``pre`` buffer until the whole
    group is waited on, so a group holding the full bulk keeps the full bulk's
    scratch allocated at once, and a caller that empties the allocator cache
    between rounds pays for allocating and freeing all of it every round. Calls are collected into windows of about half the
    budget. Closing a window records an event on every lane it used; once two
    windows are closed the older one is waited on and dropped, so the allocator
    can hand its buffers to later calls while the newer window is still on the
    wire. Nothing is dropped before the event covering its receive and its
    ``post`` hook has completed.
    """

    def __init__(self, budget_bytes: int):
        self.chunk_bytes = max(1, budget_bytes // 2)
        # (lane marks, contexts, scratch storage regions) per closed window.
        self.closed: deque[
            tuple[
                list[tuple[LaneCommunicator, Any]],
                list[RefitCtx],
                list[_StorageRegion],
            ]
        ] = deque()
        self._reset()

    def _reset(self) -> None:
        self._lanes: OrderedDict[int, LaneCommunicator] = OrderedDict()
        self._contexts: list[RefitCtx] = []
        self._regions: list[_StorageRegion] = []
        self._bytes = 0

    def add(
        self,
        lane: LaneCommunicator,
        ctx: RefitCtx,
        nbytes: int,
        regions: list[_StorageRegion],
    ) -> bool:
        """Account one issued call; True once the open window is full."""
        self._lanes[id(lane)] = lane
        self._contexts.append(ctx)
        self._regions.extend(regions)
        self._bytes += nbytes
        return self._bytes >= self.chunk_bytes

    def close(self) -> None:
        marks = [(lane, lane.record_event()) for lane in self._lanes.values()]
        self.closed.append((marks, self._contexts, self._regions))
        self._reset()


class _blocked_reshard_watch:
    """Log while one nccl.m2n reshard call still blocks the host.

    The transfer deadline is checked only between calls, and a call blocked
    on a peer that never enters the same collective returns never; without
    this the trainer stays silent for as long as it is stuck.
    """

    def __init__(
        self,
        entry: ParamPlan,
        lane: LaneCommunicator,
        started: float,
        *,
        calls: int = 1,
    ):
        self._entry = entry
        self._lane = lane
        self._started = started
        self._calls = calls
        self._done = threading.Event()
        self._thread = threading.Thread(
            target=self._watch, name="mx-reshard-watch", daemon=True
        )

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._done.set()
        self._thread.join()
        return False

    def _watch(self) -> None:
        while not self._done.wait(_SLOW_RESHARD_WARN_S):
            group = (
                f"submission of {self._calls} starting at " if self._calls > 1 else ""
            )
            logger.warning(
                "nccl.m2n reshard %sof %r on lane %s (rank %s of %s) still blocks "
                "the host after %.0fs",
                group,
                self._entry.name,
                self._entry.partition_id,
                self._lane.rank,
                self._lane.world_size,
                time.monotonic() - self._started,
            )


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


def installed_m2n_build() -> str:
    """Name and version of the installed distribution that ships ``nccl.m2n``.

    Read from package metadata only, so nothing native is loaded. ``nccl`` is
    a namespace shared with nccl4py, so the distribution is the one whose
    files include the ``nccl/m2n`` package rather than any ``nccl`` provider.
    Returns "" when no installed distribution ships it.
    """
    from importlib import metadata

    for name in metadata.packages_distributions().get("nccl", []):
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        files = dist.files or []
        if any(tuple(path.parts[:2]) == ("nccl", "m2n") for path in files):
            return f"{dist.metadata['Name']}=={dist.version}"
    return ""


def require_nccl_m2n() -> None:
    """Fail before rendezvous when the optional M2N runtime is unavailable.

    Importability is necessary and not sufficient. The reshard entry points
    are resolved inside the native call, so a process that imports ``nccl.m2n``
    against an older libnccl fails mid-collective with peers already waiting.
    Checking the mapped library here turns that into a refusal before anyone
    joins a group.
    """
    try:
        from nccl.m2n import reshard as _  # noqa: F401
    except (ImportError, OSError) as error:  # pragma: no cover - environment dependent
        raise NcclUnavailableError(
            "the collective refit data plane needs nccl.m2n, which ships as "
            "the nccl-extensions distribution rather than as part of nccl4py; "
            "install nccl-extensions[cu12] or nccl-extensions[cu13] to match "
            "the host CUDA toolkit"
        ) from error
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
    handle: Any,
) -> None:
    """Issue one ``nccl.m2n.reshard`` on the half's explicit M2N handle.

    Co-called: the trainer passes ``dst=None`` and the generator passes
    ``src=None``, and NCCL routes the many-to-many redistribution internally
    from the two meshes. Both sides still pass *both* meshes, because each has
    to know the shape of the other end to route into it.

    The argument shape is pinned to the one NeMo RL's ``xferdtensor`` uses, so
    an MX-brokered deployment and a NeMo-RL-native one issue the identical
    call: tensors and communicator positional, meshes and placements by
    keyword, meshes nested to their shape, placements as real DTensor objects,
    and ``stream`` passed as a raw handle only when there is one. The M2N
    ``handle`` is passed by keyword and always, so no call ever lands on the
    process-wide default handle that nothing finalizes.
    """
    require_nccl_m2n()
    from nccl.m2n import reshard

    kwargs: dict[str, Any] = {
        "src_mesh": entry.src_mesh.nested(),
        "src_placements": [p.to_dtensor() for p in entry.src_placements],
        "dst_mesh": entry.dst_mesh.nested(),
        "dst_placements": [p.to_dtensor() for p in entry.dst_placements],
        "handle": handle,
    }
    stream = comm.stream
    if stream is not None:
        kwargs["stream"] = stream_handle(stream)

    # NCCL M2N's native runtime is process-global and not host-thread-safe,
    # including across different communicator handles. Serializing the Python
    # submissions still permits device-side overlap on the supplied streams.
    with _M2N_CALL_LOCK:
        reshard(src, dst, comm.handle, **kwargs)


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
        grouped_submission: bool | None = None,
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
        # Frozen at construction: the client hashes this value into its
        # admitted digest, so the wire path must not follow a later env change.
        self._grouped_submission = (
            envs.MX_NCCL_REFIT_GROUPED_SUBMISSION
            if grouped_submission is None
            else grouped_submission
        )
        # Wire order is canonical, independent of how the engine enumerated
        # bulk. The digest is stricter and also pins the declared order.
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
        self._groups_run: set[int] = set()
        self._active_lanes: OrderedDict[int, LaneCommunicator] = OrderedDict()
        self._pending_contexts: list[RefitCtx] = []
        # Layer groups whose receives are enqueued but not yet waited on, each
        # with the per-lane events marking its end and the contexts it holds.
        self._in_flight: OrderedDict[
            int, tuple[list[tuple[LaneCommunicator, Any]], list[RefitCtx]]
        ] = OrderedDict()
        self._deadline: float | None = None
        self._timeout_s: float | None = None
        self._version: str | None = None
        # This half's explicit M2N handle. One per half, and a half is one group
        # at one epoch, so the handle never outlives the lanes it was used on.
        self._m2n_handle: Any | None = None
        self._closed = False

    @property
    def m2n_handle(self) -> Any | None:
        return self._m2n_handle

    def _ensure_m2n_handle(self) -> Any:
        if self._closed:
            raise RuntimeError(
                f"the collective half for group {self._group_id} at epoch "
                f"{self._epoch} was aborted or closed; compute_plan must run again"
            )
        if self._m2n_handle is None:
            self._m2n_handle = _m2n_init()
        return self._m2n_handle

    def _lanes_quiesced(self, quiesce_s: float) -> bool:
        """Whether every lane with work in flight finished it within ``quiesce_s``.

        ``quiesce_s`` bounds the whole wait, not each lane: every lane draws on
        one shared deadline, so teardown never takes lane_count x quiesce_s.
        """
        deadline = time.monotonic() + quiesce_s
        lanes = list(self._active_lanes.values())
        for marks, _ in self._in_flight.values():
            lanes.extend(lane for lane, _ in marks)
        seen: set[int] = set()
        for lane in lanes:
            if id(lane) in seen:
                continue
            seen.add(id(lane))
            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0 or lane.aborted:
                return False
            try:
                lane.synchronize(timeout_s=remaining)
            except Exception as error:  # noqa: BLE001 - any failure means "not idle"
                logger.warning(
                    "lane %s/%s did not quiesce before M2N handle release: %r",
                    lane.rank,
                    lane.world_size,
                    error,
                )
                return False
        return True

    def _release_m2n_handle(self, *, quiesce_s: float) -> None:
        """Destroy the M2N handle while the communicators it used are still live.

        M2N caches windows, device communicators, streams and staging against
        each communicator it has seen, so the handle goes first and the lanes
        after. Destroying it while GPU work may still reference those resources
        is not safe either, so the half's in-flight work is given ``quiesce_s``
        to land; when it does not, the handle is quarantined - kept referenced
        and never destroyed - and the caller tears the lanes down regardless,
        because releasing peers matters more than reclaiming the handle. A
        process restart reclaims a quarantined handle.
        """
        handle = self._m2n_handle
        self._m2n_handle = None
        if handle is None:
            return
        if not self._lanes_quiesced(quiesce_s):
            _QUARANTINED_M2N_HANDLES.append(handle)
            logger.error(
                "M2N handle of group %s at epoch %s quarantined: work on its lanes "
                "may still be in flight, so it is not destroyed (%d quarantined in "
                "this process; a restart reclaims them)",
                self._group_id,
                self._epoch,
                len(_QUARANTINED_M2N_HANDLES),
            )
            return
        try:
            _m2n_destroy(handle)
        except Exception:  # noqa: BLE001 - the lanes must still be torn down
            _QUARANTINED_M2N_HANDLES.append(handle)
            logger.exception(
                "destroying the M2N handle of group %s at epoch %s failed; "
                "quarantined",
                self._group_id,
                self._epoch,
            )

    def close(self) -> None:
        """Healthy teardown of this half, before its lanes are torn down.

        Destroys the M2N handle while the lanes are still live. Does not touch
        the communicators: the caller owns those and tears them down next.
        Idempotent.
        """
        self._release_m2n_handle(quiesce_s=self._quiesce_bound())
        self._closed = True
        self._active_lanes.clear()
        self._pending_contexts.clear()
        self._in_flight.clear()
        self._deadline = None

    def _quiesce_bound(self) -> float:
        if self._deadline is None:
            return _M2N_QUIESCE_S
        return max(0.0, min(_M2N_QUIESCE_S, self._deadline - time.monotonic()))

    def setup_layer_groups(self, groupings: list[list[str]] | None) -> None:
        """Partition the bulk parameters into layer groups.

        Default is one group holding everything. Each group is issued as its
        own submission, so splitting bounds how much a single submission
        carries, and on the generator lets ``Loader.install`` of one group
        overlap the transfer of the next (``MX_NCCL_REFIT_PIPELINE_DEPTH``).
        It costs more wire operations. It does not bound trainer staging
        memory: the sender keeps every group's staging contexts until
        ``finish_weight_update``, so peak trainer staging is the whole refit
        whether or not the parameters are split.
        """
        self._groups = OrderedDict()
        if groupings is None:
            self._groups[DEFAULT_LAYER_GROUP] = list(self._bulk)
            return

        validate_layer_groups(self._plan, groupings)
        active_names = {entry.name for entry in self._bulk}
        for group_id, names in enumerate(groupings):
            name_set = set(names)
            # Caller order is not trusted as collective order. Every worker
            # executes the group's active subset in the same canonical order.
            self._groups[group_id] = [
                entry
                for entry in self._all_bulk
                if entry.name in name_set and entry.name in active_names
            ]

    @property
    def layer_group_ids(self) -> list[int]:
        return list(self._groups)

    def entries(self, layer_group_id: int) -> list[ParamPlan]:
        if layer_group_id not in self._groups:
            raise KeyError(f"no layer group {layer_group_id}")
        return self._groups[layer_group_id]

    def _claim_group(self, layer_group_id: int) -> list[ParamPlan]:
        """Each layer group runs once per round; a repeat would desync peers."""
        entries = self.entries(layer_group_id)
        if layer_group_id in self._groups_run:
            raise RuntimeError(
                f"layer group {layer_group_id} already ran in this round"
            )
        self._groups_run.add(layer_group_id)
        return entries

    def _require_every_group(self) -> None:
        """A skipped group leaves peers waiting on operations never issued."""
        skipped = [g for g in self._groups if g not in self._groups_run]
        if skipped:
            raise RuntimeError(
                f"finish_weight_update called before layer group(s) {skipped} ran"
            )

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
        self._groups_run = set()
        self._in_flight.clear()
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
        """Abort the group and name what overran.

        The abort is not optional. Peers that disagree about which collectives
        completed cannot be recovered by retrying on the same communicator, so
        the group is torn down and has to re-form at a fresh epoch.
        """
        timeout_s = self._timeout_s
        version = self._version
        self.abort()
        raise TimeoutError(
            f"collective refit of version {version!r} did not complete within "
            f"MX_NCCL_REFIT_TRANSFER_TIMEOUT_S ({timeout_s:.1f}s); group "
            f"{self._group_id} was aborted and must re-form at a new epoch"
        )

    def abort(self) -> None:
        """Give up on every lane of this group at once.

        The M2N handle is released first, while the lanes are still live, and
        the communicators are aborted after it. The half cannot run another
        round; the group has to re-form at a fresh epoch.

        If a lane's abort wedges (``ncclCommAbort`` cannot interrupt a running
        M2N kernel), every lane is still aborted and this half is still closed,
        then :class:`CommunicatorWedgedError` is re-raised with this group's
        epoch and version attached. That error is fatal: the runner exits the
        process so the group re-forms without this worker. The half's pending
        buffers are then quarantined for the life of the process rather than
        dropped, since the kernel the abort could not stop may still use them.
        """
        wedged: CommunicatorWedgedError | None = None
        try:
            self._release_m2n_handle(quiesce_s=self._quiesce_bound())
        finally:
            self._closed = True
            try:
                self._cache.abort_group(self._group_id)
            except CommunicatorWedgedError as error:
                error.version = self._version
                if error.epoch is None:
                    error.epoch = self._epoch
                if error.group_id is None:
                    error.group_id = self._group_id
                wedged = error
        if wedged is not None:
            # The kernel the abort could not stop may still use these buffers,
            # so they are quarantined, never freed; new containers replace them.
            _QUARANTINED_REFIT_BUFFERS.append(
                (self._pending_contexts, self._in_flight)
            )
            self._pending_contexts = []
            self._in_flight = OrderedDict()
        self._active_lanes.clear()
        self._pending_contexts.clear()
        self._in_flight.clear()
        self._deadline = None
        if wedged is not None:
            raise wedged

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
            remaining = self._remaining()
            try:
                lane.synchronize(
                    timeout_s=None if remaining == math.inf else remaining
                )
            except TimeoutError:
                self._fail_deadline()
        self._active_lanes.clear()
        self._pending_contexts.clear()

    def _submit_layer_group(
        self,
        entries: list[ParamPlan],
        *,
        src: Any,
        dst: Any,
        window_bytes: int = 0,
    ) -> list[LaneCommunicator]:
        """Issue one layer group's reshards, fused in one ``m2n.group()`` if opted in.

        By default each entry is issued as its own ``nccl.m2n.reshard`` call
        (same handle, lane and stream as the grouped path), because
        nccl-extensions 0.1.0 needs GIN for a group of two or more reshards
        and fails the group end on every rank without it.
        ``MX_NCCL_REFIT_GROUPED_SUBMISSION=1`` records them in one
        ``m2n.group()`` instead; what follows describes that path, and the
        ordering, storage check, hook placement and deadline check apply to
        both.

        Entries are recorded in :func:`submission_order`: grouped into runs by
        lane and source/destination mesh, runs in ascending lane order. M2N
        buckets the group by handle, communicator and stream and PACK-fuses the
        calls of a bucket that share a topology, so a run is what it can fuse;
        every rank records the same per-communicator sequence, and a rank on
        one lane records a subsequence of the same total order.

        Inside the group ``reshard`` only records, so ``pre`` hooks run while
        recording (stream-ordered before the issued transfer) and ``post``
        hooks run only after the group has been issued. A failure while
        recording discards the whole group: nothing is issued. The transfer
        deadline is checked before the group opens, never inside it, because
        the abort it triggers destroys the M2N handle and that is refused while
        a group is open.

        ``window_bytes`` bounds the ``pre`` scratch the per-call path keeps
        alive (see :class:`_ScratchWindow`); each ``post`` hook then runs right
        after its own call instead of after the last one, which is the same
        stream order. 0 keeps every context until the group is waited on. A
        grouped submission ignores it: nothing in a group is issued before the
        group closes, so nothing in it can be released early. The per-call
        path then takes the call lock per call rather than across the group,
        so a window wait never holds it.

        Returns the lanes the group used, in submission order.
        """
        self._remaining()
        ordered = submission_order(entries)
        if not ordered:
            return []
        handle = self._m2n_handle
        if handle is None:
            raise RuntimeError("start_weight_update must run before a transfer")
        lanes: OrderedDict[int, LaneCommunicator] = OrderedDict()
        for entry in ordered:
            lane = self._lane(entry.partition_id)
            lanes[id(lane)] = lane
            # Recorded before anything is enqueued, so a hook or reshard that
            # raises partway still leaves the lane in the drain set.
            self._record_lane(lane)

        import nccl.m2n as m2n

        staged: list[tuple[ParamPlan, LaneCommunicator, LocalParamSpec, RefitCtx]] = []
        regions: list[_StorageRegion] = []
        grouped = self._grouped_submission
        window = _ScratchWindow(window_bytes) if window_bytes and not grouped else None
        started = time.monotonic()
        # The native runtime is process-global and not host-thread-safe, and a
        # group records on the calling thread, so one lock spans the group.
        # Each per-call reshard takes the lock itself, so the windowed path
        # does not hold it across the window waits.
        group_lock = _M2N_CALL_LOCK if window is None else nullcontext()
        with group_lock, _blocked_reshard_watch(
            ordered[0], next(iter(lanes.values())), started, calls=len(ordered)
        ), (m2n.group() if grouped else nullcontext()):
            for entry in ordered:
                if not grouped:
                    # Each per-call reshard is issued on its own, so the
                    # budget is re-checked before every one; a group only
                    # records here and was checked before it opened.
                    self._remaining()
                lane = self._lane(entry.partition_id)
                spec = self._specs[entry.name]
                with self._stream_context(lane, spec):
                    ctx = spec.enter()
                # Retain staging buffers and hook state until the asynchronous
                # CUDA work is complete. Relying only on an allocator's stream
                # tracking is not sufficient for engine-owned or external buffers.
                self._pending_contexts.append(ctx)
                checked = len(regions)
                _check_storage(regions, entry, src(ctx), dst(ctx))
                _reshard(
                    comm=lane,
                    entry=entry,
                    src=src(ctx),
                    dst=dst(ctx),
                    handle=handle,
                )
                if window is None:
                    staged.append((entry, lane, spec, ctx))
                    continue
                with self._stream_context(lane, spec):
                    spec.leave(ctx)
                nbytes = _scratch_bytes(spec, ctx)
                # Only scratch regions are forgotten on release: the allocator
                # may hand that memory to a later call of this group, which is
                # then not a collision. Live storage stays checked.
                full = window.add(
                    lane, ctx, nbytes, regions[checked:] if nbytes else []
                )
                del ctx
                if full:
                    window.close()
                    self._release_windows(window, regions)
        elapsed = time.monotonic() - started
        if elapsed >= _SLOW_RESHARD_WARN_S:
            # nccl.m2n issues host-blocking collectives while issuing a group or
            # a call (DevComm creation, window registration) that wait for every
            # lane rank to enter them; a submission that blocks this long is a
            # peer that has not reached the same submission.
            logger.warning(
                "nccl.m2n %s submission of %d reshard(s) starting at %r on lane %s "
                "blocked the host for %.1fs",
                "grouped" if grouped else "per-call",
                len(ordered),
                ordered[0].name,
                ordered[0].partition_id,
                elapsed,
            )
        for _, lane, spec, ctx in staged:
            with self._stream_context(lane, spec):
                spec.leave(ctx)
        return list(lanes.values())

    def _wait_marks(self, marks: list[tuple[LaneCommunicator, Any]]) -> None:
        """Wait for each lane's mark; a lane without an event is drained whole."""
        for lane, event in marks:
            remaining = self._remaining()
            try:
                if event is None:
                    lane.synchronize(
                        timeout_s=None if remaining == math.inf else remaining
                    )
                else:
                    lane.wait_event(event, remaining)
            except TimeoutError:
                self._fail_deadline()

    def _release_windows(
        self, window: _ScratchWindow, regions: list[_StorageRegion]
    ) -> None:
        """Wait for and drop every closed window but the newest."""
        while len(window.closed) > 1:
            marks, contexts, released = window.closed.popleft()
            self._wait_marks(marks)
            done = {id(ctx) for ctx in contexts}
            self._pending_contexts = [
                ctx for ctx in self._pending_contexts if id(ctx) not in done
            ]
            gone = {id(region) for region in released}
            regions[:] = [region for region in regions if id(region) not in gone]

    def _finish_misc(self, broadcast_lane_id: int) -> None:
        if not self._pending_misc:
            return
        # Every rank drains every reshard stream it used before entering the
        # overlapping all-ranks communicator.
        self._drain_active_lanes()
        lane = self._lane(broadcast_lane_id)
        for misc in self._plan.misc:
            self._remaining()
            spec = self._specs[misc.name]
            # Recorded before anything is enqueued, as for a reshard, so a
            # broadcast or hook that raises still leaves the lane drained.
            self._record_lane(lane)
            with self._stream_context(lane, spec):
                ctx = spec.enter()
                self._pending_contexts.append(ctx)
                _broadcast(lane, ctx.buf, root=0)
                spec.leave(ctx)
        # Keep temporary buffers and post hooks alive until the broadcast work
        # has completed, and do not report success for merely enqueued work.
        self._drain_active_lanes()
        self._pending_misc = False


class NcclM2nSender(_CollectiveHalf):
    """Trainer half: supplies each parameter's local shard to the collective."""

    def __init__(self, *, source_partition: int | None = None, **kwargs: Any) -> None:
        super().__init__(active_partition=source_partition, **kwargs)

    def start_weight_update(self, version: str) -> None:
        self._ensure_m2n_handle()
        self._begin_transfer(version)
        logger.debug("collective sender starting version %s", version)

    def publish_weights(self, layer_group_id: int) -> None:
        """Issue this group's reshards. Bulk only.

        The misc broadcast deliberately does not happen here. Its communicator
        spans every rank, so it overlaps every reshard lane; entering it while
        another layer group is still resharding is two overlapping
        communicators with operations in flight in different orders, which is
        the case that deadlocks.

        On the per-call path the ``pre`` scratch this group keeps alive (for
        example grouped experts stacked into a fresh buffer per call) is
        bounded by ``MX_NCCL_REFIT_SEND_WINDOW_BYTES``: earlier calls are waited
        on and dropped while the group is still being issued, as on the
        receive side.
        """
        started = time.monotonic()
        entries = self._claim_group(layer_group_id)
        self._submit_layer_group(
            entries,
            src=lambda ctx: ctx.buf,
            dst=lambda ctx: None,
            window_bytes=envs.MX_NCCL_REFIT_SEND_WINDOW_BYTES,
        )
        logger.info(
            "collective sender issued layer group %s (%d entries) in %.2fs",
            layer_group_id,
            len(entries),
            time.monotonic() - started,
        )

    def finish_weight_update(self, broadcast_lane_id: int) -> None:
        """Drain every reshard lane, then broadcast the misc parameters once."""
        self._require_every_group()
        self._finish_misc(broadcast_lane_id)


class NcclM2nReceiver(_CollectiveHalf):
    """Generator half: supplies each parameter's destination to the collective."""

    def start_weight_update(self, version: str) -> None:
        self._ensure_m2n_handle()
        self._begin_transfer(version)
        logger.debug("collective receiver starting version %s", version)

    def update_weights(self, layer_group_id: int) -> None:
        """Receive one layer group and return once it has fully landed."""
        self.issue_weights(layer_group_id)
        self.wait_weights(layer_group_id)

    def issue_weights(self, layer_group_id: int) -> None:
        """Enqueue one layer group's receives without waiting for them.

        Each lane the group used gets an event recorded right after its last
        receive, so wait_weights can wait for this group alone while later
        groups are already queued on the same streams.

        On the per-call path the group's ``pre`` scratch is bounded by
        ``MX_NCCL_REFIT_RECEIVE_WINDOW_BYTES``: earlier calls of the group are
        waited on and dropped while it is still being issued, so a large group
        does not keep all of its receive buffers at once.
        """
        started = time.monotonic()
        first = len(self._pending_contexts)
        entries = self._claim_group(layer_group_id)
        lanes = self._submit_layer_group(
            entries,
            src=lambda ctx: None,
            dst=lambda ctx: ctx.buf,
            window_bytes=envs.MX_NCCL_REFIT_RECEIVE_WINDOW_BYTES,
        )
        # Recorded after the group's post hooks, so a mark covers them too.
        marks = [(lane, lane.record_event()) for lane in lanes]
        self._in_flight[layer_group_id] = (marks, self._pending_contexts[first:])
        logger.info(
            "collective receiver issued layer group %s (%d entries) in %.2fs",
            layer_group_id,
            len(entries),
            time.monotonic() - started,
        )

    def wait_weights(self, layer_group_id: int) -> None:
        """Block until one issued layer group's receives and post hooks are done.

        Loader.install runs after this. It may read or release the receive
        buffers, so it must never see a group whose transfers are in flight.
        A lane whose stream cannot carry an event falls back to draining the
        whole stream, which is correct and only loses the overlap.
        """
        if layer_group_id not in self._in_flight:
            raise RuntimeError(
                f"layer group {layer_group_id} was not issued, or was already waited on"
            )
        started = time.monotonic()
        marks, contexts = self._in_flight.pop(layer_group_id)
        self._wait_marks(marks)
        # A lane no later in-flight group uses is now fully drained, so the
        # finish-time drain has nothing left to wait for on it.
        still_used = {
            id(lane) for pending, _ in self._in_flight.values() for lane, _ in pending
        }
        for lane, _ in marks:
            if id(lane) not in still_used:
                self._active_lanes.pop(id(lane), None)
        done = {id(ctx) for ctx in contexts}
        self._pending_contexts = [
            ctx for ctx in self._pending_contexts if id(ctx) not in done
        ]
        logger.info(
            "collective receiver landed layer group %s: drained in %.2fs",
            layer_group_id,
            time.monotonic() - started,
        )

    def finish_weight_update(self, broadcast_lane_id: int) -> None:
        self._require_every_group()
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
            stream=stream_handle(stream),
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
