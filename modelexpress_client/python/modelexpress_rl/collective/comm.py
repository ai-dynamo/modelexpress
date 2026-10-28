# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NCCL communicators for the collective refit path.

The only module that imports ``nccl``. The import is lazy so the plan contract
and the rendezvous stay testable without nccl4py installed, and so a deployment
that never selects this transport does not need it at all.

Two rules here exist because their absence turns a failure into a hang:

- a communicator is cached under ``(group_id, epoch)`` and dropped when the
  epoch moves, because an epoch move means the membership or the plan changed
  and the cached communicator no longer describes the group;
- an aborted communicator is never reused. Abort is what makes a deadline mean
  anything at all -- without it the call is still blocked, it merely has an
  error attached.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from . import envs


def stream_handle(stream: Any) -> int:
    """Raw CUDA stream handle for an NCCL call.

    A ``torch.cuda.Stream`` carries it on ``cuda_stream``; anything already
    integral is passed through so a caller can supply a handle directly.
    """
    return int(getattr(stream, "cuda_stream", stream))

logger = logging.getLogger("modelexpress_rl.collective.comm")

#: Event-poll backoff for bounded waits, in seconds.
_POLL_MIN_S = 20e-6
_POLL_MAX_S = 1e-3


class NcclUnavailableError(RuntimeError):
    """nccl4py is not installed.

    Raised at communicator creation rather than import, so the surrounding
    contract stays usable and testable in an environment that will never run
    the transport.
    """


class CommunicatorWedgedError(RuntimeError):
    """A lane communicator's abort did not return within its deadline.

    ``ncclCommAbort`` cannot interrupt an M2N kernel that is still running
    (fixed upstream in NCCL v2.33; the floor this path accepts is below that).
    When a transfer deadline fires while that kernel is in flight, the abort it
    triggers can therefore block forever -- which is the hang this error
    replaces. The abort is run on a daemon thread bounded by
    ``MX_NCCL_REFIT_ABORT_TIMEOUT_S``; if it overruns, the communicator is
    *wedged*: it cannot be torn down in this process, the daemon thread is left
    holding it, and this fatal error is raised.

    It is a process-level failure, not a round-level one. The integration (the
    refit runner) must catch it at its top level and exit the OS process: the
    server's epoch bump then re-forms the group from a fresh process, which is
    the only recovery for a communicator that cannot be aborted. See
    ``docs/NCCL_M2N_REFIT.md`` section 8.2. The attributes carry what overran,
    filled in as the error travels up (lane, then group, then round).
    """

    def __init__(
        self,
        message: str,
        *,
        group_id: str | None = None,
        epoch: int | None = None,
        lane_id: int | None = None,
        version: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.group_id = group_id
        self.epoch = epoch
        self.lane_id = lane_id
        self.version = version
        self.timeout_s = timeout_s


def abort_timeout() -> float:
    """How long a lane abort may run before the communicator is wedged."""
    return envs.MX_NCCL_REFIT_ABORT_TIMEOUT_S


def _run_bounded(fn: Any, timeout_s: float, *, name: str) -> bool:
    """Run ``fn`` on a daemon thread, returning whether it finished in time.

    ``fn`` must swallow its own errors: a clean abort that merely raises is a
    dirty teardown, not a wedge, and is handled by the caller's warning path.
    Only *not returning* within ``timeout_s`` is the wedge this guards. The
    thread is a daemon and is never joined past the deadline on purpose -- a
    wedged ``ncclCommAbort`` holds it forever, and the whole point is to stop
    it holding the process with it.
    """
    done = threading.Event()

    def runner() -> None:
        try:
            fn()
        finally:
            done.set()

    threading.Thread(target=runner, name=name, daemon=True).start()
    return done.wait(timeout_s)


def _nccl() -> Any:
    try:
        import nccl.bindings.nccl as bindings
        from nccl.core import communicator, utils
    except (ImportError, OSError) as error:  # pragma: no cover - environment dependent
        raise NcclUnavailableError(
            "the NCCL M2N refit path needs a compatible nccl4py installation "
            "and nccl.m2n, which ships as the nccl-extensions distribution; "
            "install nccl-extensions[cu12] or nccl-extensions[cu13] to match "
            "the host CUDA toolkit"
        ) from error
    return communicator, utils, bindings


def _reject_forced_communicator_id() -> None:
    forced_id = os.environ.get("NCCL_COMM_ID")
    if forced_id:
        raise RuntimeError(
            "NCCL_COMM_ID is incompatible with MX-brokered communicator bootstrap: "
            "ncclGetUniqueId would encode the forced endpoint instead of opening a "
            "listener on this lane's rank 0. Unset NCCL_COMM_ID on every worker "
            "before initializing NCCL."
        )


def new_unique_id() -> bytes:
    """Generate an ``ncclUniqueId`` for a lane this worker leads."""
    _reject_forced_communicator_id()
    _, utils, _ = _nccl()
    return bytes(utils.get_unique_id().as_bytes)


def _unique_id_from_bytes(raw: bytes) -> Any:
    _, utils, _ = _nccl()
    return utils.UniqueId.from_bytes(raw)


@dataclass(frozen=True)
class LaneKey:
    group_id: str
    epoch: int
    lane_id: int


class LaneCommunicator:
    """One lane's communicator, plus the stream its work is ordered on."""

    def __init__(
        self,
        comm: Any,
        *,
        rank: int,
        world_size: int,
        stream: Any,
        device: Any = None,
        unique_id: bytes | None = None,
        control: Any = None,
    ) -> None:
        self._comm = comm
        self.rank = rank
        self.world_size = world_size
        self.stream = stream
        self.device = device
        self.unique_id = unique_id
        self._control = control
        self._aborted = False
        self._wedged = False

    @property
    def aborted(self) -> bool:
        return self._aborted

    @property
    def wedged(self) -> bool:
        """The last abort of this lane overran its deadline and never returned."""
        return self._wedged

    @property
    def handle(self) -> Any:
        if self._aborted:
            raise RuntimeError(
                "this communicator was aborted; a new one must be bootstrapped "
                "against a fresh epoch before the lane can be used again"
            )
        return self._comm

    def abort(self, deadline: float | None = None) -> None:
        """Tear the communicator down after a deadline or a peer failure.

        Marked aborted even if the underlying abort raises or wedges: the point
        is that nothing may use it again, and a communicator whose peers
        disagree about what already completed is worse than one that is simply
        gone.

        ``ncclCommAbort`` cannot interrupt an M2N kernel still on the stream, so
        the underlying abort is run on a daemon thread bounded by
        ``MX_NCCL_REFIT_ABORT_TIMEOUT_S`` rather than called inline. An abort
        that returns in time keeps the existing behaviour. One that overruns
        marks the lane *wedged*, leaves the daemon thread holding the stuck
        call, and raises :class:`CommunicatorWedgedError`: the lane cannot be
        torn down in this process, and the fatal error is what lets the runner
        exit it so the group re-forms elsewhere.

        ``deadline`` is an absolute ``time.monotonic()`` instant shared by every
        lane of a group teardown, so the data and control communicators of all
        lanes together get one ``MX_NCCL_REFIT_ABORT_TIMEOUT_S`` budget rather
        than one each. Once it has passed no further abort is launched: the
        remaining handles are left untouched (never aborted concurrently with a
        stuck one) and the lane is reported wedged.
        """
        if self._aborted:
            return
        self._aborted = True
        timeout_s = abort_timeout()
        if deadline is None:
            deadline = time.monotonic() + timeout_s
        for comm in (self._comm, self._control):
            abort = getattr(comm, "abort", None)
            if abort is None:
                continue
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                self._wedged = True
                logger.error(
                    "the shared %.1fs teardown budget was spent before the NCCL "
                    "communicator for lane %s/%s could be aborted; it is left as "
                    "is and the process must exit so the group re-forms without it.",
                    timeout_s,
                    self.rank,
                    self.world_size,
                )
                break

            def _abort(abort=abort) -> None:
                try:
                    abort()
                except Exception as error:  # noqa: BLE001 - teardown must not mask the original failure
                    logger.warning(
                        "aborting the NCCL communicator did not complete cleanly: %r",
                        error,
                    )

            returned = _run_bounded(
                _abort, remaining_s, name=f"nccl-abort-{self.rank}-{self.world_size}"
            )
            if not returned:
                self._wedged = True
                logger.error(
                    "aborting the NCCL communicator for lane %s/%s did not return "
                    "within the %.1fs teardown budget; the M2N kernel is presumed "
                    "still running and the communicator is wedged. The process must "
                    "exit so the group re-forms without it.",
                    self.rank,
                    self.world_size,
                    timeout_s,
                )
                break
        if self._wedged:
            raise CommunicatorWedgedError(
                f"aborting the lane communicator for rank {self.rank} of "
                f"{self.world_size} did not return within {timeout_s:.1f}s "
                "(MX_NCCL_REFIT_ABORT_TIMEOUT_S); the communicator is wedged and "
                "this process must exit",
                timeout_s=timeout_s,
            )

    def synchronize(self, timeout_s: float | None = None) -> None:
        """Wait for work enqueued on this lane's stream.

        With ``timeout_s`` the wait is bounded: a CUDA event is recorded on the
        stream and polled, so a collective that never completes raises rather
        than blocking the process forever. A plain ``stream.synchronize()``
        cannot be bounded from Python, which is why the deadline needs an event
        rather than a check either side of the call. Without ``timeout_s`` the
        wait is the blocking synchronize, unchanged.

        Falls through to the blocking wait when the stream is not something a
        CUDA event can be recorded on - a test double, or a build without
        torch. Those cannot hang on device work, so there is nothing to bound.
        """
        if timeout_s is not None:
            event = self.record_event()
            if event is not None:
                self.wait_event(event, timeout_s)
                return
        stream = self.stream
        if stream is not None and callable(getattr(stream, "synchronize", None)):
            stream.synchronize()
            return

        import torch

        device_context = (
            torch.cuda.device(self.device) if self.device is not None else nullcontext()
        )
        with device_context:
            if stream is None:
                torch.cuda.current_stream().synchronize()
            else:
                torch.cuda.ExternalStream(stream_handle(stream)).synchronize()

    def record_event(self) -> Any | None:
        """Mark everything enqueued on this lane's stream so far.

        Returns a CUDA event that completes once that work has, or None when the
        stream cannot carry one, in which case the caller falls back to a plain
        synchronize rather than reporting a bound it did not apply. Work enqueued
        after the mark is not covered, which is what lets a caller wait for one
        layer group while the next one is already queued behind it.
        """
        try:
            import torch
        except ImportError:
            return None
        if not torch.cuda.is_available():
            return None

        stream = self.stream
        device_context = (
            torch.cuda.device(self.device) if self.device is not None else nullcontext()
        )
        # Attempt the record rather than predict the stream's shape: carrying a
        # cuda_stream attribute does not make it recordable, and a wrong guess
        # only raises on GPU hosts -- CPU-only boxes never reach the record.
        try:
            with device_context:
                if stream is None:
                    target = torch.cuda.current_stream()
                elif isinstance(stream, torch.cuda.Stream):
                    target = stream
                elif isinstance(stream, int):
                    target = torch.cuda.ExternalStream(stream)
                elif hasattr(stream, "cuda_stream"):
                    target = torch.cuda.ExternalStream(int(stream.cuda_stream))
                else:
                    return None
                event = torch.cuda.Event()
                event.record(target)
        except Exception as error:  # noqa: BLE001 - any resolve failure means fall back
            logger.debug(
                "this lane's stream cannot carry a CUDA event; the caller must "
                "fall back to a blocking wait: %r",
                error,
            )
            return None
        return event

    def wait_event(self, event: Any, timeout_s: float | None) -> None:
        """Poll ``event`` until it completes, raising once ``timeout_s`` passes.

        None applies no bound: the wait is ``event.synchronize()`` itself.

        The poll backs off from tens of microseconds to a millisecond. A refit
        waits once per layer group, so a fixed multi-millisecond sleep would be
        paid on every group and can exceed the transfer it is waiting for.
        """
        if timeout_s is None:
            event.synchronize()
            return
        deadline = time.monotonic() + timeout_s
        delay = _POLL_MIN_S
        while not event.query():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"lane {self.rank}/{self.world_size} did not finish its enqueued "
                    f"work within {timeout_s:.1f}s"
                )
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, _POLL_MAX_S)


def _resolve_timeout_s(timeout_s: float | None) -> float:
    # Validated locally: modelexpress_rl.envs imports modelexpress.envs, which
    # loads the NIXL-side modelexpress package this path must not pull in.
    value = timeout_s if timeout_s is not None else envs.MX_NCCL_REFIT_COMM_INIT_TIMEOUT_S
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timeout_s must be finite and positive")
    return value


def _wait_until_initialized(comm: Any, bindings: Any, timeout_s: float) -> None:
    """Poll a non-blocking communicator until success or a bounded failure."""
    result = getattr(bindings, "Result", None)
    success = getattr(result, "Success", None)
    in_progress = getattr(result, "InProgress", None)
    if success is None or in_progress is None or not hasattr(comm, "get_async_error"):
        raise NcclUnavailableError(
            "this nccl4py build cannot provide bounded communicator initialization; "
            "Communicator.init(config=NCCLConfig(blocking=False)) and "
            "get_async_error() are required"
        )

    deadline = time.monotonic() + timeout_s
    while True:
        status = comm.get_async_error()
        if int(status) == int(success):
            return
        if int(status) != int(in_progress):
            detail = ""
            try:
                detail = f": {comm.get_last_error()}"
            except Exception as error:  # noqa: BLE001 - diagnostic only
                logger.debug("NCCL did not provide init failure detail: %r", error)
            raise RuntimeError(
                f"NCCL communicator initialization failed with {status!r}{detail}"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"NCCL communicator initialization did not complete within {timeout_s:.1f}s"
            )
        time.sleep(min(0.01, remaining))


def _split_blocking(
    parent: Any, config_type: Any, bindings: Any, rank: int, timeout_s: float
) -> Any:
    """Split a blocking data-plane communicator off a bounded non-blocking one.

    A non-blocking communicator costs a host-side thread per NCCL group on the
    M2N NVLink path, which dominates a refit made of hundreds of reshards. The
    blocking child avoids that, and the bounded part is kept: the split is a
    collective on the non-blocking parent, so it is polled against the same
    deadline as init, and a peer that never joins leaves the parent in progress
    where aborting it returns.
    """
    child = parent.split(color=0, key=rank, config=config_type(blocking=True))
    _wait_until_initialized(parent, bindings, timeout_s)
    if not getattr(child, "is_valid", True):
        raise RuntimeError("the blocking data-plane split returned no communicator")
    return child


class CommunicatorCache:
    """Communicators for one worker, keyed by ``(group_id, epoch, lane_id)``.

    ``Communicator.init`` is the expensive part of this path -- hundreds of
    milliseconds to seconds at scale, and once per lane. Caching it across
    refits is not an optimization; at a realistic pipeline depth it is what
    makes the path viable, which is why the invalidation rule is explicit
    rather than incidental.
    """

    def __init__(self, *, blocking_data_plane: bool | None = None) -> None:
        self._lanes: dict[LaneKey, LaneCommunicator] = {}
        # Frozen here rather than read per lane: the client hashes this value
        # into its admitted digest, so every lane must run the mode it named.
        self._blocking_data_plane = (
            envs.MX_NCCL_REFIT_BLOCKING_DATA_PLANE
            if blocking_data_plane is None
            else blocking_data_plane
        )

    def get(self, key: LaneKey) -> LaneCommunicator | None:
        lane = self._lanes.get(key)
        if lane is not None and lane.aborted:
            del self._lanes[key]
            return None
        return lane

    def create(
        self,
        key: LaneKey,
        *,
        rank: int,
        world_size: int,
        unique_id: bytes,
        device: Any,
        stream: Any,
        timeout_s: float | None = None,
    ) -> LaneCommunicator:
        """Bootstrap one lane's communicator.

        ``timeout_s`` bounds the bootstrap itself. Group formation being
        bounded is not enough: READY only means the group formed, and this call
        can still block indefinitely on its own if a peer never arrives.
        """
        existing = self.get(key)
        if existing is not None:
            if (
                existing.rank != rank
                or existing.world_size != world_size
                or existing.unique_id != unique_id
                or existing.stream is not stream
            ):
                raise RuntimeError(
                    f"lane {key.lane_id} of group {key.group_id} was already initialized "
                    "with different rank, world-size, bootstrap, or stream metadata"
                )
            return existing

        _reject_forced_communicator_id()
        communicator, _, bindings = _nccl()
        timeout_s = _resolve_timeout_s(timeout_s)
        blocking_override = os.environ.get("NCCL_COMM_BLOCKING")
        if blocking_override not in (None, "", "0"):
            raise RuntimeError(
                "NCCL_COMM_BLOCKING forces blocking communicator initialization and "
                "would defeat MX_NCCL_REFIT_COMM_INIT_TIMEOUT_S; unset it or set it to 0"
            )
        config_type = getattr(communicator, "NCCLConfig", None)
        if config_type is None:
            raise NcclUnavailableError(
                "this nccl4py build lacks NCCLConfig; a build supporting "
                "non-blocking communicator initialization is required"
            )
        logger.info(
            "bootstrapping NCCL lane %s of group %s at epoch %s (rank %s of %s)",
            key.lane_id,
            key.group_id,
            key.epoch,
            rank,
            world_size,
        )
        device_context = nullcontext()
        if device is not None:
            import torch

            device_context = torch.cuda.device(device)

        comm = None
        data = None
        try:
            with device_context:
                comm = communicator.Communicator.init(
                    nranks=world_size,
                    rank=rank,
                    unique_id=_unique_id_from_bytes(unique_id),
                    config=config_type(blocking=False),
                )
                _wait_until_initialized(comm, bindings, timeout_s)
                if self._blocking_data_plane:
                    data = _split_blocking(comm, config_type, bindings, rank, timeout_s)
        except BaseException:
            if comm is not None:
                try:
                    comm.abort()
                except Exception as error:  # noqa: BLE001 - preserve bootstrap failure
                    logger.warning(
                        "aborting a failed NCCL communicator init did not complete cleanly: %r",
                        error,
                    )
            raise

        lane = LaneCommunicator(
            data if data is not None else comm,
            rank=rank,
            world_size=world_size,
            stream=stream,
            device=device,
            unique_id=bytes(unique_id),
            control=comm if data is not None else None,
        )
        self._lanes[key] = lane
        return lane

    def settle_group(
        self,
        group_id: str,
        epoch: int,
        *,
        timeout_s: float | None = None,
    ) -> int:
        """Wait until every live lane in one group epoch is usable.

        NCCL rejects a collective while the communicator still reports
        ``ncclInProgress``, so every rank polls here before the next shared
        bootstrap barrier - including ranks that created no lane in this
        step and otherwise have no readiness check.
        """
        timeout_s = _resolve_timeout_s(timeout_s)

        _, _, bindings = _nccl()
        deadline = time.monotonic() + timeout_s
        settled = 0
        for key, lane in list(self._lanes.items()):
            if key.group_id != group_id or key.epoch != epoch or lane.aborted:
                continue
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                raise TimeoutError(
                    f"lane {key.lane_id} of group {group_id} at epoch {epoch} "
                    f"did not settle: the shared {timeout_s:.3f}s budget was "
                    f"already exhausted"
                )
            try:
                _wait_until_initialized(lane.handle, bindings, remaining_s)
            except TimeoutError as error:
                raise TimeoutError(
                    f"lane {key.lane_id} of group {group_id} at epoch {epoch} "
                    f"did not settle within its remaining {remaining_s:.3f}s "
                    f"of the shared {timeout_s:.3f}s budget"
                ) from error
            except Exception as error:
                raise RuntimeError(
                    f"lane {key.lane_id} of group {group_id} at epoch {epoch} "
                    f"did not return to a usable state: {error!r}"
                ) from error
            settled += 1
        return settled

    def invalidate_epoch(self, group_id: str, epoch: int) -> int:
        """Drop every lane not at ``epoch``. Returns how many were dropped."""
        stale = [
            key
            for key in self._lanes
            if key.group_id == group_id and key.epoch != epoch
        ]
        wedged = self._abort_keys(stale)
        if wedged is not None:
            raise wedged
        return len(stale)

    def abort_group(self, group_id: str) -> int:
        """Abort every lane of a group, not just the one that failed.

        A partially aborted group leaves peers blocked in the lanes that did
        not time out, waiting on ranks that have already given up. A lane whose
        abort wedges is still removed, and every remaining lane is dropped too
        (aborted while the shared teardown budget lasts, marked wedged once it
        is spent); the :class:`CommunicatorWedgedError` is raised only after
        every lane has been dealt with, and the process exit it forces is what
        releases any lane left un-aborted.
        """
        keys = [key for key in self._lanes if key.group_id == group_id]
        wedged = self._abort_keys(keys)
        if wedged is not None:
            raise wedged
        return len(keys)

    def _abort_keys(self, keys: list[LaneKey]) -> CommunicatorWedgedError | None:
        """Abort and drop every listed lane; return the first wedge, if any.

        Every lane is aborted before any wedge propagates, because leaving a
        lane un-aborted is exactly the hang this path exists to prevent. All of
        them share one ``MX_NCCL_REFIT_ABORT_TIMEOUT_S`` deadline: once it has
        passed, the remaining lanes are dropped and marked wedged without
        launching another abort, so a wedged teardown delays the fatal report
        by at most one budget, not one per lane and handle.
        """
        wedged: CommunicatorWedgedError | None = None
        deadline = time.monotonic() + abort_timeout() if keys else None
        for key in keys:
            lane = self._lanes.pop(key, None)
            if lane is None:
                continue
            try:
                lane.abort(deadline=deadline)
            except CommunicatorWedgedError as error:
                error.group_id = key.group_id
                error.epoch = key.epoch
                if error.lane_id is None:
                    error.lane_id = key.lane_id
                if wedged is None:
                    wedged = error
        return wedged

    def __len__(self) -> int:
        return len(self._lanes)
