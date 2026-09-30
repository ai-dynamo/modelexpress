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
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

from . import envs

logger = logging.getLogger("modelexpress_rl.collective.comm")


class NcclUnavailableError(RuntimeError):
    """nccl4py is not installed.

    Raised at communicator creation rather than import, so the surrounding
    contract stays usable and testable in an environment that will never run
    the transport.
    """


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
    ) -> None:
        self._comm = comm
        self.rank = rank
        self.world_size = world_size
        self.stream = stream
        self.device = device
        self.unique_id = unique_id
        self._aborted = False

    @property
    def aborted(self) -> bool:
        return self._aborted

    @property
    def handle(self) -> Any:
        if self._aborted:
            raise RuntimeError(
                "this communicator was aborted; a new one must be bootstrapped "
                "against a fresh epoch before the lane can be used again"
            )
        return self._comm

    def abort(self) -> None:
        """Tear the communicator down after a deadline or a peer failure.

        Marked aborted even if the underlying abort raises: the point is that
        nothing may use it again, and a communicator whose peers disagree about
        what already completed is worse than one that is simply gone.
        """
        if self._aborted:
            return
        self._aborted = True
        abort = getattr(self._comm, "abort", None)
        if abort is None:
            return
        try:
            abort()
        except Exception as error:  # noqa: BLE001 - teardown must not mask the original failure
            logger.warning(
                "aborting the NCCL communicator did not complete cleanly: %r", error
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
        CUDA event can be recorded on, such as a test double. Without torch
        (a JAX worker, for one) the event and the blocking wait both go
        through the CUDA runtime bindings that nccl4py already depends on.
        """
        if timeout_s is not None and self._synchronize_bounded(timeout_s):
            return
        stream = self.stream
        if stream is not None and callable(getattr(stream, "synchronize", None)):
            stream.synchronize()
            return

        try:
            import torch
        except ImportError:
            with cuda_device(self.device):
                _cudart_check(_cudart().cudaStreamSynchronize(_stream_handle(stream)))
            return

        device_context = (
            torch.cuda.device(self.device) if self.device is not None else nullcontext()
        )
        with device_context:
            if stream is None:
                torch.cuda.current_stream().synchronize()
            else:
                handle = getattr(stream, "cuda_stream", stream)
                torch.cuda.ExternalStream(int(handle)).synchronize()

    def _synchronize_bounded(self, timeout_s: float) -> bool:
        """Poll a CUDA event until this lane's work lands or the deadline passes.

        Returns False when the stream cannot carry an event, so the caller can
        fall back rather than silently reporting a bound it did not apply.
        """
        try:
            import torch
        except ImportError:
            return self._synchronize_bounded_cudart(timeout_s)
        if not torch.cuda.is_available():
            return False

        stream = self.stream
        device_context = (
            torch.cuda.device(self.device) if self.device is not None else nullcontext()
        )
        # Resolving and recording is attempted rather than predicted. An object
        # merely CARRYING a cuda_stream attribute is not necessarily one torch
        # can record against - the test doubles in this repo have exactly that
        # shape - and a wrong guess here would raise on a GPU box while every
        # CPU box stayed green.
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
                    return False
                event = torch.cuda.Event()
                event.record(target)
        except Exception as error:  # noqa: BLE001 - any resolve failure means fall back
            logger.debug(
                "this lane's stream cannot carry a CUDA event, falling back to a "
                "blocking wait: %r",
                error,
            )
            return False

        deadline = time.monotonic() + timeout_s
        while not event.query():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"lane {self.rank}/{self.world_size} did not finish its enqueued "
                    f"work within {timeout_s:.1f}s"
                )
            time.sleep(min(0.005, remaining))
        return True


    def _synchronize_bounded_cudart(self, timeout_s: float) -> bool:
        """The event-polling wait, through the CUDA runtime instead of torch."""
        try:
            runtime = _cudart()
        except ImportError:
            return False
        with cuda_device(self.device):
            error, event = runtime.cudaEventCreateWithFlags(
                runtime.cudaEventDisableTiming
            )
            _cudart_check((error,))
            try:
                _cudart_check(
                    runtime.cudaEventRecord(event, _stream_handle(self.stream))
                )
                deadline = time.monotonic() + timeout_s
                while True:
                    (status,) = runtime.cudaEventQuery(event)
                    if status == runtime.cudaError_t.cudaSuccess:
                        return True
                    if status != runtime.cudaError_t.cudaErrorNotReady:
                        _cudart_check((status,))
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"lane {self.rank}/{self.world_size} did not finish its "
                            f"enqueued work within {timeout_s:.1f}s"
                        )
                    time.sleep(min(0.005, remaining))
            finally:
                runtime.cudaEventDestroy(event)


def _cudart() -> Any:
    """The CUDA runtime bindings, which nccl4py pulls in through cuda.core."""
    from cuda.bindings import runtime  # noqa: PLC0415

    return runtime


def _cudart_check(result: Any) -> Any:
    """Raise on a failed cuda-python call, which reports status in slot 0."""
    runtime = _cudart()
    error = result[0]
    if error != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA runtime call failed: {error!r}")
    return result[1:] if len(result) > 1 else None


def _stream_handle(stream: Any) -> int:
    """A raw stream handle; None is the legacy default stream."""
    if stream is None:
        return 0
    return int(getattr(stream, "cuda_stream", stream))


def _device_ordinal(device: Any) -> int:
    if isinstance(device, int):
        return device
    index = getattr(device, "index", None)
    if isinstance(index, int):
        return index
    text = str(device)
    if text.startswith("cuda:"):
        return int(text.split(":", 1)[1])
    return int(text)


@contextmanager
def cuda_device(device: Any):
    """Make ``device`` current for the block, torch or not.

    ``torch.cuda.device`` where torch is installed, so a torch worker behaves
    exactly as before; otherwise the CUDA runtime's own set/restore, which is
    what lets a worker whose framework is not torch run the refit path.
    """
    if device is None:
        yield
        return
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        with torch.cuda.device(device):
            yield
        return
    runtime = _cudart()
    (previous,) = _cudart_check(runtime.cudaGetDevice())
    _cudart_check(runtime.cudaSetDevice(_device_ordinal(device)))
    try:
        yield
    finally:
        _cudart_check(runtime.cudaSetDevice(previous))


def _resolve_timeout_s(timeout_s: float | None) -> float:
    resolved = (
        timeout_s if timeout_s is not None else envs.MX_NCCL_REFIT_COMM_INIT_TIMEOUT_S
    )
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"timeout_s must be finite and positive, got {resolved!r}")
    return resolved


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


class CommunicatorCache:
    """Communicators for one worker, keyed by ``(group_id, epoch, lane_id)``.

    ``Communicator.init`` is the expensive part of this path -- hundreds of
    milliseconds to seconds at scale, and once per lane. Caching it across
    refits is not an optimization; at a realistic pipeline depth it is what
    makes the path viable, which is why the invalidation rule is explicit
    rather than incidental.
    """

    def __init__(self) -> None:
        self._lanes: dict[LaneKey, LaneCommunicator] = {}

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
        device_context = cuda_device(device)

        comm = None
        try:
            with device_context:
                comm = communicator.Communicator.init(
                    nranks=world_size,
                    rank=rank,
                    unique_id=_unique_id_from_bytes(unique_id),
                    config=config_type(blocking=False),
                )
                _wait_until_initialized(comm, bindings, timeout_s)
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
            comm,
            rank=rank,
            world_size=world_size,
            stream=stream,
            device=device,
            unique_id=bytes(unique_id),
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

        NCCL rejects a collective while that communicator still reports
        ``ncclInProgress``. Every rank polls here before the next shared
        bootstrap barrier, including ranks that did not create the current
        reshard lane and therefore had no other readiness check in this step.
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
                    f"did not settle within {timeout_s:.1f}s"
                )
            try:
                _wait_until_initialized(lane.handle, bindings, remaining_s)
            except TimeoutError as error:
                raise TimeoutError(
                    f"lane {key.lane_id} of group {group_id} at epoch {epoch} "
                    f"did not settle within {timeout_s:.1f}s"
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
        for key in stale:
            lane = self._lanes.pop(key, None)
            if lane is not None:
                lane.abort()
        return len(stale)

    def abort_group(self, group_id: str) -> int:
        """Abort every lane of a group, not just the one that failed.

        A partially aborted group leaves peers blocked in the lanes that did
        not time out, waiting on ranks that have already given up.
        """
        keys = [key for key in self._lanes if key.group_id == group_id]
        for key in keys:
            lane = self._lanes.pop(key)
            lane.abort()
        return len(keys)

    def __len__(self) -> int:
        return len(self._lanes)
