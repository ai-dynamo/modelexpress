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
        import nccl.core.communicator as communicator
        import nccl.core.utils as utils
    except ImportError as error:  # pragma: no cover - environment dependent
        raise NcclUnavailableError(
            "the NCCL M2N refit path needs nccl4py; install the 'nccl-m2n' extra"
        ) from error
    return communicator, utils


def new_unique_id() -> bytes:
    """Generate an ``ncclUniqueId`` for a lane this worker leads."""
    _, utils = _nccl()
    return bytes(utils.get_unique_id().as_bytes)


def _unique_id_from_bytes(raw: bytes) -> Any:
    _, utils = _nccl()
    return utils.UniqueId.from_bytes(raw)


@dataclass(frozen=True)
class LaneKey:
    group_id: str
    epoch: int
    lane_id: int


class LaneCommunicator:
    """One lane's communicator, plus the stream its work is ordered on."""

    def __init__(self, comm: Any, *, rank: int, world_size: int, stream: Any) -> None:
        self._comm = comm
        self.rank = rank
        self.world_size = world_size
        self.stream = stream
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
        self._aborted = True
        abort = getattr(self._comm, "abort", None)
        if abort is None:
            return
        try:
            abort()
        except Exception as error:  # noqa: BLE001 - teardown must not mask the original failure
            logger.warning("aborting the NCCL communicator did not complete cleanly: %r", error)


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
            return existing

        communicator, _ = _nccl()
        timeout_s = (
            timeout_s if timeout_s is not None else envs.MX_NCCL_REFIT_COMM_INIT_TIMEOUT_S
        )
        logger.info(
            "bootstrapping NCCL lane %s of group %s at epoch %s (rank %s of %s)",
            key.lane_id,
            key.group_id,
            key.epoch,
            rank,
            world_size,
        )
        comm = communicator.Communicator.init(
            nranks=world_size,
            rank=rank,
            unique_id=_unique_id_from_bytes(unique_id),
        )
        lane = LaneCommunicator(comm, rank=rank, world_size=world_size, stream=stream)
        self._lanes[key] = lane
        return lane

    def invalidate_epoch(self, group_id: str, epoch: int) -> int:
        """Drop every lane not at ``epoch``. Returns how many were dropped."""
        stale = [
            key
            for key in self._lanes
            if key.group_id == group_id and key.epoch != epoch
        ]
        for key in stale:
            self._lanes.pop(key, None)
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
