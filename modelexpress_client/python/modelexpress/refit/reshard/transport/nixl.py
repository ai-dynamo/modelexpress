# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""NIXL-backed ``Transport`` for the reshard pull.

Wraps an initialized ``modelexpress.nixl_transfer.NixlTransferManager`` (MX's
own NIXL agent: lifecycle, memory registration, ``add_remote_agent``, RDMA READ)
and adapts it to the reshard ``Transport`` protocol. The reference
``InMemoryReferenceTransport`` and this share the identical
``(src_addr, dst_addr, nbytes)`` descriptor shape, so a plan validated in-memory
runs unchanged over RDMA here.

The manager is transport machinery only; peer discovery / agent-metadata
exchange happens out-of-band (see ``refit.reshard.rendezvous``), which yields the
``session -> remote agent name`` and ``session -> remote device id`` maps this
transport is constructed with. Each ``session`` is one published shard's owning
agent; descriptors are grouped per session into one batched READ.

Those per-session batches are posted **together** and awaited as a set, so a
receiver pulling from N trainers has N transfers in flight instead of draining
one peer before starting the next. Set ``MX_RESHARD_SERIAL_READS=1`` to restore
the one-peer-at-a-time behavior (baseline A/B only).
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any

logger = logging.getLogger("modelexpress.refit.reshard.transport.nixl")


def _serial_reads_enabled() -> bool:
    """Whether to drain each peer before posting the next. Read at call time so
    an A/B can toggle it without re-importing."""
    return os.environ.get("MX_RESHARD_SERIAL_READS", "0") not in ("", "0", "false")


class NixlReshardTransport:
    """A ``Transport`` that executes reshard pulls over NIXL RDMA READs.

    Args:
        manager: an initialized ``NixlTransferManager`` whose local destination
            params are registered and whose remote peers have been loaded via
            ``add_remote_agent``.
        session_to_agent: map from a ``ReadDescriptor.session`` to the remote
            NIXL agent name to READ from.
        session_to_device: map from session to the remote device id the shard
            lives on. Every agent session must have a matching device entry.
        mem_type: NIXL memory type override (defaults to the manager's
            accelerator mem type, e.g. VRAM).
        timeout_seconds: per-batch READ timeout.
    """

    def __init__(
        self,
        manager: Any,
        session_to_agent: dict,
        session_to_device: dict | None = None,
        mem_type: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self._manager = manager
        self._session_to_agent = session_to_agent
        self._session_to_device = session_to_device or {}
        self._mem_type = mem_type
        self._timeout = timeout_seconds
        self.bytes_moved = 0
        self.reads_issued = 0

    def read(self, descriptors: list) -> None:
        by_session: dict = defaultdict(list)
        for d in descriptors:
            by_session[d.session].append(d)
        if not by_session:
            return

        batches = [
            (self._agent_for(session), self._ranges_for(session, group))
            for session, group in by_session.items()
        ]

        if _serial_reads_enabled():
            for agent, ranges in batches:
                total_bytes, num_reads, _duration = self._manager.execute_read_batch(
                    remote_agent_name=agent,
                    ranges=ranges,
                    mem_type=self._mem_type,
                    timeout_seconds=self._timeout,
                )
                self.bytes_moved += total_bytes
                self.reads_issued += num_reads
            return

        posted: list = []
        try:
            for agent, ranges in batches:
                posted.append(
                    self._manager.post_read_batch(
                        remote_agent_name=agent,
                        ranges=ranges,
                        mem_type=self._mem_type,
                    )
                )
        except Exception:
            # Batches posted before the failure are in flight and still own
            # handles. Drain them so nothing leaks, but never let that cleanup
            # replace the original error.
            if posted:
                try:
                    self._manager.await_read_batches(
                        posted, timeout_seconds=self._timeout
                    )
                except Exception as exc:  # noqa: BLE001 - cleanup must not mask
                    logger.warning(
                        "draining %d posted READ batch(es) after a post failure "
                        "did not complete cleanly: %r",
                        len(posted),
                        exc,
                    )
            raise

        total_bytes, num_reads, _duration = self._manager.await_read_batches(
            posted, timeout_seconds=self._timeout
        )
        self.bytes_moved += total_bytes
        self.reads_issued += num_reads

    def _agent_for(self, session: Any) -> str:
        agent = self._session_to_agent.get(session)
        if agent is None:
            raise KeyError(f"no remote NIXL agent registered for session {session!r}")
        if session not in self._session_to_device:
            raise KeyError(f"no remote device id registered for session {session!r}")
        return agent

    def _ranges_for(self, session: Any, group: list) -> list:
        device_id = self._session_to_device[session]
        return [(d.src_addr, d.dst_addr, d.nbytes, device_id) for d in group]
