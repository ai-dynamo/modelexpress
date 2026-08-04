# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute pre-built RDMA plans via NIXL.

NixlExecutor adapts a list of RdmaDescriptors onto NixlTransferManager's batched
READ API: descriptors are grouped by remote agent, each group is posted with
post_read_batch, and the whole set is awaited once. The manager owns handle
lifetime, polling and the device synchronize, so this file holds no NIXL calls
of its own.

Used by PullRole.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from ..protocol.types import RdmaDescriptor

if TYPE_CHECKING:
    from ...nixl_transfer import NixlTransferManager

logger = logging.getLogger("modelexpress.weight_transfer.transport")

# RdmaDescriptor has no remote-device field, so the trainer side is assumed to
# sit on device 0. Lifting this means adding the field to the descriptor and
# threading it through every planner that builds one.
REMOTE_DEVICE_ID = 0


class NixlExecutor:
    """Execute a grouped list of RdmaDescriptors via NIXL READ."""

    def __init__(
        self,
        nixl_manager: NixlTransferManager,
        remote_agents: dict[int, str],
        device_id: int,
        timeout: float = 300.0,
    ) -> None:
        # post_read_batch builds local descriptors from the manager's own device
        # id, so a divergence here would silently land weights on the wrong
        # device. Today TrainerPullStrategy passes the same ctx.device_id to
        # both and nothing else enforces it.
        if device_id != nixl_manager._device_id:
            raise ValueError(
                f"NixlExecutor device_id {device_id} does not match the "
                f"NixlTransferManager's {nixl_manager._device_id}; local RDMA "
                "descriptors are built from the manager's device id"
            )
        self._manager = nixl_manager
        self._remote_agents = remote_agents
        self._device_id = device_id
        self._timeout = timeout

    def execute(self, descriptors: list[RdmaDescriptor]) -> tuple[int, float]:
        """Issue NIXL READs for all descriptors and wait for completion."""
        if not descriptors:
            return 0, 0.0

        by_agent: dict[str, list[RdmaDescriptor]] = {}
        for desc in descriptors:
            remote_name = self._remote_agents.get(desc.agent_index)
            if remote_name is None:
                logger.warning(
                    "No remote agent loaded for agent_index %d, skipping",
                    desc.agent_index,
                )
                continue
            by_agent.setdefault(remote_name, []).append(desc)

        start = time.perf_counter()
        posted: list = []
        try:
            for remote_name, descs in by_agent.items():
                posted.append(
                    self._manager.post_read_batch(
                        remote_agent_name=remote_name,
                        ranges=[
                            (d.src_addr, d.dst_addr, d.nbytes, REMOTE_DEVICE_ID)
                            for d in descs
                        ],
                    )
                )
        except Exception:
            # Earlier batches are already in flight and own handles nobody else
            # will release. Drain them, but never let a drain failure mask the
            # post failure that got us here.
            try:
                self._manager.await_read_batches(
                    posted, self._timeout, label="NIXL weight-sync READ batch"
                )
            except Exception as exc:  # noqa: BLE001 - cleanup must not mask the cause
                logger.warning("Drain after a failed READ post: %r", exc)
            raise

        total_bytes, _num_reads, _duration = self._manager.await_read_batches(
            posted, self._timeout, label="NIXL weight-sync READ batch"
        )

        elapsed = time.perf_counter() - start
        gbps = (total_bytes * 8) / (elapsed * 1e9) if elapsed > 0 else 0.0
        logger.info(
            "READ complete: %.2f GB in %.3fs (%.1f Gbps)",
            total_bytes / 1e9,
            elapsed,
            gbps,
        )
        return total_bytes, elapsed
