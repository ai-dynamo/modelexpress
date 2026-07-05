# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute pre-built RDMA plans via NIXL.

NixlExecutor wraps a NixlTransferManager and executes a list of
RdmaDescriptors as one batched NIXL READ or WRITE, grouped by remote agent.
Shared by PullRole (READ) and PushRole (WRITE).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Literal

from ..protocol.types import RdmaDescriptor

if TYPE_CHECKING:
    from ...nixl_transfer import NixlTransferManager

logger = logging.getLogger("modelexpress.weight_transfer.transport")

NixlOperation = Literal["READ", "WRITE"]


class NixlExecutor:
    """Execute a grouped list of RdmaDescriptors via NIXL."""

    def __init__(
        self,
        nixl_manager: NixlTransferManager,
        remote_agents: dict[int, str],
        device_id: int,
        timeout: float = 300.0,
    ) -> None:
        self._manager = nixl_manager
        self._remote_agents = remote_agents
        self._device_id = device_id
        self._timeout = timeout

    def execute(
        self,
        descriptors: list[RdmaDescriptor],
        operation: NixlOperation = "READ",
    ) -> tuple[int, float]:
        """Issue NIXL transfers for all descriptors and wait for completion."""
        if not descriptors:
            return 0, 0.0

        agent = self._manager._agent
        if agent is None:
            raise RuntimeError("NIXL agent not initialized")

        backends = self._manager._backends
        mem_type = self._manager._accelerator_backend.nixl_mem_type

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
        total_bytes = 0
        handles = []

        for remote_name, descs in by_agent.items():
            if operation == "READ":
                src_list = [(d.src_addr, d.nbytes, 0) for d in descs]
                dst_list = [(d.dst_addr, d.nbytes, self._device_id) for d in descs]
            else:
                src_list = [(d.src_addr, d.nbytes, self._device_id) for d in descs]
                dst_list = [(d.dst_addr, d.nbytes, 0) for d in descs]

            indices = list(range(len(descs)))
            src_prepped = agent.prep_xfer_dlist(
                agent_name=remote_name if operation == "READ" else "",
                xfer_list=src_list,
                mem_type=mem_type,
                backends=backends,
            )
            dst_prepped = agent.prep_xfer_dlist(
                agent_name="" if operation == "READ" else remote_name,
                xfer_list=dst_list,
                mem_type=mem_type,
                backends=backends,
            )
            handle = agent.make_prepped_xfer(
                operation=operation,
                local_xfer_side=dst_prepped if operation == "READ" else src_prepped,
                local_indices=indices,
                remote_xfer_side=src_prepped if operation == "READ" else dst_prepped,
                remote_indices=indices,
                backends=backends,
            )
            agent.transfer(handle)
            handles.append(handle)
            total_bytes += sum(d.nbytes for d in descs)

        wait_start = time.monotonic()
        for handle in handles:
            while True:
                if time.monotonic() - wait_start >= self._timeout:
                    agent.release_xfer_handle(handle)
                    raise TimeoutError(f"NIXL {operation} timed out after {self._timeout}s")
                status = agent.check_xfer_state(handle)
                if status in ("DONE", "SUCCESS"):
                    agent.release_xfer_handle(handle)
                    break
                if status in ("ERR", "ERROR", "FAIL"):
                    agent.release_xfer_handle(handle)
                    raise RuntimeError(f"NIXL {operation} failed with status {status}")
                time.sleep(0.001)

        self._manager._accelerator_backend.synchronize(self._device_id)

        elapsed = time.perf_counter() - start
        gbps = (total_bytes * 8) / (elapsed * 1e9) if elapsed > 0 else 0.0
        logger.info(
            "%s complete: %.2f GB in %.3fs (%.1f Gbps)",
            operation,
            total_bytes / 1e9,
            elapsed,
            gbps,
        )
        return total_bytes, elapsed
