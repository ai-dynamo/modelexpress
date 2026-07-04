# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute a globally-coordinated M2N plan via NIXL.

M2nExecutor accepts M2nDescriptors produced by M2nPlanner and executes the
receive side for this inference worker.  Descriptors are grouped by
src_agent_index (trainer rank) and issued as parallel NIXL READ handles.

When NIXL exposes a native many-to-many transfer API
(e.g. make_prepped_m2n_xfer), the inner loop can be replaced with a single
collective call for better NIC utilization.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from ..protocol.types import M2nDescriptor

if TYPE_CHECKING:
    from ...nixl_transfer import NixlTransferManager

logger = logging.getLogger("modelexpress.weight_transfer.m2n_executor")


class M2nExecutor:
    """Execute M2nDescriptors via NIXL RDMA READ.

    Args:
        nixl_manager: Initialized NixlTransferManager for this worker.
        remote_agents: Mapping from src_agent_index (trainer rank) to the
            NIXL remote agent name loaded via add_remote_agent().
        device_id: Local CUDA device index.
        timeout: Seconds to wait per-transfer before raising TimeoutError.
    """

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

    def execute(self, descriptors: list[M2nDescriptor]) -> tuple[int, float]:
        """Issue NIXL READ transfers for all M2N descriptors and wait.

        Descriptors are grouped by src_agent_index so each trainer rank gets
        one NIXL handle.  All handles are submitted before waiting, so trainer
        ranks transfer in parallel.

        When NIXL adds a native M2N collective API, replace the per-agent loop
        below with a single make_prepped_m2n_xfer call.

        Args:
            descriptors: Per-worker receive-side M2N descriptors from M2nPlanner.

        Returns:
            (total_bytes_transferred, elapsed_seconds)
        """
        if not descriptors:
            return 0, 0.0

        agent = self._manager._agent
        if agent is None:
            raise RuntimeError("NIXL agent not initialized")

        backends = self._manager._backends
        mem_type = self._manager._accelerator_backend.nixl_mem_type

        # Group by trainer rank (src_agent_index).
        by_src: dict[str, list[M2nDescriptor]] = {}
        for desc in descriptors:
            remote_name = self._remote_agents.get(desc.src_agent_index)
            if remote_name is None:
                logger.warning(
                    "No remote agent loaded for src_agent_index %d, skipping",
                    desc.src_agent_index,
                )
                continue
            by_src.setdefault(remote_name, []).append(desc)

        start = time.perf_counter()
        total_bytes = 0
        handles = []

        for remote_name, descs in by_src.items():
            src_list = [(d.src_addr, d.nbytes, 0) for d in descs]
            dst_list = [(d.dst_addr, d.nbytes, self._device_id) for d in descs]
            indices = list(range(len(descs)))

            src_prepped = agent.prep_xfer_dlist(
                agent_name=remote_name,
                xfer_list=src_list,
                mem_type=mem_type,
                backends=backends,
            )
            dst_prepped = agent.prep_xfer_dlist(
                agent_name="",
                xfer_list=dst_list,
                mem_type=mem_type,
                backends=backends,
            )
            handle = agent.make_prepped_xfer(
                operation="READ",
                local_xfer_side=dst_prepped,
                local_indices=indices,
                remote_xfer_side=src_prepped,
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
                    raise TimeoutError(f"NIXL M2N READ timed out after {self._timeout}s")
                status = agent.check_xfer_state(handle)
                if status in ("DONE", "SUCCESS"):
                    agent.release_xfer_handle(handle)
                    break
                if status in ("ERR", "ERROR", "FAIL"):
                    agent.release_xfer_handle(handle)
                    raise RuntimeError(f"NIXL M2N READ failed with status {status}")
                time.sleep(0.001)

        self._manager._accelerator_backend.synchronize(self._device_id)

        elapsed = time.perf_counter() - start
        gbps = (total_bytes * 8) / (elapsed * 1e9) if elapsed > 0 else 0.0
        logger.info(
            "M2N READ complete: %.2f GB in %.3fs (%.1f Gbps)",
            total_bytes / 1e9,
            elapsed,
            gbps,
        )
        return total_bytes, elapsed
