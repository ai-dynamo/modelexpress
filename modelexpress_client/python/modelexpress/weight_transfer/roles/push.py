# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PushRole: trainer pushes updated weights to inference workers.

In PUSH mode the trainer holds the active role: build a push plan from the
InferenceTable (inference workers' live GPU addresses), then issue NIXL WRITEs
after each optimizer step.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from .base import WeightSyncRole
from ..protocol.types import (
    InferenceTable,
    InferenceShard,
    RdmaDescriptor,
    TrainerTable,
)
from ..transport.nixl_executor import NixlExecutor

if TYPE_CHECKING:
    from ...nixl_transfer import NixlTransferManager

logger = logging.getLogger("modelexpress.weight_transfer.push")


class PushRole(WeightSyncRole):
    """Trainer-side push of updated weights to inference workers."""

    def __init__(
        self,
        nixl_manager: NixlTransferManager,
        device_id: int,
        trainer_rank: int = 0,
        sync_timeout: float = 300.0,
    ) -> None:
        self._nixl_manager = nixl_manager
        self._device_id = device_id
        self._trainer_rank = trainer_rank
        self._sync_timeout = sync_timeout

        self._descriptors: list[RdmaDescriptor] = []
        self._executor: NixlExecutor | None = None

    def initialize(
        self,
        model: Any,
        inference_table: InferenceTable,
        trainer_table: TrainerTable | None = None,
    ) -> None:
        """Build the push plan from an InferenceTable."""
        t0 = time.perf_counter()

        local_params: dict[str, Any] = {}
        if hasattr(model, "named_parameters"):
            for name, param in model.named_parameters():
                if param.requires_grad:
                    local_params[name] = param

        self._descriptors = self._build_push_plan(local_params, inference_table)

        logger.info(
            "[Trainer %d] Push plan built: %d RDMA descriptors, %.2f GB (%.3fs)",
            self._trainer_rank,
            len(self._descriptors),
            sum(d.nbytes for d in self._descriptors) / 1e9,
            time.perf_counter() - t0,
        )

        remote_agents: dict[int, str] = {}
        for i, nixl_bytes in enumerate(inference_table.agents):
            if nixl_bytes:
                name = self._nixl_manager.add_remote_agent(nixl_bytes)
                remote_agents[i] = name

        self._executor = NixlExecutor(
            nixl_manager=self._nixl_manager,
            remote_agents=remote_agents,
            device_id=self._device_id,
            timeout=self._sync_timeout,
        )

    def sync(self) -> None:
        """Push current trainer parameter values to all registered inference workers."""
        if self._executor is None:
            raise RuntimeError("PushRole not initialized; call initialize() first")
        self._executor.execute(self._descriptors, operation="WRITE")

    def teardown(self) -> None:
        self._descriptors = []
        self._executor = None

    def reshard(
        self,
        model: Any,
        table: TrainerTable,
        executor: Any,
        tp_src: int,
        tp_dst: int,
    ) -> tuple[int, float]:
        """Send weights via the nccl_m2n collective (source side).

        Collective semantics: this MUST be co-called by every generator's
        ``PullRole.reshard()`` in the same step (engine-scheduled).  This trainer
        rank supplies the source tile; the destination dataPtr is NULL.  Delegates
        layout + staging to ``NcclM2nExecutor``.
        """
        from ..transport.nccl_m2n_executor import build_reshard_params

        params, window_bytes = build_reshard_params(model, table, tp_src, tp_dst)
        return executor.execute(params, window_bytes)

    def _build_push_plan(
        self,
        local_params: dict[str, Any],
        inference_table: InferenceTable,
    ) -> list[RdmaDescriptor]:
        """Build WRITE descriptors: trainer param addr -> inference shard addr."""
        descriptors: list[RdmaDescriptor] = []

        for inf_shard in inference_table.shards:
            param = local_params.get(inf_shard.param_name)
            if param is None:
                logger.debug(
                    "[Trainer %d] Parameter %s not in local params, skipping",
                    self._trainer_rank,
                    inf_shard.param_name,
                )
                continue

            local_size = param.numel() * param.element_size()
            if local_size != inf_shard.size_bytes:
                logger.warning(
                    "[Trainer %d] Size mismatch for %s: local=%d inf=%d, skipping",
                    self._trainer_rank,
                    inf_shard.param_name,
                    local_size,
                    inf_shard.size_bytes,
                )
                continue

            descriptors.append(RdmaDescriptor(
                agent_index=inf_shard.agent_index,
                src_addr=param.data_ptr(),
                dst_addr=inf_shard.device_addr,
                nbytes=inf_shard.size_bytes,
            ))

        return descriptors
