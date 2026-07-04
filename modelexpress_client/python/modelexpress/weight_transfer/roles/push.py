# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PushRole: trainer pushes updated weights to inference workers.

In PUSH mode the trainer holds the active role:

  1. initialize(inference_table)
       - Enumerate local parameter tensors (trainer side, already updated).
       - Build a push plan: for each inference shard, compute the src address
         in trainer memory that corresponds to the same parameter + row range.
       - Register inference worker NIXL agents.

  2. sync(local_params)
       - Execute pre-built plan via NixlExecutor (NIXL WRITE).
       - Optionally wait for acknowledgment from inference workers.

  3. teardown()
       - Shut down NIXL resources.

PUSH vs PULL
------------
PULL (PullRole): workers initiate, trainer is passive, trainer publishes
  TrainerTable, workers do bake+resolve+plan, workers issue NIXL READs.

PUSH (PushRole): trainer initiates, workers are passive, workers publish
  InferenceTable (their live parameter GPU addresses), trainer does the
  planning and issues NIXL WRITEs.
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
    """Trainer-side push of updated weights to inference workers.

    The trainer calls initialize() once (or whenever the inference topology
    changes) and sync() after each optimizer step that updates parameters.

    Args:
        nixl_manager: Initialized NixlTransferManager on the trainer.
        device_id: CUDA device index on the trainer.
        trainer_rank: Trainer rank index (used for plan_key + logging).
        sync_timeout: Seconds to wait for each NIXL WRITE batch.
    """

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
        """Build the push plan from an InferenceTable.

        Args:
            model: Trainer model (used to look up current parameter addresses).
            inference_table: Published by inference workers; contains their
                live GPU parameter addresses and NIXL metadata.
            trainer_table: If provided, used to validate that the trainer's
                current shard for each parameter aligns with what is expected.
        """
        t0 = time.perf_counter()

        # Collect local parameter addresses from the trainer model
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

        # Register inference worker NIXL agents
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

            # For PUSH, the trainer writes its full local shard to the inference
            # worker's parameter storage.  Sizes must match.
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
