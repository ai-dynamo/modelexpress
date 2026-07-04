# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PullRole: inference worker pulls live weights from a sharded trainer.

Lifecycle
---------
  1. initialize(model, trainer_table)
       - Run bake pass (lazy load_weights to record op chains).
       - Resolve op chains to ResolvedRegions (torch, client-side).
       - Build RdmaDescriptor plan via the configured planner
         (LocalPlanner or ServerPlanner).
       - Load trainer NIXL agent metadata and register remote agents.

  2. sync() -- repeat each training step
       - Execute pre-built plan via NixlExecutor (NIXL READ).
       - Call engine adapter's post_pull_hook() for quantization repack.

  3. teardown()
       - Shut down NIXL resources.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

import torch

from .base import WeightSyncRole
from ..engine.lazy import bake_model
from ..planner.resolver import resolve_copies
from ..planner.local import LocalPlanner
from ..protocol.types import RdmaDescriptor, TrainerTable
from ..transport.nixl_executor import NixlExecutor

if TYPE_CHECKING:
    from ..engine.base import WeightLoaderAdapter
    from ..planner.base import AbstractPlanner
    from ...nixl_transfer import NixlTransferManager

logger = logging.getLogger("modelexpress.weight_transfer.pull")


class PullRole(WeightSyncRole):
    """Inference-worker side of trainer -> inference weight sync.

    Args:
        adapter: Engine-specific weight loader adapter.
        planner: LocalPlanner (default) or ServerPlanner.
        nixl_manager: Initialized NixlTransferManager for this worker.
        device_id: CUDA device index for this worker.
        worker_rank: Global worker rank (used for the plan_key).
        sync_timeout: Seconds to wait for each NIXL READ batch.
    """

    def __init__(
        self,
        adapter: WeightLoaderAdapter,
        nixl_manager: NixlTransferManager,
        device_id: int,
        worker_rank: int = 0,
        planner: AbstractPlanner | None = None,
        sync_timeout: float = 300.0,
    ) -> None:
        self._adapter = adapter
        self._nixl_manager = nixl_manager
        self._device_id = device_id
        self._worker_rank = worker_rank
        self._planner = planner or LocalPlanner()
        self._sync_timeout = sync_timeout

        self._descriptors: list[RdmaDescriptor] = []
        self._executor: NixlExecutor | None = None
        self._plan_key: str = ""
        self._current_step: int = -1

    def initialize(self, model: Any, table: TrainerTable) -> None:
        """Bake, resolve, plan, and register remote NIXL agents."""
        t0 = time.perf_counter()

        # Build per-tensor shape and dtype maps for the resolver
        tensor_shapes = {tt.name: tuple(tt.shape) for tt in table.tensors}
        tensor_dtypes = {
            tt.name: getattr(torch, tt.dtype.replace("torch.", ""))
            for tt in table.tensors
        }

        # Bake: drive load_weights with LazyWeights, capture op chains
        weight_iter = self._adapter.iter_lazy_weights(table)
        copies = bake_model(model, weight_iter)

        # Resolve: op chains -> ResolvedRegions (torch-dependent, client-side)
        regions = resolve_copies(copies, tensor_shapes, tensor_dtypes)
        logger.info(
            "[Worker %d] Resolved %d regions from %d copies (%.3fs)",
            self._worker_rank,
            len(regions),
            len(copies),
            time.perf_counter() - t0,
        )

        # Plan key is stable across steps for the same model + worker
        self._plan_key = f"{id(model)}-rank{self._worker_rank}"

        # Build plan (local or server-side routing)
        self._descriptors = self._planner.build(regions, table, self._plan_key)
        logger.info(
            "[Worker %d] Plan built: %d RDMA descriptors, %.2f GB total",
            self._worker_rank,
            len(self._descriptors),
            sum(d.nbytes for d in self._descriptors) / 1e9,
        )

        # Register trainer NIXL agents
        remote_agents: dict[int, str] = {}
        for i, nixl_bytes in enumerate(table.agents):
            if nixl_bytes:
                name = self._nixl_manager.add_remote_agent(nixl_bytes)
                remote_agents[i] = name

        self._executor = NixlExecutor(
            nixl_manager=self._nixl_manager,
            remote_agents=remote_agents,
            device_id=self._device_id,
            timeout=self._sync_timeout,
        )
        self._current_step = table.step

    def sync(self) -> None:
        """Execute one PULL using the pre-built plan."""
        if self._executor is None:
            raise RuntimeError("PullRole not initialized; call initialize() first")
        self._executor.execute(self._descriptors, operation="READ")

    def sync_and_post_process(self, model: Any) -> None:
        """PULL then run the engine's post_pull_hook (e.g. FP8 repack)."""
        self.sync()
        self._adapter.post_pull_hook(model)

    def refresh(self, model: Any, table: TrainerTable) -> None:
        """Refresh the plan when the trainer reshards (new step with shape change).

        If the training step changed but the shard layout is the same, sync()
        is all that is needed.  Call refresh() only when trainer ranks change.
        """
        if table.step == self._current_step:
            return
        self._planner.invalidate(self._plan_key)
        self.initialize(model, table)

    def teardown(self) -> None:
        self._descriptors = []
        self._executor = None
        self._plan_key = ""
