# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TrainerPullStrategy: load weight updates by pulling from a running trainer.

This strategy wraps PullRole from weight_transfer.roles.pull and plugs it
into the ModelExpress LoadStrategyChain.  It selects between LocalPlanner
(no server) and ServerPlanner (server-side routing + plan caching) based
on whether MX_WEIGHT_SYNC_SERVER is set.

Environment variables
---------------------
MX_TRAINER_TABLE_KEY     Redis / MX metadata key for the TrainerTable.
                         Default: "mx:trainer_table:{model_name}"
MX_TRAINER_SYNC_TIMEOUT  Seconds to wait for trainer table or pull ACK.
                         Default: 300
MX_WEIGHT_SYNC_SERVER    If set, use ServerPlanner (routes regions on the
                         MX server in Rust, caches plan for all workers).
                         If unset, use LocalPlanner (client-side routing).
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

from ..adapter import EngineAdapter, StrategyFailed
from ..nixl_transfer import is_nixl_available
from .base import LoadContext, LoadResult, LoadStrategy, _as_load_result, _init_nixl_manager
from .. import envs
from ..weight_transfer.protocol.serialization import decode_trainer_table
from ..weight_transfer.roles.pull import PullRole
from ..weight_transfer.planner.local import LocalPlanner
from ..weight_transfer.planner.server import ServerPlanner

if TYPE_CHECKING:
    from ..weight_transfer.protocol.types import TrainerTable
    from ..weight_transfer.engine.base import WeightLoaderAdapter as WtAdapter

logger = logging.getLogger("modelexpress.strategy_trainer_pull")


def _use_server_planner() -> bool:
    return bool(os.environ.get("MX_WEIGHT_SYNC_SERVER", ""))


def _trainer_table_key(model_name: str) -> str:
    key = envs.MX_TRAINER_TABLE_KEY
    if key:
        return key
    safe = model_name.replace("/", "_").replace(":", "_")
    return f"mx:trainer_table:{safe}"


class TrainerPullStrategy(LoadStrategy):
    """Pull live weight updates from a sharded trainer via NIXL RDMA.

    Placed at P0 in the strategy chain.  Falls through to the next strategy
    (RdmaStrategy / ModelStreamer / GDS / Default) if no trainer is active.

    Subsequent weight syncs (after initial load) are driven by calling
    update_weights() directly on this strategy instance -- the chain does
    not need to run again.
    """

    name = "trainer_pull"
    requires = (EngineAdapter.discover_tensors,)

    def __init__(self) -> None:
        self._pull_role: PullRole | None = None

    def is_available(self, ctx: LoadContext) -> bool:
        if not super().is_available(ctx):
            return False
        if not is_nixl_available():
            return False
        if not ctx.accelerator_backend.supports_rdma_p2p():
            return False
        if ctx.mx_client is None:
            return False
        return True

    def load(self, result: LoadResult, ctx: LoadContext) -> LoadResult:
        result = _as_load_result(result)

        try:
            table = self._fetch_table(ctx)
        except Exception as e:
            logger.info("[Worker %d] TrainerTable not available: %s", ctx.global_rank, e)
            raise StrategyFailed(f"TrainerTable not available: {e}", mutated=False)

        if ctx.nixl_manager is None:
            ctx.nixl_manager = _init_nixl_manager(
                ctx.global_rank,
                ctx.device_id,
                "trainer-pull",
                accelerator_backend=ctx.accelerator_backend,
            )

        planner = (
            ServerPlanner(mx_client=ctx.mx_client)
            if _use_server_planner()
            else LocalPlanner()
        )

        # Build the engine adapter from the existing ctx adapter
        wt_adapter = _CtxEngineAdapter(ctx)

        self._pull_role = PullRole(
            adapter=wt_adapter,
            nixl_manager=ctx.nixl_manager,
            device_id=ctx.device_id,
            worker_rank=ctx.global_rank,
            planner=planner,
            sync_timeout=float(envs.MX_TRAINER_SYNC_TIMEOUT),
        )

        try:
            self._pull_role.initialize(result.model, table)
        except Exception as e:
            raise StrategyFailed(f"PullRole init failed: {e}", mutated=True) from e

        try:
            self._pull_role.sync()
        except Exception as e:
            raise StrategyFailed(f"Initial RDMA pull failed: {e}", mutated=True) from e

        return result

    def update_weights(self, ctx: LoadContext) -> None:
        """Execute a weight sync using the pre-built static plan.

        Called by the vLLM worker after each training step notification.
        Raises RuntimeError if load() has not been called successfully.
        """
        if self._pull_role is None:
            raise RuntimeError("TrainerPullStrategy not loaded; call load() first")
        self._pull_role.sync()

    def rollback(self, ctx: LoadContext) -> None:
        if ctx.nixl_manager is not None:
            try:
                ctx.nixl_manager.shutdown()
            except Exception as e:
                logger.warning("[Worker %d] NIXL shutdown error: %s", ctx.global_rank, e)
        ctx.nixl_manager = None
        self._pull_role = None

    def _fetch_table(self, ctx: LoadContext) -> TrainerTable:
        key = _trainer_table_key(ctx.identity.model_name)
        timeout = envs.MX_TRAINER_SYNC_TIMEOUT
        deadline = time.monotonic() + timeout

        logger.info(
            "[Worker %d] Waiting for TrainerTable at %r (timeout=%ds)",
            ctx.global_rank,
            key,
            timeout,
        )

        while time.monotonic() < deadline:
            raw = self._read_raw(ctx, key)
            if raw:
                table = decode_trainer_table(raw)
                logger.info(
                    "[Worker %d] TrainerTable fetched: %d tensors, %d agents",
                    ctx.global_rank,
                    len(table.tensors),
                    len(table.agents),
                )
                return table
            time.sleep(1.0)

        raise TimeoutError(f"TrainerTable not found at {key!r} after {timeout}s")

    def _read_raw(self, ctx: LoadContext, key: str) -> bytes | None:
        # Try the MX WeightSyncService first (GetTrainerTable RPC)
        try:
            resp = ctx.mx_client.get_trainer_table(model_key=key)
            if resp and resp.found:
                return resp.table_payload
        except AttributeError:
            pass
        except Exception as e:
            logger.debug("[Worker %d] GetTrainerTable RPC failed: %s", ctx.global_rank, e)

        # Redis fallback
        redis_url = envs.MX_REDIS_URL
        try:
            import redis as redis_lib
            r = redis_lib.from_url(redis_url)
            return r.get(key)
        except ImportError:
            logger.debug("[Worker %d] redis-py not installed", ctx.global_rank)
        except Exception as e:
            logger.debug("[Worker %d] Redis GET failed: %s", ctx.global_rank, e)

        return None


class _CtxEngineAdapter:
    """Bridge from LoadContext adapter to WeightLoaderAdapter interface."""

    def __init__(self, ctx: LoadContext) -> None:
        self._ctx = ctx

    def iter_lazy_weights(self, table: TrainerTable):
        from ..weight_transfer.engine.lazy import LazyWeight
        import torch
        for tt in table.tensors:
            yield (
                tt.name,
                LazyWeight(
                    name=tt.name,
                    shape=torch.Size(tt.shape),
                    dtype=getattr(torch, tt.dtype.replace("torch.", "")),
                ),
            )

    def iter_param_shards(self, model: Any):
        if hasattr(model, "named_parameters"):
            yield from model.named_parameters()

    def post_pull_hook(self, model: Any) -> None:
        if self._ctx.adapter is not None and hasattr(self._ctx.adapter, "process_weights_after_loading"):
            try:
                self._ctx.adapter.process_weights_after_loading(model)
            except Exception as e:
                logger.warning("post_pull_hook failed: %s", e)

    def post_push_hook(self, model: Any) -> None:
        pass
