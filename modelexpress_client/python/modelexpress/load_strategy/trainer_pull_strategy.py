# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TrainerPullStrategy: load weight updates by pulling from a running trainer.

This strategy wraps PullRole from weight_transfer.roles.pull and plugs it
into the ModelExpress LoadStrategyChain.  Region routing runs client-side
via LocalPlanner.

Environment variables
---------------------
MX_TRAINER_TABLE_KEY     Redis / MX metadata key for the TrainerTable.
                         Required, no default: is_available() declines the
                         strategy when it is unset, so the historical
                         "mx:trainer_table:{model_name}" fallback is not
                         reachable through the strategy chain.
MX_TRAINER_SYNC_TIMEOUT  Seconds to wait for trainer table or pull ACK.
                         Default: 300

Without MX_TRAINER_TABLE_KEY the deployment has no trainer to pull from.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from ..adapter import EngineAdapter, StrategyFailed
from ..nixl_transfer import is_nixl_available
from .base import LoadContext, LoadResult, LoadStrategy, _as_load_result, _init_nixl_manager
from .. import envs
from ..weight_transfer.protocol.serialization import decode_trainer_table
from ..weight_transfer.roles.pull import PullRole
from ..weight_transfer.planner.local import LocalPlanner

if TYPE_CHECKING:
    from ..weight_transfer.protocol.types import TrainerTable
    from ..weight_transfer.engine.base import WeightLoaderAdapter as WtAdapter

logger = logging.getLogger("modelexpress.strategy_trainer_pull")


_DISABLED_VALUES = {"", "0", "false", "no", "off"}

# Bound the Redis fallback so it cannot outlive the table-fetch deadline.
_REDIS_TIMEOUT = 5.0


def _trainer_sync_configured() -> bool:
    """Return whether this deployment is set up to pull from a trainer.

    MX_TRAINER_TABLE_KEY names the table this deployment pulls from. With it
    unset there is no trainer to wait for, so the strategy must decline rather
    than spend the whole MX_TRAINER_SYNC_TIMEOUT budget polling for a table
    that never appears and starving the strategies behind it.
    """
    return bool((envs.MX_TRAINER_TABLE_KEY or "").strip())


def _trainer_table_key(model_name: str) -> str:
    # is_available() already declined the strategy if this is unset, so the
    # historical derived-name fallback is unreachable from the chain.  Kept as
    # a defensive default for direct callers rather than silently building a
    # key the deployment never configured.
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
        self._sync_failed: bool = False

    def is_available(self, ctx: LoadContext) -> bool:
        if not super().is_available(ctx):
            return False
        if not _trainer_sync_configured():
            logger.info(
                "[Worker %d] MX_TRAINER_TABLE_KEY is not set, skipping trainer pull",
                ctx.global_rank,
            )
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
            raise StrategyFailed(f"TrainerTable not available: {e}", mutated=False) from e

        if ctx.nixl_manager is None:
            ctx.nixl_manager = _init_nixl_manager(
                ctx.global_rank,
                ctx.device_id,
                "trainer-pull",
                accelerator_backend=ctx.accelerator_backend,
            )

        planner = LocalPlanner()

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

        A sync writes straight into live parameter memory, so a failure partway
        through leaves the model holding a mix of old and new weights.  There is
        no cheap undo for that, so the strategy latches into a failed state and
        refuses further syncs rather than layering another partial update on top
        of an already-inconsistent model.  Recovery is a reload: rollback() then
        load() again.
        """
        if self._pull_role is None:
            raise RuntimeError("TrainerPullStrategy not loaded; call load() first")
        if self._sync_failed:
            raise RuntimeError(
                "TrainerPullStrategy is in a failed state: a previous sync() threw "
                "partway through writing live parameter memory, so the model holds "
                "a mix of old and new weights. Reload before syncing again."
            )
        try:
            self._pull_role.sync()
        except Exception:
            self._sync_failed = True
            logger.error(
                "[Worker %d] Weight sync failed partway through; parameters may be "
                "partially updated. Refusing further syncs until reload.",
                ctx.global_rank,
            )
            raise

    def rollback(self, ctx: LoadContext) -> None:
        if ctx.nixl_manager is not None:
            try:
                ctx.nixl_manager.shutdown()
            except Exception as e:
                logger.warning("[Worker %d] NIXL shutdown error: %s", ctx.global_rank, e)
        ctx.nixl_manager = None
        self._pull_role = None
        self._sync_failed = False

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
        redis_url = envs.MX_REDIS_URL
        try:
            import redis as redis_lib
            # The caller polls us inside a MX_TRAINER_SYNC_TIMEOUT deadline loop;
            # an unbounded connect/read here would block past that deadline.
            r = redis_lib.from_url(
                redis_url,
                socket_connect_timeout=_REDIS_TIMEOUT,
                socket_timeout=_REDIS_TIMEOUT,
            )
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
        # Deliberately unguarded: post-processing (e.g. FP8 repack) is part of
        # making the pulled weights usable.  Swallowing a failure here would let
        # sync_and_post_process() report success over half-processed weights.
        if self._ctx.adapter is not None and hasattr(self._ctx.adapter, "process_weights_after_loading"):
            self._ctx.adapter.process_weights_after_loading(model)

    def post_push_hook(self, model: Any) -> None:
        pass
