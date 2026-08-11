# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server cache loading strategy: fetch weights from ModelExpress Server.

Runs after RdmaStrategy, so a live P2P source is always preferred. This is the
cold-miss path: no peer is serving the model, the local cache holds only the
metadata installed before the engine started, and the worker has no route to
Hugging Face of its own.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .. import model_prefetch
from ..adapter import EngineAdapter, StrategyFailed
from .base import LoadContext, LoadStrategy, _as_load_result, register_tensors
from .context import LoadResult

logger = logging.getLogger("modelexpress.strategy_server_cache")


class ServerCacheStrategy(LoadStrategy):
    """Install weights from the server, then load them with the engine's loader."""

    name = "server-cache"
    requires = (EngineAdapter.load_via_native,)

    def is_available(self, ctx: LoadContext) -> bool:
        """Return whether the server can supply weights for this model.

        Needs the no-shared-storage switch, a server address, and a Hugging
        Face repo id. The repo id may have to be recovered from the resolved
        cache path, because the engine rewrites the model name in place and
        loads weights in a process that never ran the prefetch.
        """
        if not super().is_available(ctx):
            return False
        if not model_prefetch.is_enabled():
            return False
        if _repo_id(ctx) is None:
            logger.info(
                f"[Worker {ctx.global_rank}] No Hugging Face repo id for "
                f"{ctx.identity.model_name!r}, skipping server cache"
            )
            return False
        return True

    def load(self, result: LoadResult, ctx: LoadContext) -> LoadResult:
        """Stream the weights into the resolved snapshot, then load natively.

        Raises :class:`StrategyFailed` with ``mutated=False`` while the model
        is still untouched, so the chain can try the next strategy, and with
        ``mutated=True`` once the engine's own loader has started writing into
        it and only a reinit can recover.
        """
        result = _as_load_result(result)
        if ctx.adapter is None:
            raise StrategyFailed(
                "ModelExpress Server cache requires an engine adapter", mutated=False
            )

        repo_id = _repo_id(ctx)
        if repo_id is None:
            raise StrategyFailed("No Hugging Face repo id for this model", mutated=False)

        try:
            snapshot_path = self._snapshot_path(ctx, repo_id)
            logger.info(
                f"[Worker {ctx.global_rank}] Fetching {repo_id} weights from "
                f"ModelExpress Server into {snapshot_path}"
            )
            self._install_weights(repo_id, snapshot_path)
        except StrategyFailed:
            raise
        except Exception as exc:
            raise StrategyFailed(
                f"ModelExpress Server cache failed: {exc}", mutated=False
            ) from exc

        try:
            result = ctx.adapter.load_via_native(result)
            result = ctx.adapter.after_native_load(result)
        except Exception as exc:
            raise StrategyFailed(str(exc), mutated=True) from exc

        register_tensors(result, ctx)
        return result

    def _snapshot_path(self, ctx: LoadContext, repo_id: str) -> Path:
        """Return the snapshot the engine resolved, installing it if needed."""
        engine_path = getattr(ctx.model_config, "model", None)
        if engine_path:
            candidate = Path(str(engine_path))
            if candidate.is_dir():
                return candidate

        revision = getattr(ctx.model_config, "revision", None)
        snapshot_path = model_prefetch.ensure_metadata(repo_id, revision)
        if snapshot_path is None:
            raise StrategyFailed(
                f"No local snapshot for {repo_id} and metadata prefetch did not apply",
                mutated=False,
            )
        return snapshot_path

    @staticmethod
    def _install_weights(repo_id: str, snapshot_path: Path) -> None:
        from ..model_client import ModelCacheClient

        with ModelCacheClient(
            chunk_size=model_prefetch.configured_chunk_size()
        ) as client:
            client.install_weight_files(repo_id, snapshot_path)


def _repo_id(ctx: LoadContext) -> str | None:
    """Resolve the repo id to ask the server for.

    ``identity.model_name`` is whatever the engine put in ModelConfig, which
    vLLM overwrites with the resolved local path while parsing engine args.
    model_prefetch keeps the mapping back to the original repo id.
    """
    return model_prefetch.repo_id_for(ctx.identity.model_name)
