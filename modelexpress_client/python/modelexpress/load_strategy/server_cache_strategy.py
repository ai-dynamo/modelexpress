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

from .. import model_prefetch, model_snapshot
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
            snapshot_path, cache_root = self._snapshot_path(ctx, repo_id)
            logger.info(
                f"[Worker {ctx.global_rank}] Fetching {repo_id} weights from "
                f"ModelExpress Server into {snapshot_path}"
            )
            self._install_weights(repo_id, snapshot_path, cache_root)
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

    def _snapshot_path(
        self, ctx: LoadContext, repo_id: str
    ) -> tuple[Path, Path | None]:
        """Return the snapshot the engine resolved, and the root holding it.

        The root comes back so the weight install can target it explicitly. It
        cannot be looked up later: this runs in a separate EngineCore process
        that never saw the prefetch, and the engine loads from the path in
        ``ModelConfig`` no matter which root this worker would default to. A
        root of None means nothing better than that default is known, which is
        the case for any path that does not follow the cache layout.
        """
        engine_path = getattr(ctx.model_config, "model", None)
        if engine_path:
            candidate = Path(str(engine_path))
            location = model_snapshot.snapshot_location(repo_id, candidate)
            if location is not None:
                cache_root, commit = location
                if candidate.is_dir():
                    return candidate, cache_root
                # The frontend resolved this path against its own cache; this
                # node has nothing there yet. Install the commit the directory
                # names, under the root the path carries -- the default root
                # would be a snapshot the engine is not going to read.
                return self._install_metadata(repo_id, commit, cache_root), cache_root
            if candidate.is_dir():
                return candidate, None

        revision = getattr(ctx.model_config, "revision", None)
        return self._install_metadata(repo_id, revision, None), None

    @staticmethod
    def _install_metadata(
        repo_id: str, revision: str | None, cache_root: Path | None
    ) -> Path:
        snapshot_path = model_prefetch.ensure_metadata(
            repo_id, revision, cache_directory=cache_root
        )
        if snapshot_path is None:
            raise StrategyFailed(
                f"No local snapshot for {repo_id} and metadata prefetch did not apply",
                mutated=False,
            )
        return snapshot_path

    @staticmethod
    def _install_weights(
        repo_id: str, snapshot_path: Path, cache_root: Path | None = None
    ) -> None:
        from ..model_client import ModelCacheClient

        with ModelCacheClient(
            cache_directory=cache_root,
            chunk_size=model_prefetch.configured_chunk_size(),
        ) as client:
            client.install_weight_files(repo_id, snapshot_path)


def _repo_id(ctx: LoadContext) -> str | None:
    """Resolve the repo id to ask the server for.

    ``identity.model_name`` is whatever the engine put in ModelConfig, which
    vLLM overwrites with the resolved local path while parsing engine args.
    model_prefetch keeps the mapping back to the original repo id.
    """
    return model_prefetch.repo_id_for(ctx.identity.model_name)
