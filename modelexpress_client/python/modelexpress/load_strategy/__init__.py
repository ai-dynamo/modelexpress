# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
load_strategy: prioritized chain of model loading strategies.

Detects the environment and builds an ordered list of eligible loaders.
MxModelLoader iterates the chain until one succeeds.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import torch.nn as nn

from modelexpress.tracing import tracer

from ..adapter import StrategyFailed, StrategyRecoveryError, UnsupportedCapability
from .base import (
    LoadContext,
    LoadResult,
    LoadStrategy,
    SourceTransferError,
    clear_exception_tracebacks,
    drain_tensor_readers,
    publish_source_if_supported,
    register_tensors,
    publish_metadata,
    unpublish_metadata,
)

__all__ = [
    "LoadContext",
    "LoadResult",
    "LoadStrategy",
    "StrategyFailed",
    "StrategyRecoveryError",
    "LoadStrategyChain",
    "execute_load_strategies",
    "run_load_strategy_chain",
    "SourceTransferError",
    "drain_tensor_readers",
    "register_tensors",
    "publish_metadata",
    "unpublish_metadata",
]

logger = logging.getLogger("modelexpress.load_strategy")


class LoadStrategyChain:
    """Prioritized chain of model loading strategies.

    Detects the environment, builds an ordered list of eligible loaders,
    and runs them until one succeeds.
    """

    @staticmethod
    def run(model: nn.Module, ctx: LoadContext) -> nn.Module:
        """Build the chain and execute strategies until one succeeds.

        Strategies return LoadResult on success. Expected misses raise
        StrategyFailed; mutated failures trigger adapter re-initialization
        before the next strategy runs. Unexpected exceptions are rolled back
        and treated as fallback to preserve the existing chain behavior.
        After each strategy the engine ranks agree on its outcome, so they all
        succeed or all fall back together.

        Returns the (possibly re-initialized) model on success.
        Raises RuntimeError if no strategy succeeds.
        """
        from .rdma_strategy import RdmaStrategy
        from .server_cache_strategy import ServerCacheStrategy
        from .instant_tensor_strategy import InstantTensorStrategy
        from .model_streamer_strategy import ModelStreamerStrategy
        from .gds_strategy import GdsStrategy
        from .default_strategy import DefaultStrategy
        from .. import envs
        from ..plugins import get_plugin, initialize_plugins, validate_plugin_names

        all_strategies: list[LoadStrategy] = [
            RdmaStrategy(),
            ServerCacheStrategy(),
            InstantTensorStrategy(),
            ModelStreamerStrategy(),
            GdsStrategy(),
            DefaultStrategy(),
        ]

        if name := envs.MX_LOAD_PLUGIN:
            before = envs.MX_LOAD_PLUGIN_BEFORE
            strategy_names = [strategy.name for strategy in all_strategies]
            if before not in strategy_names:
                raise ValueError(
                    f"MX_LOAD_PLUGIN_BEFORE must name a built-in strategy "
                    f"from {strategy_names}, got {before!r}"
                )
            initialize_plugins()
            validate_plugin_names(set(strategy_names))
            all_strategies.insert(strategy_names.index(before), get_plugin(name))
        return execute_load_strategies(model, ctx, all_strategies)

    @staticmethod
    def _reinit_for_retry(
        result: LoadResult,
        ctx: LoadContext,
        strategy: LoadStrategy,
    ) -> LoadResult:
        if ctx.adapter is None:
            raise RuntimeError(
                f"[Worker {ctx.global_rank}] Strategy '{strategy.name}' mutated "
                "the model but no adapter can reinitialize it"
            )
        try:
            return ctx.adapter.reinit_for_retry(result)
        except UnsupportedCapability as exc:
            raise RuntimeError(
                f"[Worker {ctx.global_rank}] Strategy '{strategy.name}' mutated "
                "the model but adapter does not support retry reinitialization"
            ) from exc


@contextmanager
def _strategy_phases(
    strategy: LoadStrategy,
    result: LoadResult,
    ctx: LoadContext,
):
    """Wrap one attempt in the strategy's prepare() and finalize().

    The chain only sequences the phases. What each one does, including how it
    differs on a reload, is the strategy's own decision. finalize() sits after
    the yield rather than in a finally, because a failed attempt has nothing
    to bring to a final layout; abort() runs instead, to undo whatever
    prepare() started.

    The phases work on ``result`` in place: LoadResult is the stable envelope
    the chain holds for the whole attempt, which is why reinit_for_retry
    copies a replacement's state back into it rather than handing one out.
    """
    strategy.prepare(result, ctx)
    try:
        yield
    except BaseException:
        strategy.abort(result, ctx)
        raise
    strategy.finalize(result, ctx)


_LOADED = "loaded"
_FAILED = "failed"
_FATAL = "fatal"


def _eligible_on_every_rank(
    strategies: list[LoadStrategy], ctx: LoadContext
) -> list[LoadStrategy]:
    """Return the available strategies that every engine rank can run.

    Availability can depend on the node (GDS hardware, NIXL, local files), but
    the ranks must walk the same strategies to agree on each outcome. A strategy
    that some rank cannot run is skipped on every rank. With nothing eligible on
    any rank, the chain reports that no strategy succeeded, as before.
    """
    eligible = [strategy for strategy in strategies if strategy.is_available(ctx)]
    names = [strategy.name for strategy in eligible]
    state = ("eligible", names)
    states = ctx.adapter.all_gather_state(state) if ctx.adapter is not None else (state,)
    for rank, (peer_phase, _peer_names) in enumerate(states):
        if peer_phase != "eligible":
            raise StrategyRecoveryError(
                f"[Worker {ctx.global_rank}] Ranks are out of step in the loading "
                f"chain: this rank is at 'eligible', rank {rank} at {peer_phase!r}"
            )
    per_rank = [peer_names for _peer_phase, peer_names in states]
    common = set(names).intersection(*per_rank)
    shared = [strategy for strategy in eligible if strategy.name in common]
    if not shared and any(per_rank):
        raise RuntimeError(
            f"[Worker {ctx.global_rank}] No loading strategy is eligible on every "
            f"rank for model '{ctx.identity.model_name}'; eligible per rank: "
            f"{per_rank}"
        )
    if len(shared) < len(eligible):
        logger.warning(
            f"[Worker {ctx.global_rank}] Skipping loaders that are not eligible on "
            f"every rank: {[name for name in names if name not in common]}"
        )
    return shared


def _gather_load_outcomes(
    outcome: str, strategy: LoadStrategy, ctx: LoadContext
) -> list[str]:
    """Share this rank's load outcome for ``strategy`` and return every rank's."""
    phase = f"outcome:{strategy.name}"
    state = (phase, outcome)
    states = ctx.adapter.all_gather_state(state) if ctx.adapter is not None else (state,)
    for rank, (peer_phase, _peer_outcome) in enumerate(states):
        if peer_phase != phase:
            raise StrategyRecoveryError(
                f"[Worker {ctx.global_rank}] Ranks are out of step in the loading "
                f"chain: this rank is at {phase!r}, rank {rank} at {peer_phase!r}"
            )
    return [peer_outcome for _peer_phase, peer_outcome in states]


def execute_load_strategies(
    model: nn.Module,
    ctx: LoadContext,
    strategies: list[LoadStrategy],
) -> nn.Module:
    """Execute an ordered policy using the common fallback lifecycle."""
    eligible = _eligible_on_every_rank(strategies, ctx)
    logger.info(f"Eligible loaders: {[strategy.name for strategy in eligible]}")

    result = LoadResult(value=model, model=model)
    with tracer.start_as_current_span("Load model") as span:
        span.set_attribute("model_name", ctx.identity.model_name)
        span.set_attribute("global_rank", ctx.global_rank)
        span.set_attribute("eligible_strategies", [s.name for s in eligible])

        for strategy in eligible:
            logger.info(f"[Worker {ctx.global_rank}] Trying strategy: {strategy.name}")
            try:
                with _strategy_phases(strategy, result, ctx):
                    # Every rank leaves this strategy the same way. Later strategies
                    # (InstantTensor, distributed ModelStreamer) and the engine's
                    # post-load barrier are collective across ranks, so a rank keeps
                    # its load only if every rank loaded. Every rank reports exactly
                    # once per strategy, also when its load raised, so no rank waits
                    # on a peer that never reports. Reporting before finalize() keeps
                    # a rank from registering or publishing weights its peers did not
                    # load; a finalize() that fails on one rank afterwards is not
                    # covered.
                    try:
                        result = strategy.load(result, ctx)
                    except StrategyRecoveryError:
                        _gather_load_outcomes(_FATAL, strategy, ctx)
                        raise
                    except Exception as e:
                        if _FATAL in _gather_load_outcomes(_FAILED, strategy, ctx):
                            raise StrategyRecoveryError(
                                f"[Worker {ctx.global_rank}] Another rank could not "
                                f"recover from strategy {strategy.name}"
                            ) from e
                        raise
                    outcomes = _gather_load_outcomes(_LOADED, strategy, ctx)
                    if _FATAL in outcomes:
                        raise StrategyRecoveryError(
                            f"[Worker {ctx.global_rank}] Another rank could not "
                            f"recover from strategy {strategy.name}"
                        )
                    failed = sum(1 for outcome in outcomes if outcome != _LOADED)
                    if failed:
                        raise StrategyFailed(
                            f"Strategy {strategy.name} succeeded here but failed on "
                            f"{failed} of {len(outcomes)} ranks",
                            mutated=True,
                        )
                publish_source_if_supported(result, ctx)
                span.set_attribute("weight_loading_strategy", strategy.name)
                return result.value
            except StrategyRecoveryError:
                strategy.rollback(ctx)
                raise
            except StrategyFailed as e:
                logger.warning(
                    f"[Worker {ctx.global_rank}] Strategy {strategy.name} failed, "
                    f"trying next: {e}"
                )
                strategy.rollback(ctx)
                # A reload writes into the engine's live model, which the next
                # strategy overwrites in full, and a rebuilt model would never
                # reach the engine.
                if e.mutated and not ctx.is_reload:
                    clear_exception_tracebacks(e)
                    result = LoadStrategyChain._reinit_for_retry(result, ctx, strategy)
                continue
            except Exception as e:
                logger.warning(
                    f"[Worker {ctx.global_rank}] Strategy {strategy.name} "
                    f"raised unexpected error, trying next: {e}"
                )
                strategy.rollback(ctx)

    raise RuntimeError(
        f"[Worker {ctx.global_rank}] No loading strategy succeeded "
        f"for model '{ctx.identity.model_name}'"
    )


def validate_load_strategy_chain() -> None:
    """Reject policies that require a separate engine integration."""
    from .. import envs

    if envs.MX_LOAD_STRATEGY_CHAIN != "INFERENCE":
        raise ValueError(
            f"Loading policy {envs.MX_LOAD_STRATEGY_CHAIN!r} requires its engine "
            "integration plugin; the base loader supports only 'INFERENCE'"
        )


def run_load_strategy_chain(model: nn.Module, ctx: LoadContext) -> nn.Module:
    """Execute the standalone inference loading policy."""
    validate_load_strategy_chain()
    return LoadStrategyChain.run(model, ctx)
