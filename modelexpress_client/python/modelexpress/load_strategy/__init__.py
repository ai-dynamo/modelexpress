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
from typing import Any

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
        The engine ranks agree after each strategy's prepare(), load() and
        finalize(), so they all succeed or all fall back together, no rank
        loads unless every rank can, and no rank publishes until every rank
        has finalized.

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
    recovery_failure: Exception | None = None,
):
    """Wrap one attempt in the strategy's prepare() and finalize().

    The chain only sequences the phases. What each one does, including how it
    differs on a reload, is the strategy's own decision. finalize() sits after
    the yield rather than in a finally, because a failed attempt has nothing
    to bring to a final layout; abort() runs instead, to undo whatever
    prepare() started.

    The engine ranks agree after prepare(), before any rank loads, and after
    finalize(), before any rank publishes; the caller's block agrees after
    load(). ``recovery_failure`` is this rank's error from recovering after the
    previous strategy, which the first agreement reports in place of prepare().

    The phases work on ``result`` in place: LoadResult is the stable envelope
    the chain holds for the whole attempt, which is why reinit_for_retry
    copies a replacement's state back into it rather than handing one out.
    """
    _prepare_all_ranks(strategy, result, ctx, recovery_failure)
    try:
        yield
    except BaseException:
        strategy.abort(result, ctx)
        raise
    _finalize_all_ranks(strategy, result, ctx)


_SUCCEEDED = "succeeded"
_FAILED = "failed"
_FATAL = "fatal"


def _gather_all_ranks(phase: str, value: Any, ctx: LoadContext) -> list[Any]:
    """Share this rank's ``value`` for ``phase`` and return every engine rank's.

    Every value carries the phase it is about, so ranks that reached different
    points of the chain fail loudly instead of comparing unrelated values.
    """
    state = (phase, value)
    states = ctx.adapter.all_gather_state(state) if ctx.adapter is not None else (state,)
    for rank, (peer_phase, _peer_value) in enumerate(states):
        if peer_phase != phase:
            raise StrategyRecoveryError(
                f"[Worker {ctx.global_rank}] Ranks are out of step in the loading "
                f"chain: this rank is at {phase!r}, rank {rank} at {peer_phase!r}"
            )
    return [peer_value for _peer_phase, peer_value in states]


def _eligible_on_every_rank(
    strategies: list[LoadStrategy], ctx: LoadContext
) -> list[LoadStrategy]:
    """Return the available strategies that every engine rank can run, in order.

    Availability can depend on the node (GDS hardware, NIXL, local files), but
    the ranks must walk the same strategies to agree on each outcome. A strategy
    that some rank cannot run is skipped on every rank. With nothing eligible on
    any rank, the chain reports that no strategy succeeded, as before.

    Every rank stops before loading if a rank's availability check raised, or if
    the ranks order the shared strategies differently (MX_LOAD_PLUGIN_BEFORE set
    differently per node), which would pair one rank's load with another's.
    """
    eligible: list[LoadStrategy] = []
    failure: Exception | None = None
    error: str | None = None
    # Hold an availability error until the other ranks have heard of it; raising
    # here would leave them waiting in the gather below. Check the remaining
    # strategies anyway: an availability check can gather across ranks itself.
    for strategy in strategies:
        try:
            if strategy.is_available(ctx):
                eligible.append(strategy)
        except Exception as e:
            if failure is None:
                failure, error = e, f"{strategy.name}: {e}"
    names = [strategy.name for strategy in eligible]
    per_rank = _gather_all_ranks("eligible", (names, error), ctx)

    if failure is not None:
        raise failure
    for rank, (_peer_names, peer_error) in enumerate(per_rank):
        if peer_error is not None:
            raise RuntimeError(
                f"[Worker {ctx.global_rank}] Rank {rank} could not check which "
                f"loading strategies it can run: {peer_error}"
            )
    names_per_rank = [peer_names for peer_names, _peer_error in per_rank]
    common = set(names).intersection(*names_per_rank)
    orders = {tuple(name for name in peer if name in common) for peer in names_per_rank}
    if len(orders) > 1:
        raise RuntimeError(
            f"[Worker {ctx.global_rank}] Ranks order the loading strategies "
            f"differently; eligible per rank: {names_per_rank}"
        )
    shared = [strategy for strategy in eligible if strategy.name in common]
    if not shared and any(names_per_rank):
        raise RuntimeError(
            f"[Worker {ctx.global_rank}] No loading strategy is eligible on every "
            f"rank for model '{ctx.identity.model_name}'; eligible per rank: "
            f"{names_per_rank}"
        )
    if len(shared) < len(eligible):
        logger.warning(
            f"[Worker {ctx.global_rank}] Skipping loaders that are not eligible on "
            f"every rank: {[name for name in names if name not in common]}"
        )
    return shared


def _prepare_all_ranks(
    strategy: LoadStrategy,
    result: LoadResult,
    ctx: LoadContext,
    recovery_failure: Exception | None,
) -> None:
    """Run ``strategy.prepare()`` and return only if every engine rank can load.

    load() can be collective (InstantTensor), so a rank that cannot load must say
    so before its peers start: they would otherwise wait for it inside load()
    while it waits for them in the next agreement. A rank that could not recover
    from the previous strategy skips prepare() and reports that instead; every
    rank then stops. If prepare() failed on some rank, every rank falls back
    together, and the ranks that prepared undo it with abort() first.
    """
    outcome = _SUCCEEDED
    failure: Exception | None = None
    if recovery_failure is not None:
        outcome, failure = _FATAL, recovery_failure
    else:
        try:
            strategy.prepare(result, ctx)
        except StrategyRecoveryError as e:
            outcome, failure = _FATAL, e
        except Exception as e:
            outcome, failure = _FAILED, e

    outcomes = _gather_all_ranks(f"prepare:{strategy.name}", outcome, ctx)
    if failure is None and all(peer_outcome == _SUCCEEDED for peer_outcome in outcomes):
        return
    if outcome == _SUCCEEDED:
        strategy.abort(result, ctx)
    if outcome == _FATAL or _FATAL in outcomes:
        if isinstance(failure, StrategyRecoveryError):
            raise failure
        # Raised as StrategyRecoveryError so that no rank, this one included,
        # treats it as a reason to try the next strategy.
        cause = f": {failure}" if outcome == _FATAL else ""
        raise StrategyRecoveryError(
            f"[Worker {ctx.global_rank}] A rank could not recover from a failed "
            f"strategy before {strategy.name}{cause}"
        ) from failure
    if failure is not None:
        raise failure
    raise StrategyFailed(
        f"Strategy {strategy.name} prepared here but failed to prepare on "
        f"{len(outcomes) - outcomes.count(_SUCCEEDED)} of {len(outcomes)} ranks",
        mutated=False,
    )


def _load_all_ranks(
    strategy: LoadStrategy, result: LoadResult, ctx: LoadContext
) -> LoadResult:
    """Run ``strategy.load()`` and keep its result only if every engine rank loaded.

    Later strategies (InstantTensor, distributed ModelStreamer) and the engine's
    post-load barrier are collective across ranks, so every rank must leave a
    strategy the same way. Each rank reports its outcome exactly once, also when
    its load raised, so no rank waits on a peer that never reports. Called before
    finalize(), so a rank never registers weights its peers did not load.

    Raises this rank's own load error; StrategyRecoveryError if any rank could not
    recover; or StrategyFailed(mutated=True) on a rank that loaded while another
    rank failed, so it falls back with the others.
    """
    outcome = _SUCCEEDED
    failure: Exception | None = None
    # Hold this rank's error until the others have heard its outcome; raising
    # here would leave them waiting in the gather below.
    try:
        result = strategy.load(result, ctx)
    except StrategyRecoveryError as e:
        outcome, failure = _FATAL, e
    except Exception as e:
        outcome, failure = _FAILED, e

    outcomes = _gather_all_ranks(f"load:{strategy.name}", outcome, ctx)

    # Every rank sees the same outcomes, so every rank picks the same branch: all
    # stop if any rank is fatal, all fall back if any rank failed, else all keep.
    if _FATAL in outcomes and outcome != _FATAL:
        raise StrategyRecoveryError(
            f"[Worker {ctx.global_rank}] Another rank could not recover from "
            f"strategy {strategy.name}"
        ) from failure
    if failure is not None:
        raise failure
    failed = sum(1 for peer_outcome in outcomes if peer_outcome != _SUCCEEDED)
    if failed:
        raise StrategyFailed(
            f"Strategy {strategy.name} succeeded here but failed on {failed} of "
            f"{len(outcomes)} ranks",
            mutated=True,
        )
    return result


def _finalize_all_ranks(
    strategy: LoadStrategy, result: LoadResult, ctx: LoadContext
) -> None:
    """Run ``strategy.finalize()`` and return only if every engine rank finalized.

    The chain publishes a rank as a source only after this returns, so no rank
    serves weights whose post-processing failed on a peer. If every rank failed,
    they fall back together, as a single rank would. If only some failed, every
    rank stops instead: a rank that finalized may have registered its tensors
    with NIXL, and a later strategy's register_tensors() keeps that catalog
    instead of registering the rebuilt model, so falling back could publish stale
    tensors.
    """
    outcome = _SUCCEEDED
    failure: Exception | None = None
    # As in _load_all_ranks, report before raising so no peer waits on this rank.
    try:
        strategy.finalize(result, ctx)
    except StrategyRecoveryError as e:
        outcome, failure = _FATAL, e
    except Exception as e:
        outcome, failure = _FAILED, e

    outcomes = _gather_all_ranks(f"finalize:{strategy.name}", outcome, ctx)
    if failure is not None and (
        outcome == _FATAL or all(peer_outcome == _FAILED for peer_outcome in outcomes)
    ):
        raise failure
    if any(peer_outcome != _SUCCEEDED for peer_outcome in outcomes):
        raise StrategyRecoveryError(
            f"[Worker {ctx.global_rank}] Strategy {strategy.name} finalized on only "
            f"{outcomes.count(_SUCCEEDED)} of {len(outcomes)} ranks; the ranks that "
            "finalized may have registered their weights, so no rank falls back"
        ) from failure


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

        # This rank's error from recovering after a failed strategy. Raising it
        # here would leave the other ranks waiting in the next strategy's
        # agreements, so the next strategy reports it to them instead.
        recovery_failure: Exception | None = None
        for strategy in eligible:
            logger.info(f"[Worker {ctx.global_rank}] Trying strategy: {strategy.name}")
            try:
                with _strategy_phases(strategy, result, ctx, recovery_failure):
                    result = _load_all_ranks(strategy, result, ctx)
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
                try:
                    strategy.rollback(ctx)
                    # A reload writes into the engine's live model, which the next
                    # strategy overwrites in full, and a rebuilt model would never
                    # reach the engine.
                    if e.mutated and not ctx.is_reload:
                        clear_exception_tracebacks(e)
                        result = LoadStrategyChain._reinit_for_retry(result, ctx, strategy)
                except Exception as recovery_error:
                    recovery_failure = recovery_error
                continue
            except Exception as e:
                logger.warning(
                    f"[Worker {ctx.global_rank}] Strategy {strategy.name} "
                    f"raised unexpected error, trying next: {e}"
                )
                try:
                    strategy.rollback(ctx)
                except Exception as recovery_error:
                    recovery_failure = recovery_error

    raise RuntimeError(
        f"[Worker {ctx.global_rank}] No loading strategy succeeded "
        f"for model '{ctx.identity.model_name}'"
    ) from recovery_failure


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
