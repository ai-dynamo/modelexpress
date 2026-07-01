# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side P2P source selection.

Ranks the compatible READY source workers returned by the metadata backend
before transfer attempts. The flow is:

    metadata backend -> ListSources(identity, READY) -> filter by worker_rank
      -> SourceSelector.order(candidates, ctx) -> slice to MAX_SOURCE_RETRIES
      -> NIXL/RDMA transfer

Selectors order candidates using the live ``LoadContext`` (the same object the
load strategies already pass around): only a few fields are read
(``worker_rank``, ``worker_id``, ``identity.model_name``), so there is no
parallel context type to keep in sync. The annotation is a forward reference to
avoid importing the load_strategy package at module load.

Phase 1 is stateless: a policy ranks candidates using only the current
candidate list, the target context, and the configured policy. It keeps no
global counters, leases, or server-side state. ``random`` preserves today's
behavior and stays the fallback whenever a non-random policy is misconfigured
or fails to construct.
"""

from __future__ import annotations

import hashlib
import logging
import random
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

from . import envs, p2p_pb2

if TYPE_CHECKING:
    from .load_strategy.context import LoadContext

logger = logging.getLogger("modelexpress.source_selection")

# Environment variable that chooses the active policy. Invalid values log a
# warning and fall back to DEFAULT_SELECTOR.
ENV_SELECTOR = "MX_P2P_SOURCE_SELECTOR"
DEFAULT_SELECTOR = "random"


@runtime_checkable
class SourceSelector(Protocol):
    """Orders compatible candidates into a per-target preference list."""

    name: str

    def order(
        self,
        candidates: list[p2p_pb2.SourceInstanceRef],
        ctx: LoadContext,
    ) -> list[p2p_pb2.SourceInstanceRef]: ...


class ScoredSelector:
    """Base for stateless policies that assign each candidate a score.

    Subclasses implement ``score``; the base orders by descending score so the
    top candidate is the winner and the full order is the preference list. Pure
    and deterministic, which keeps policies tiny and unit-testable.
    """

    name = "scored"

    def score(
        self,
        candidate: p2p_pb2.SourceInstanceRef,
        ctx: LoadContext,
    ) -> float:
        raise NotImplementedError

    def order(
        self,
        candidates: list[p2p_pb2.SourceInstanceRef],
        ctx: LoadContext,
    ) -> list[p2p_pb2.SourceInstanceRef]:
        return sorted(
            candidates,
            key=lambda c: self.score(c, ctx),
            reverse=True,
        )


class RandomSelector:
    """Behavior-preserving default: shuffle candidates with a local RNG.

    Uses a local ``random.Random`` rather than process-global state so it does
    not perturb other callers. Matches the previous ``random.shuffle`` behavior.
    """

    name = "random"

    def order(
        self,
        candidates: list[p2p_pb2.SourceInstanceRef],
        ctx: LoadContext,
    ) -> list[p2p_pb2.SourceInstanceRef]:
        out = list(candidates)
        random.Random().shuffle(out)
        return out


# blake2b(digest_size=8) yields an 8-byte digest, so the score space is 2**64.
# _unit_hash maps that score into [0, 1) for policies (load_aware) that blend the
# rendezvous term with a normalized load penalty on the same scale.
_HASH_SPACE = 2**64


def _rendezvous_score(
    candidate: p2p_pb2.SourceInstanceRef,
    ctx: LoadContext,
) -> int:
    """Stable, process-independent HRW score for (target, candidate).

    Uses blake2b (not Python's salted ``hash()``) so orderings are identical
    across processes and restarts. Shared by ``rendezvous_hash`` and the
    rendezvous base term of ``load_aware``.
    """
    key = "|".join(
        str(x)
        for x in (
            ctx.identity.model_name,
            ctx.worker_id,
            ctx.worker_rank,
            candidate.mx_source_id,
            candidate.worker_id,
            candidate.worker_rank,
        )
    )
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _unit_hash(
    candidate: p2p_pb2.SourceInstanceRef,
    ctx: LoadContext,
) -> float:
    """Rendezvous score mapped into ``[0, 1)``."""
    return _rendezvous_score(candidate, ctx) / _HASH_SPACE


class RendezvousHashSelector(ScoredSelector):
    """Stateless deterministic spreading via rendezvous (HRW) hashing.

    Each candidate's score is a stable hash of the target identity and the
    candidate identity. Ordering by descending score gives each target a
    deterministic preference list; because the target identity is part of the
    key, different targets get different first choices, spreading first-choice
    sources across independently starting targets without a shared counter,
    server coordination, or new metadata fields.

    Stable across process restarts (uses blake2b, not Python's salted
    ``hash()``) and across small source-set changes: adding or removing one
    source perturbs only a fraction of rankings. The distribution is
    probabilistic, not guaranteed -- with more targets than sources, collisions
    still occur.
    """

    name = "rendezvous_hash"

    def score(
        self,
        candidate: p2p_pb2.SourceInstanceRef,
        ctx: LoadContext,
    ) -> float:
        return _rendezvous_score(candidate, ctx)


class LoadAwareSelector:
    """Load-aware spreading: rendezvous base biased away from busy sources.

    Score is ``unit_hash(target, candidate) - w_load * normalized_load``, so the
    stable rendezvous ordering (Section 1's deterministic spread) is preserved
    while a per-source load penalty steers new targets away from sources that
    are already serving many transfers -- the single-source convergence a purely
    deterministic hash cannot avoid when a source is persistently available.

    Load comes from ``candidate.active_transfers`` (a server-estimated,
    TTL-decayed count surfaced in ListSources). It is normalized by the maximum
    across the current candidate set, so the penalty is relative and needs no
    tuned capacity constant: when loads are equal (or all zero -- older servers,
    disabled tracking) every penalty is identical and ordering collapses exactly
    to ``rendezvous_hash``. This policy overrides ``order()`` rather than
    implementing ``ScoredSelector.score()`` because that normalization requires
    the whole candidate set, not one candidate at a time.
    """

    name = "load_aware"

    def __init__(self, w_load: float | None = None) -> None:
        # Weight trading rendezvous spread (base in [0, 1)) against the load
        # penalty (also in [0, 1] after normalization). Default from env.
        self.w_load = envs.MX_P2P_LOAD_WEIGHT if w_load is None else w_load

    @staticmethod
    def _load(candidate: p2p_pb2.SourceInstanceRef) -> float:
        # active_transfers is optional on the wire; older servers or disabled
        # load tracking omit it -> treat as 0 (no penalty).
        return float(max(0, getattr(candidate, "active_transfers", 0) or 0))

    def order(
        self,
        candidates: list[p2p_pb2.SourceInstanceRef],
        ctx: LoadContext,
    ) -> list[p2p_pb2.SourceInstanceRef]:
        if not candidates:
            return []
        max_load = max(self._load(c) for c in candidates)
        if max_load <= 0.0:
            # No observable load: identical to rendezvous_hash.
            return sorted(
                candidates, key=lambda c: _rendezvous_score(c, ctx), reverse=True
            )
        return sorted(
            candidates,
            key=lambda c: _unit_hash(c, ctx) - self.w_load * (self._load(c) / max_load),
            reverse=True,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# A policy registers a factory by name so adding one does not touch the RDMA
# strategy. Factories are context-free: the per-target signal flows in through
# the LoadContext passed to order(), not through construction.
SelectorFactory = Callable[[], SourceSelector]

SELECTORS: dict[str, SelectorFactory] = {}


def register_selector(name: str, factory: SelectorFactory) -> None:
    """Register a selector factory under ``name`` (last registration wins)."""
    SELECTORS[name] = factory


register_selector("random", lambda: RandomSelector())
register_selector("rendezvous_hash", lambda: RendezvousHashSelector())
register_selector("load_aware", lambda: LoadAwareSelector())


def get_selector(name: str) -> SourceSelector:
    """Resolve a policy name to a selector instance.

    Unknown names, or factories that raise, log a warning and fall back to
    ``random`` so a misconfiguration never blocks loading.
    """
    factory = SELECTORS.get(name)
    if factory is None:
        logger.warning(
            "Unknown P2P source selector %r, falling back to %r. Known: %s",
            name,
            DEFAULT_SELECTOR,
            sorted(SELECTORS),
        )
        return SELECTORS[DEFAULT_SELECTOR]()
    try:
        return factory()
    except Exception as e:  # defensive: a broken factory must not block loading
        logger.warning(
            "Failed to construct P2P source selector %r (%s), falling back to %r",
            name,
            e,
            DEFAULT_SELECTOR,
        )
        return SELECTORS[DEFAULT_SELECTOR]()


def get_configured_selector() -> SourceSelector:
    """Resolve the selector named by ``MX_P2P_SOURCE_SELECTOR`` (default random)."""
    return get_selector(envs.MX_P2P_SOURCE_SELECTOR or DEFAULT_SELECTOR)


def configured_policy_label() -> str:
    """Resolved policy name for metric/label use, matching get_selector.

    Resolves through the same registry path selection uses, including the
    fallback to ``random`` when the configured policy is unknown *or* its factory
    raises -- so emitted labels never claim a policy that did not actually run.
    """
    return get_configured_selector().name
