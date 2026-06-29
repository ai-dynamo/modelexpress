# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side P2P source selection.

Ranks the compatible READY source workers returned by the metadata backend
before transfer attempts. The flow is:

    metadata backend -> ListSources(identity, READY) -> filter by worker_rank
      -> SourceSelector.order(candidates, context) -> slice to MAX_SOURCE_RETRIES
      -> NIXL/RDMA transfer

Phase 1 is stateless: a policy ranks candidates using only the current
candidate list, the target context, and the configured policy. It keeps no
global counters, leases, or server-side state. ``random`` preserves today's
behavior and stays the fallback whenever a non-random policy is misconfigured
or fails to construct.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from . import p2p_pb2

logger = logging.getLogger("modelexpress.source_selection")

# Environment variable that chooses the active policy. Invalid values log a
# warning and fall back to DEFAULT_SELECTOR.
ENV_SELECTOR = "MX_P2P_SOURCE_SELECTOR"
DEFAULT_SELECTOR = "random"


@dataclass(frozen=True)
class SourceSelectionContext:
    """Client-side target context passed to a selector.

    Holds only values already available on the client when source discovery
    runs; it introduces no new metadata or schema dependency.
    """

    worker_rank: int
    global_rank: int
    worker_id: str
    model_name: str
    # Optional seed for the random policy. None matches today's behavior
    # (process-global entropy); a fixed int makes ordering reproducible.
    selector_seed: int | None = None


@runtime_checkable
class SourceSelector(Protocol):
    """Orders compatible candidates into a per-target preference list."""

    name: str

    def order(
        self,
        candidates: list[p2p_pb2.SourceInstanceRef],
        context: SourceSelectionContext,
    ) -> list[p2p_pb2.SourceInstanceRef]:
        ...


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
        context: SourceSelectionContext,
    ) -> float:
        raise NotImplementedError

    def order(
        self,
        candidates: list[p2p_pb2.SourceInstanceRef],
        context: SourceSelectionContext,
    ) -> list[p2p_pb2.SourceInstanceRef]:
        return sorted(
            candidates,
            key=lambda c: self.score(c, context),
            reverse=True,
        )


class RandomSelector:
    """Behavior-preserving default: shuffle candidates with a local RNG.

    Uses a local ``random.Random`` rather than process-global state so it does
    not perturb other callers and stays reproducible when seeded. With
    ``selector_seed=None`` it matches the previous ``random.shuffle`` behavior.
    """

    name = "random"

    def order(
        self,
        candidates: list[p2p_pb2.SourceInstanceRef],
        context: SourceSelectionContext,
    ) -> list[p2p_pb2.SourceInstanceRef]:
        out = list(candidates)
        rng = random.Random(context.selector_seed)
        rng.shuffle(out)
        return out


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
        context: SourceSelectionContext,
    ) -> float:
        key = "|".join(
            str(x)
            for x in (
                context.model_name,
                context.worker_id,
                context.worker_rank,
                candidate.mx_source_id,
                candidate.worker_id,
                candidate.worker_rank,
            )
        )
        digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# A policy registers a factory by name so adding one does not touch the RDMA
# strategy. The factory receives the selection context (future policies may use
# it for construction; the Phase 1 policies are context-free).
SelectorFactory = Callable[[SourceSelectionContext], SourceSelector]

SELECTORS: dict[str, SelectorFactory] = {}


def register_selector(name: str, factory: SelectorFactory) -> None:
    """Register a selector factory under ``name`` (last registration wins)."""
    SELECTORS[name] = factory


register_selector("random", lambda _ctx: RandomSelector())
register_selector("rendezvous_hash", lambda _ctx: RendezvousHashSelector())


def get_selector(name: str, context: SourceSelectionContext) -> SourceSelector:
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
        return SELECTORS[DEFAULT_SELECTOR](context)
    try:
        return factory(context)
    except Exception as e:  # defensive: a broken factory must not block loading
        logger.warning(
            "Failed to construct P2P source selector %r (%s), falling back to %r",
            name,
            e,
            DEFAULT_SELECTOR,
        )
        return SELECTORS[DEFAULT_SELECTOR](context)


def get_configured_selector(context: SourceSelectionContext) -> SourceSelector:
    """Resolve the selector named by ``MX_P2P_SOURCE_SELECTOR`` (default random)."""
    return get_selector(os.environ.get(ENV_SELECTOR, DEFAULT_SELECTOR), context)


def configured_policy_label() -> str:
    """Resolved policy name for metric/label use, matching get_selector.

    Resolves through the same registry path selection uses, including the
    fallback to ``random`` when the configured policy is unknown *or* its factory
    raises -- so emitted labels never claim a policy that did not actually run.
    """
    name = os.environ.get(ENV_SELECTOR, DEFAULT_SELECTOR)
    factory = SELECTORS.get(name)
    if factory is None:
        return DEFAULT_SELECTOR
    try:
        return factory(SourceSelectionContext(0, 0, "", "")).name
    except Exception:
        return DEFAULT_SELECTOR
