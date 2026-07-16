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


class TopologyAwareSelector(ScoredSelector):
    """Locality-first spreading: prefer sources in the narrowest shared RDMA domain.

    ``score = (shared_depth, tiebreak)``. ``shared_depth`` is the index of the
    narrowest topology level (broad -> narrow) at which the target and the
    candidate share a domain value; ``order()`` sorts by descending score, so the
    closest source wins and the tiebreak only decides among equidistant peers.
    Because MX P2P is NIXL over RDMA between replicas, the relevant locality is
    the RDMA fabric -- same rail (one leaf hop) -> same rack -> same block ->
    cross-rack. The level hierarchy is not hard-coded: it comes from
    ``MX_P2P_TOPOLOGY_LEVELS`` (the cluster's Dynamo ``ClusterTopology``
    ``spec.levels``), and the per-source domain values from each candidate's
    published ``topology`` metadata -- so MX consumes existing Dynamo/Grove node
    labels rather than inventing its own.

    The ``tiebreak`` is the rendezvous jitter, so peers in the same tier still
    spread instead of piling onto the one nearest source. When no levels are
    configured, this node has no topology, or nothing is shared, ``shared_depth``
    is constant and ordering collapses exactly to that jitter (rendezvous order)
    -- never worse than the deterministic baseline. NVLink is intentionally not
    modeled: co-located source/target replicas let NIXL auto-select the NVLink
    backend, so the selector need not.

    Composes with load-aware selection without depending on it: when
    ``MX_P2P_TOPOLOGY_LOAD_WEIGHT > 0`` the within-tier tiebreak becomes
    ``unit_hash - w * source_load`` (``source_load`` read defensively off the
    candidate), so within a locality tier it also steers away from busy sources.
    With the weight at 0 (default) the tiebreak is the pure jitter and the
    selector needs nothing from the load-aware feature.
    """

    name = "topology_aware"

    def __init__(self) -> None:
        from . import envs
        from .topology import local_topology, resolve_levels

        self._levels = resolve_levels()
        self._local = local_topology()
        self.w_load = envs.MX_P2P_TOPOLOGY_LOAD_WEIGHT

    def _unit_hash(
        self,
        candidate: p2p_pb2.SourceInstanceRef,
        ctx: LoadContext,
    ) -> float:
        """Rendezvous digest for (target, candidate), mapped into ``[0, 1)``."""
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
        return int.from_bytes(digest, "big") / 2**64

    def _shared_depth(self, source_topology: dict[str, str]) -> int:
        """Index of the narrowest level where target and source share a value.

        Later level = narrower = closer; -1 when nothing is shared.
        """
        depth = -1
        for i, level in enumerate(self._levels):
            local_value = self._local.get(level)
            if local_value is not None and local_value == source_topology.get(level):
                depth = i
        return depth

    def score(  # type: ignore[override]
        self,
        candidate: p2p_pb2.SourceInstanceRef,
        ctx: LoadContext,
    ) -> tuple[int, float]:
        # Read defensively so old servers / non-topology candidates degrade to
        # rendezvous ordering rather than raising.
        raw = getattr(candidate, "topology", None)
        source_topology = dict(raw) if raw else {}
        depth = self._shared_depth(source_topology)
        tiebreak = self._unit_hash(candidate, ctx)
        if self.w_load > 0.0:
            load = min(1.0, max(0.0, getattr(candidate, "source_load", 0.0)))
            tiebreak -= self.w_load * load
        return (depth, tiebreak)


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
register_selector("topology_aware", lambda: TopologyAwareSelector())


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
