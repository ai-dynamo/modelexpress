# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expert-parallel layout helpers for MoE refit.

When a framework adapter wants to publish or request per-expert MoE
tensors through the substrate (:class:`SliceOwnership.owned_expert_ids`,
:class:`SliceRequest.required_experts`), it needs to translate
``(ep_rank, ep_world_size, num_experts, placement_strategy)`` into the
expert-id set this rank owns. The math is small but boilerplate, and
the three placement strategies vLLM, SGLang, and Megatron-Core ship
today are stable enough to centralize once.

This module is intentionally pure-Python and framework-agnostic — no
torch, no NIXL, no engine knowledge. Framework adapters import it,
compute their expert ids, and pass the result to the substrate.

See the design doc at ``temp/MX_EP_Support_PR_Design.md`` for the
full context and the verl rank-to-rank prototype's existing
``RankLocalPublisher`` API for how the per-expert pattern composes
with the rest of the rank-local publishing flow.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

ExpertPlacement = Literal["linear", "round_robin", "external"]
"""How experts are partitioned across an EP world.

- ``linear`` — contiguous blocks. EP rank ``r`` owns experts
  ``[r * (E / W), (r + 1) * (E / W))``. Standard vLLM and SGLang default.
- ``round_robin`` — interleaved. EP rank ``r`` owns experts
  ``{r, r + W, r + 2W, ...}``. Less common; supported by Megatron-Core
  when expert-balanced rebalancing is enabled.
- ``external`` — explicit map supplied by the caller. Used for EPLB
  (expert-parallel load balancer) deployments where the framework
  picks the placement at runtime based on observed expert usage.
"""


def compute_local_expert_ids(
    ep_rank: int,
    ep_world_size: int,
    num_experts: int,
    placement: ExpertPlacement = "linear",
    external_map: dict[int, list[int]] | None = None,
) -> tuple[int, ...]:
    """Return the expert-ids that ``ep_rank`` owns under ``placement``.

    The return is an immutable tuple, sorted ascending — suitable for
    passing directly into :class:`SliceOwnership.owned_expert_ids`
    (which is typed as ``tuple[int, ...]``) or wrapping in a
    ``frozenset`` for :class:`SliceRequest.required_experts`.

    Args:
        ep_rank: this worker's EP rank, in ``[0, ep_world_size)``.
        ep_world_size: total number of EP ranks.
        num_experts: total experts in the model.
        placement: one of ``"linear"`` (default), ``"round_robin"``,
            or ``"external"``.
        external_map: when ``placement="external"``, the explicit
            ``{ep_rank: [expert_id, ...]}`` map. Required for
            ``"external"``, ignored otherwise. The mapping does not
            need to be a partition — overlapping ownership across
            ranks is permitted (e.g. for replicated-expert deployments).

    Returns:
        Sorted tuple of expert ids this rank owns. Empty tuple if
        ``num_experts == 0``.

    Raises:
        ValueError: if ``ep_rank`` is out of range, if
            ``placement="linear"`` and ``num_experts`` isn't divisible
            by ``ep_world_size``, or if ``placement="external"`` and
            ``external_map`` is missing or doesn't include ``ep_rank``.

    Examples:
        >>> compute_local_expert_ids(0, 2, 8, "linear")
        (0, 1, 2, 3)
        >>> compute_local_expert_ids(1, 2, 8, "linear")
        (4, 5, 6, 7)
        >>> compute_local_expert_ids(0, 2, 8, "round_robin")
        (0, 2, 4, 6)
        >>> compute_local_expert_ids(1, 2, 8, "round_robin")
        (1, 3, 5, 7)
        >>> compute_local_expert_ids(
        ...     0, 2, 8, "external",
        ...     external_map={0: [0, 1, 4, 7], 1: [2, 3, 5, 6]},
        ... )
        (0, 1, 4, 7)
    """
    if ep_world_size < 1:
        raise ValueError(
            f"compute_local_expert_ids: ep_world_size must be >= 1, got {ep_world_size}"
        )
    if ep_rank < 0 or ep_rank >= ep_world_size:
        raise ValueError(
            f"compute_local_expert_ids: ep_rank={ep_rank} out of range "
            f"[0, {ep_world_size})"
        )
    if num_experts < 0:
        raise ValueError(
            f"compute_local_expert_ids: num_experts must be >= 0, got {num_experts}"
        )

    if num_experts == 0:
        return ()

    if placement == "linear":
        if num_experts % ep_world_size != 0:
            raise ValueError(
                f"compute_local_expert_ids: 'linear' placement requires "
                f"num_experts ({num_experts}) divisible by ep_world_size "
                f"({ep_world_size}). For uneven splits, use 'external' "
                f"with an explicit map."
            )
        per_rank = num_experts // ep_world_size
        start = ep_rank * per_rank
        return tuple(range(start, start + per_rank))

    if placement == "round_robin":
        return tuple(range(ep_rank, num_experts, ep_world_size))

    if placement == "external":
        if external_map is None:
            raise ValueError(
                "compute_local_expert_ids: 'external' placement requires "
                "external_map to be provided"
            )
        if ep_rank not in external_map:
            raise ValueError(
                f"compute_local_expert_ids: 'external' placement, but "
                f"external_map has no entry for ep_rank={ep_rank}"
            )
        owned = external_map[ep_rank]
        if not all(0 <= e < num_experts for e in owned):
            raise ValueError(
                f"compute_local_expert_ids: external_map[{ep_rank}]={owned} "
                f"contains expert ids outside [0, {num_experts})"
            )
        return tuple(sorted(owned))

    raise ValueError(
        f"compute_local_expert_ids: unknown placement {placement!r}; "
        f"expected one of 'linear', 'round_robin', 'external'"
    )


def expert_ids_to_contiguous_ranges(
    expert_ids: Iterable[int],
) -> tuple[tuple[int, int], ...]:
    """Compress a set of expert ids into contiguous ``[lo, hi)`` runs.

    This is the **ranges adapter**: it translates the framework concept
    "expert ids" into plain tensor ranges on the expert axis, which the
    core planner already understands as ordinary ``SHARD`` intersections —
    no expert-specific logic needed in the planner. A caller that owns (or
    wants) experts turns its id set into these ranges, emits one
    ``SliceOwnership`` / ``SliceRequest`` per run on the expert axis, and
    the range machinery does the rest.

    Contiguous (linear EP) ownership collapses to a single run; interleaved
    (round-robin / EPLB) ownership yields multiple runs — the "non-contiguous
    = multiple contiguous entries" pattern the resharding contract endorses.

    Args:
        expert_ids: any iterable of non-negative expert ids (dupes allowed;
            order irrelevant).

    Returns:
        Sorted tuple of ``(lo, hi)`` half-open runs, ``lo`` inclusive,
        ``hi`` exclusive. Empty tuple for an empty input.

    Examples:
        >>> expert_ids_to_contiguous_ranges([0, 1, 2, 3])
        ((0, 4),)
        >>> expert_ids_to_contiguous_ranges([0, 2, 4, 6])
        ((0, 1), (2, 3), (4, 5), (6, 7))
        >>> expert_ids_to_contiguous_ranges([4, 5, 6, 9, 10])
        ((4, 7), (9, 11))
    """
    ids = sorted(set(int(e) for e in expert_ids))
    if not ids:
        return ()
    runs: list[tuple[int, int]] = []
    lo = prev = ids[0]
    for e in ids[1:]:
        if e == prev + 1:
            prev = e
            continue
        runs.append((lo, prev + 1))
        lo = prev = e
    runs.append((lo, prev + 1))
    return tuple(runs)


def validate_placement_partition(
    ep_world_size: int,
    num_experts: int,
    placement: ExpertPlacement = "linear",
    external_map: dict[int, list[int]] | None = None,
) -> None:
    """Validate that a placement covers every expert at least once.

    For trainer-side publishing this is usually required — every expert
    must be owned by at least one publisher rank, or the receiver can't
    pull it. For receiver-side requests this is not required: a receiver
    asking only for its local experts is exactly the EP fan-out
    semantics we want.

    Args:
        ep_world_size: number of EP ranks.
        num_experts: total experts.
        placement: same as :func:`compute_local_expert_ids`.
        external_map: same as :func:`compute_local_expert_ids`.

    Raises:
        ValueError: if any expert in ``[0, num_experts)`` is not owned
            by any rank under this placement.
    """
    seen: set[int] = set()
    for r in range(ep_world_size):
        seen.update(compute_local_expert_ids(r, ep_world_size, num_experts, placement, external_map))
    missing = set(range(num_experts)) - seen
    if missing:
        raise ValueError(
            f"validate_placement_partition: placement {placement!r} on "
            f"{ep_world_size} ranks does not cover experts {sorted(missing)} "
            f"(of {num_experts} total). Trainer-side publishing requires "
            f"every expert to be owned by at least one rank."
        )
