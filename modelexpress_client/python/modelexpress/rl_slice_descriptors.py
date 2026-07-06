# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Slice ownership + slice request descriptors for cross-parallelism resharding.

This module defines the **data model** for the rank-to-rank no-allgather
refit contract used across ModelExpress RL framework integrations. It is
deliberately framework-agnostic: PrimeRL, NemoRL, and verl all build the
same :class:`SliceOwnership` / :class:`SliceRequest` shapes from their own
trainer-side parallelism state and feed them to the planner.

The contract has three parties:

- **Source** (trainer rank or inference replica) publishes ``SliceOwnership``:
  "I hold these tensor ranges, in this dtype, at this NIXL/CUDA address."
- **Receiver** (inference rank) declares ``SliceRequest``: "I need this
  tensor range, in this dtype, landing at this destination offset."
- **Planner** (pure function, no IO, no tensor bytes) intersects every
  request against every published ownership and emits ``SegmentPlan`` —
  a list of (source, source-range, target-range) tuples the receiver can
  drive through NIXL.

The load-bearing property: **no global allgather**. Each trainer rank
publishes only its own local shard (via ``DTensor.to_local()`` for FSDP /
TP, or native Megatron-shard math for Megatron-Core). The receiver
discovers the covering set of sources and pulls just what it needs.

These descriptors are the wire-shape carried in ``TensorDescriptorV2``
metadata; the planner side is in :mod:`modelexpress.rl_reshard_planner`,
and the publisher side is in :mod:`modelexpress.rank_local_publisher`.

The ranges-only contract (framework concepts live in adapters, not the core)
-----------------------------------------------------------------------------
The core planner speaks **only tensor ranges**: ``(axis, [lo, hi))``
intersections. Framework concepts (MoE expert ids, DTensor placements,
Megatron roles) are translated *into* ranges by per-framework adapters
before they reach the planner — the planner never understands "expert",
"DTensor", or "Megatron role". How each sharding shape maps:

- **Single-dim uniform** ``{dim, rank, count}`` → a range; ``lo = rank*len``.
  (Don't add a rank/count form — ranges are a superset.)
- **Single-dim contiguous non-uniform** ``{dim, start, length}`` → exactly
  ``local_shard_range = (start, start+length)``.
- **Single-dim non-contiguous** (round-robin / EPLB) → **multiple contiguous
  entries** for the same tensor name; the planner unions them. Use
  :func:`modelexpress.rl_expert_layout.expert_ids_to_contiguous_ranges` to
  turn an id set into these runs.
- **Multi-dim (2D-mesh) sharding** → not built yet; the ready extension is a
  ``list[(axis, range)]`` intersected per axis (product of per-axis
  intersections). Build when a 2D-sharded workload appears; keep the
  single-axis fast path.
- **Striding** → already expressible as multiple entries; an optional compact
  ``(start, stride, count)`` form is a follow-on only if entry count becomes a
  measured cost.

The one deliberate non-range field on the descriptor is ``compile_target`` /
``quantization_scope`` — these are a **compatibility gate** (reject
incompatible kernel layouts before pulling), not a sharding concept.

MoE experts are NOT a core concept: an expert shard is just a ``SHARD`` range
on the expert axis. The ``is_expert`` / ``owned_expert_ids`` /
``required_experts`` fields are a **deprecated compatibility shim** kept only
until every caller emits expert ranges via the adapter; new code should not
use them (see the field docs and the planner shim note).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

PlacementKind = Literal["REPLICATE", "SHARD", "PARTIAL"]
"""Placement of a tensor across the source's parallelism mesh.

- ``REPLICATE`` — every source holds the same tensor (e.g. layer norms,
  small bias vectors). Multiple sources advertising the same name are
  interchangeable.
- ``SHARD`` — the tensor is sliced along one or more axes; each source
  holds a contiguous sub-range. The planner uses ``local_shard_range``
  to compute coverage.
- ``PARTIAL`` — each source holds a partial-sum that needs an all-reduce
  to produce the canonical value. Reserved for future use; the current
  planner rejects ``PARTIAL`` placements with a clear error rather than
  silently producing wrong tensors.
"""

QuantizationScope = Literal["absent", "local", "global-required", "generated-on-target"]
"""How quantization metadata interacts with resharding.

- ``absent`` — no quantization (bf16/fp32 weights).
- ``local`` — per-channel or per-block scales that travel with each shard
  and stay locally interpretable (e.g. cutlass per-output-channel FP8).
- ``global-required`` — a global tensor (e.g. ``weight_scale_inv``) that
  every receiver needs in full. The planner refuses zero-copy resharding
  on these and raises :class:`QuantizationMetadataError` so callers can
  fall back to a full-copy install path for the affected tensors.
- ``generated-on-target`` — the receiver computes the scale locally
  (e.g. amax-based blockwise quant from bf16 source). No metadata
  travels; the planner treats the source as bf16.
"""


@dataclass(frozen=True)
class SliceOwnership:
    """One contiguous slice of a tensor advertised by a publisher.

    Multiple ``SliceOwnership`` entries can describe the same tensor name
    if it's sharded across multiple publisher ranks — the planner walks
    every published ownership for a given (model_name, tensor_name) and
    matches against receiver requests.

    Args:
        model_name: identifier shared between trainer and inference (e.g.
            ``"Qwen/Qwen3-30B-A3B-Instruct-2507"``). Receivers filter the
            catalog by this exact string before any slice math runs.
        tensor_name: HF-canonical name (e.g.
            ``"model.layers.0.self_attn.q_proj.weight"``). For MoE the
            convention is the stacked-expert form (``experts.w13_weight``);
            per-expert splits happen in the receiver-side translator.
        global_shape: full un-sharded shape of the tensor (across all
            source ranks). Receivers use this to validate their requests
            don't exceed the tensor's bounds.
        dtype: torch dtype string (e.g. ``"torch.bfloat16"``,
            ``"torch.float8_e4m3fn"``). The planner refuses to plan a
            transfer if request dtype != ownership dtype — the caller
            must convert before publishing or after receiving.
        placement_kind: see :data:`PlacementKind`.
        shard_axis: axis along which the tensor is sharded. ``None`` for
            ``REPLICATE``; required for ``SHARD``.
        local_shard_range: (start, end) along ``shard_axis`` that this
            source holds. End is exclusive. Required for ``SHARD``; the
            planner uses this to intersect against receiver requests.
        worker_rank: publisher's rank index. Receivers can filter by
            same-rank for routing on multi-NIC GB200 fabrics (see
            :class:`MxV2RefitReceiver` ``same_rank_only`` in NemoRL v2).
        nixl_addr: GPU memory address of this shard, as registered with
            NIXL. Receivers use this + ``byte_size`` to issue RDMA reads
            directly into their target buffer.
        byte_size: size in bytes of this shard. Derived from
            ``local_shape × dtype.element_size()`` at publish time.
        device_id: CUDA device index for the shard.
        is_expert: **DEPRECATED shim.** Experts are not a core concept —
            an expert shard is a plain ``SHARD`` range on the expert axis
            (``shard_axis`` = the expert axis, ``local_shard_range`` = the
            owned-expert range). Prefer publishing that range directly; for
            non-contiguous ownership emit one entry per run from
            :func:`modelexpress.rl_expert_layout.expert_ids_to_contiguous_ranges`.
            Kept only for back-compat until all callers migrate.
        expert_axis: **DEPRECATED shim** — use ``shard_axis`` (the expert
            axis is just the shard axis for an expert tensor).
        owned_expert_ids: **DEPRECATED shim** — encode ownership as the
            ``local_shard_range`` (contiguous) or multiple range entries
            (non-contiguous). Retained for the transitional expert filter.
        compile_target: kernel layout tag (e.g. ``"bf16_cast"``,
            ``"cutlass_fp8"``). The planner refuses to plan if request
            ``compile_target_filter`` doesn't include this — the same
            safety net as Phase 3b on the prime-rl side.
        compile_metadata: kernel-specific parameters (e.g.
            ``{"block_size": 128, "scale_layout": "per_channel"}``).
            Free-form dict; receivers compare via subset-equality.
        quantization_scope: see :data:`QuantizationScope`. The planner
            refuses zero-copy on ``"global-required"``.
    """

    model_name: str
    tensor_name: str
    global_shape: tuple[int, ...]
    dtype: str
    placement_kind: PlacementKind = "REPLICATE"

    shard_axis: int | None = None
    local_shard_range: tuple[int, int] | None = None

    worker_rank: int = 0
    nixl_addr: int = 0
    byte_size: int = 0
    device_id: int = 0

    is_expert: bool = False
    expert_axis: int = 0
    owned_expert_ids: tuple[int, ...] = field(default_factory=tuple)

    compile_target: str = "bf16_cast"
    compile_metadata: dict[str, object] = field(default_factory=dict)
    quantization_scope: QuantizationScope = "absent"

    def __post_init__(self) -> None:
        if self.placement_kind == "SHARD":
            if self.shard_axis is None:
                raise ValueError(
                    f"SliceOwnership({self.tensor_name!r}): SHARD requires shard_axis"
                )
            if self.local_shard_range is None:
                raise ValueError(
                    f"SliceOwnership({self.tensor_name!r}): SHARD requires local_shard_range"
                )
            lo, hi = self.local_shard_range
            if lo < 0 or hi <= lo:
                raise ValueError(
                    f"SliceOwnership({self.tensor_name!r}): local_shard_range "
                    f"({lo}, {hi}) is empty or negative"
                )
            if hi > self.global_shape[self.shard_axis]:
                raise ValueError(
                    f"SliceOwnership({self.tensor_name!r}): local_shard_range "
                    f"hi={hi} exceeds global_shape[{self.shard_axis}]="
                    f"{self.global_shape[self.shard_axis]}"
                )
        if self.placement_kind == "PARTIAL":
            raise NotImplementedError(
                f"SliceOwnership({self.tensor_name!r}): PARTIAL placement is "
                "not yet supported (would require an all-reduce on the "
                "receiver side, which contradicts the no-allgather contract)"
            )

    def covers(self, request_range: tuple[int, int]) -> tuple[int, int] | None:
        """Return the intersection of this ownership with ``request_range``,
        or ``None`` if there's no overlap.

        Both ranges are along the same axis; both use end-exclusive form.
        For ``REPLICATE`` ownership the whole tensor is "covered" so
        ``request_range`` is returned unchanged.
        """
        if self.placement_kind == "REPLICATE":
            return request_range
        assert self.local_shard_range is not None  # __post_init__ checked
        own_lo, own_hi = self.local_shard_range
        req_lo, req_hi = request_range
        lo = max(own_lo, req_lo)
        hi = min(own_hi, req_hi)
        if lo >= hi:
            return None
        return (lo, hi)


@dataclass(frozen=True)
class SliceRequest:
    """One contiguous slice of a tensor a receiver wants to land locally.

    A receiver typically emits one ``SliceRequest`` per tensor per
    receiver-local layout slot. For a sharded receiver, each rank emits
    its own request for its own portion. For a replicated receiver,
    every rank emits a request for the full tensor.

    Args:
        tensor_name: must match a ``SliceOwnership.tensor_name`` (HF name).
        global_range: (start, end) along the receiver's shard axis that
            this rank wants. End is exclusive. For full-tensor receives,
            use ``(0, global_shape[shard_axis])``.
        shard_axis: which axis the receiver shards along. ``None`` for
            full-tensor receive.
        dtype: receiver's expected dtype. Must match the source's dtype
            or the planner errors.
        receiver_rank: caller's rank index; carried into the plan so
            same-rank routing policies can apply.
        target_addr: GPU memory address where the received bytes should
            land. The planner uses this + ``target_offset`` + ``byte_count``
            to produce the NIXL read descriptor.
        target_offset: byte offset within the target tensor for this slice.
            For full-tensor receives this is 0; for sliced receives it's
            ``(global_range[0] - 0) * row_stride`` so the bytes land at
            the right place in a pre-allocated full-shape buffer.
        compile_target_filter: optional whitelist of acceptable source
            ``compile_target`` tags. If set, the planner only matches
            against sources whose ``compile_target`` is in this set —
            the standard safety net for staged kernel rollouts (e.g. only
            accept ``bf16_cast`` until the receiver is ready to consume
            ``cutlass_fp8``).
        required_compile_metadata: optional subset that every matched
            source's ``compile_metadata`` must agree with. E.g.
            ``{"block_size": 128}`` to require 128-block alignment.
        required_experts: **DEPRECATED shim.** The ranges-only way to want
            a subset of experts is to emit a ``SliceRequest`` with a
            ``global_range`` on the expert axis (one request per contiguous
            run for non-contiguous sets — see
            :func:`modelexpress.rl_expert_layout.expert_ids_to_contiguous_ranges`).
            When set, the planner additionally runs the transitional expert
            filter (only matches sources whose ``owned_expert_ids`` intersect
            this set); when ``None`` (default) the planner is pure range
            intersection. Retained for back-compat until callers migrate.
    """

    tensor_name: str
    global_range: tuple[int, int]
    shard_axis: int | None
    dtype: str
    receiver_rank: int = 0
    target_addr: int = 0
    target_offset: int = 0
    compile_target_filter: frozenset[str] | None = None
    required_compile_metadata: dict[str, object] = field(default_factory=dict)
    required_experts: frozenset[int] | None = None

    def __post_init__(self) -> None:
        lo, hi = self.global_range
        if lo < 0 or hi <= lo:
            raise ValueError(
                f"SliceRequest({self.tensor_name!r}): global_range "
                f"({lo}, {hi}) is empty or negative"
            )


@dataclass(frozen=True)
class SegmentPlan:
    """One planned RDMA segment — one read from one source into one target slice.

    Multiple ``SegmentPlan`` entries can land in the same target buffer
    (multi-source case) at different offsets; the receiver issues them in
    parallel and the bytes assemble in place because RDMA is one-sided.

    Args:
        source: the ``SliceOwnership`` this segment reads from.
        request: the ``SliceRequest`` this segment partially satisfies.
        source_range: (start, end) within the source's local shard, in
            source-local coordinates. ``source_range[0] = max(own_lo, req_lo)
            - own_lo`` etc.
        target_range: (start, end) within the request's global_range, in
            request-local coordinates. The receiver lands the bytes at
            ``request.target_addr + request.target_offset + target_range[0] *
            row_stride``.
        byte_count: bytes to transfer for this segment. Derived from the
            length of ``target_range`` times the row stride.
    """

    source: SliceOwnership
    request: SliceRequest
    source_range: tuple[int, int]
    target_range: tuple[int, int]
    byte_count: int


@dataclass(frozen=True)
class CoveragePlan:
    """Planner output for one (receiver_rank, refit_cycle) pair.

    Args:
        segments: every ``SegmentPlan`` the receiver should issue. Order
            doesn't matter for correctness (RDMA is one-sided so segments
            land independently), but for the bandwidth-optimal case the
            receiver should issue them concurrently and wait for all.
        missing: per-tensor list of "(tensor_name, request_range, reason)"
            for any request that couldn't be fully covered. Empty when
            the plan is complete.
    """

    segments: tuple[SegmentPlan, ...]
    missing: tuple[tuple[str, tuple[int, int], str], ...] = ()

    @property
    def complete(self) -> bool:
        """True iff no requests have uncovered ranges."""
        return not self.missing

    def raise_if_incomplete(self) -> None:
        """Raise :class:`PlanIncompleteError` if any request is missing
        coverage. Strict failure mode — callers that prefer silent
        partial loads (almost never the right choice for weight refit)
        can check ``complete`` instead.
        """
        if not self.complete:
            details = "; ".join(
                f"{name} {rng} ({reason})" for name, rng, reason in self.missing
            )
            raise PlanIncompleteError(f"plan has uncovered requests: {details}")


class PlanIncompleteError(RuntimeError):
    """Raised when :meth:`CoveragePlan.raise_if_incomplete` is called on a
    plan that has uncovered requests."""


class QuantizationMetadataError(RuntimeError):
    """Raised by the planner when a request would land on a
    ``global-required`` quantization source via zero-copy. Callers can
    catch this and fall back to a full-copy install path."""
