# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DTensor placement → MX wire bridge for v2 NemoRL integration.

Translates PyTorch's ``distributed.tensor.placement_types`` plus optional
MoE expert axis information into a small JSON payload we stash in
``SourceIdentity.extra_parameters[shape_registry]``.

The v2 design (`pensieve/RL/NemoRL/04_design_v2_moe_rank_to_rank.md`) wants
an explicit, versioned ``ShapeRegistry`` proto on the MX server. To unblock
the prototype without touching the proto + Rust server code paths, this
module implements the registry as a JSON document attached to each
trainer's ``WorkerMetadata.extra_parameters``. Receivers consult it to
know each tensor's:

  - global shape (un-sharded)
  - dtype
  - placement (REPLICATE | SHARD axis | PARTIAL axis)
  - shard range owned by *this* trainer's rank
  - whether it's a MoE expert tensor
  - which expert IDs this rank owns (when applicable)

The format is intentionally JSON-shaped so the eventual proto migration
is a near-mechanical lift. See `pensieve/RL/NemoRL/05_mx_helpers_needed.md`
for the proto-shape we'd graduate to.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import torch

try:
    from torch.distributed.tensor.placement_types import (
        Partial,
        Replicate,
        Shard,
    )

    _DTensor_AVAILABLE = True
except ImportError:  # torch < 2.4 or non-distributed builds
    Partial = Replicate = Shard = None  # type: ignore[misc, assignment]
    _DTensor_AVAILABLE = False


# Sentinel placement kinds. Match the eventual proto enum exactly.
PLACEMENT_REPLICATE = "REPLICATE"
PLACEMENT_SHARD = "SHARD"
PLACEMENT_PARTIAL = "PARTIAL"


@dataclasses.dataclass
class TensorDescriptorV2:
    """Per-tensor shape + placement + expert metadata.

    Fields:
        name: tensor's qualified name in ``model.state_dict()``.
        global_shape: shape of the *un-sharded* tensor across all DP/TP ranks.
        dtype: torch dtype string (``"bfloat16"``, ``"float16"``, ...).
        placement_kind: one of ``PLACEMENT_*``.
        shard_axis: only meaningful if ``placement_kind == PLACEMENT_SHARD`` or
            ``PLACEMENT_PARTIAL``.
        local_shard_range: ``(start, end)`` along ``shard_axis`` owned by the
            publisher's rank. ``None`` when ``REPLICATE``.
        is_expert: whether this tensor's leading axis is the MoE expert axis.
        expert_axis: index of the expert axis (only when ``is_expert``).
        owned_expert_ids: expert IDs the publisher's rank owns.
    """

    name: str
    global_shape: tuple[int, ...]
    dtype: str
    placement_kind: str = PLACEMENT_REPLICATE
    shard_axis: int = 0
    local_shard_range: tuple[int, int] | None = None
    is_expert: bool = False
    expert_axis: int = 0
    owned_expert_ids: tuple[int, ...] = ()
    # Per-tensor Megatron role + role-specific descriptor extras. None /
    # empty for DTensor / PrimeRL publishers; populated by Megatron-Core
    # publishers so the receiver can dispatch on the role per tensor.
    # See temp/NemoRL_Megatron_MX_Design.md §4 for the role enum and
    # the extras keys per role.
    megatron_role: str | None = None
    megatron_extras: dict[str, str] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "global_shape": list(self.global_shape),
            "dtype": self.dtype,
            "placement_kind": self.placement_kind,
            "shard_axis": self.shard_axis,
        }
        if self.local_shard_range is not None:
            d["local_shard_range"] = list(self.local_shard_range)
        if self.is_expert:
            d["is_expert"] = True
            d["expert_axis"] = self.expert_axis
            d["owned_expert_ids"] = list(self.owned_expert_ids)
        if self.megatron_role is not None:
            d["megatron_role"] = self.megatron_role
        if self.megatron_extras:
            d["megatron_extras"] = dict(self.megatron_extras)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TensorDescriptorV2":
        rng = d.get("local_shard_range")
        return cls(
            name=d["name"],
            global_shape=tuple(d["global_shape"]),
            dtype=d["dtype"],
            placement_kind=d.get("placement_kind", PLACEMENT_REPLICATE),
            shard_axis=int(d.get("shard_axis", 0)),
            local_shard_range=tuple(rng) if rng is not None else None,
            is_expert=bool(d.get("is_expert", False)),
            expert_axis=int(d.get("expert_axis", 0)),
            owned_expert_ids=tuple(d.get("owned_expert_ids", [])),
            megatron_role=d.get("megatron_role"),
            megatron_extras=dict(d.get("megatron_extras") or {}),
        )


def _dtype_to_str(dtype: torch.dtype) -> str:
    s = str(dtype)
    return s[len("torch.") :] if s.startswith("torch.") else s


def describe_tensor(
    *,
    name: str,
    tensor: torch.Tensor,
    rank: int,
    fsdp_world_size: int,
    is_expert: bool = False,
    expert_axis: int = 0,
    owned_expert_ids: tuple[int, ...] | set[int] | list[int] = (),
) -> TensorDescriptorV2:
    """Build a ``TensorDescriptorV2`` from a tensor + rank context.

    For a regular ``torch.Tensor`` (no ``placements`` attribute) this yields
    a ``REPLICATE`` descriptor. For a DTensor it inspects ``tensor.placements``
    and emits the matching ``SHARD`` / ``PARTIAL`` descriptor; the local shard
    range is computed assuming an even shard layout (every rank's local size
    is the same). Uneven shards are not supported in v0 — the caller should
    pre-pad or fall back to bucket pack.

    The returned descriptor refers to the **global** shape: i.e. the
    un-sharded full tensor. ``local_shard_range[0:1]`` describes the slice
    along ``shard_axis`` that this rank owns.
    """
    dtype_str = _dtype_to_str(tensor.dtype)
    placements = getattr(tensor, "placements", None)
    if not _DTensor_AVAILABLE or not placements:
        return TensorDescriptorV2(
            name=name,
            global_shape=tuple(int(s) for s in tensor.shape),
            dtype=dtype_str,
            placement_kind=PLACEMENT_REPLICATE,
            is_expert=is_expert,
            expert_axis=expert_axis,
            owned_expert_ids=tuple(sorted(owned_expert_ids)),
        )

    if len(placements) != 1:
        # v0 supports only 1D meshes (FSDP only or TP only). HSDP / 2D meshes
        # are deferred — see `04_design_v2_moe_rank_to_rank.md` §7.
        raise NotImplementedError(
            f"DTensor with {len(placements)}D mesh is not supported in v0; "
            f"only 1D meshes are. tensor={name}"
        )
    p = placements[0]

    if isinstance(p, Replicate):
        return TensorDescriptorV2(
            name=name,
            global_shape=tuple(int(s) for s in tensor.shape),
            dtype=dtype_str,
            placement_kind=PLACEMENT_REPLICATE,
            is_expert=is_expert,
            expert_axis=expert_axis,
            owned_expert_ids=tuple(sorted(owned_expert_ids)),
        )

    if isinstance(p, Shard):
        # ``tensor.shape`` semantics differ between real DTensors and plain
        # tensors:
        #   * Real ``torch.distributed.tensor.DTensor.shape`` is the GLOBAL
        #     (un-sharded) shape; the local view is ``tensor.to_local().shape``.
        #   * A plain tensor (or a stand-in object with ``.placements`` but no
        #     ``.to_local``) has shape == local-view by construction.
        # Compute global vs local accordingly.
        try:
            from torch.distributed.tensor import DTensor as _RealDTensor
        except ImportError:  # pragma: no cover — handled at module import
            _RealDTensor = None  # type: ignore[assignment]

        if _RealDTensor is not None and isinstance(tensor, _RealDTensor):
            global_shape = tuple(int(s) for s in tensor.shape)
            try:
                local_extent = int(tensor.to_local().shape[p.dim])
            except Exception:
                # Fallback: assume even sharding.
                local_extent = global_shape[p.dim] // fsdp_world_size
        else:
            local_shape = list(int(s) for s in tensor.shape)
            global_shape_list = list(local_shape)
            global_shape_list[p.dim] = local_shape[p.dim] * fsdp_world_size
            global_shape = tuple(global_shape_list)
            local_extent = local_shape[p.dim]

        start = rank * local_extent
        end = start + local_extent
        return TensorDescriptorV2(
            name=name,
            global_shape=global_shape,
            dtype=dtype_str,
            placement_kind=PLACEMENT_SHARD,
            shard_axis=int(p.dim),
            local_shard_range=(start, end),
            is_expert=is_expert,
            expert_axis=expert_axis,
            owned_expert_ids=tuple(sorted(owned_expert_ids)),
        )

    if isinstance(p, Partial):
        return TensorDescriptorV2(
            name=name,
            global_shape=tuple(int(s) for s in tensor.shape),
            dtype=dtype_str,
            placement_kind=PLACEMENT_PARTIAL,
            shard_axis=int(p.dim) if hasattr(p, "dim") else 0,
        )

    raise NotImplementedError(f"unsupported DTensor placement: {p!r}")


def even_expert_owner_map(
    *,
    num_experts: int,
    ep_world_size: int,
) -> dict[int, set[int]]:
    """Default linear shard: rank N owns experts ``[N*chunk : (N+1)*chunk)``.

    Used for sanity-checking that an MoE layout matches what the trainer
    publishes vs what the inference rank expects to receive.
    """
    if ep_world_size <= 0:
        raise ValueError("ep_world_size must be positive")
    if num_experts % ep_world_size != 0:
        raise ValueError(
            f"num_experts={num_experts} not divisible by ep_world_size={ep_world_size}; "
            f"uneven expert assignment requires explicit owner map"
        )
    chunk = num_experts // ep_world_size
    return {r: set(range(r * chunk, (r + 1) * chunk)) for r in range(ep_world_size)}


def encode_registry(
    descriptors: list[TensorDescriptorV2],
    *,
    version: int,
    trainer_world_layout: str,
    extras: dict[str, Any] | None = None,
) -> str:
    """Serialize a registry to a string for ``extra_parameters``.

    ``extras`` is an optional dict of additional top-level keys to merge
    into the registry payload. Used by Megatron-Core publishes to carry
    the transformer config + Megatron→HF name map alongside the
    per-tensor descriptors. Receivers pick the keys up via
    :func:`decode_registry` (which round-trips unknown top-level keys
    unchanged).
    """
    payload: dict[str, Any] = {
        "version": int(version),
        "trainer_world_layout": trainer_world_layout,
        "tensors": [d.to_dict() for d in descriptors],
    }
    if extras:
        for key, value in extras.items():
            # Don't let extras shadow the structural keys.
            if key in ("version", "trainer_world_layout", "tensors"):
                continue
            payload[key] = value
    return json.dumps(payload, separators=(",", ":"))


def decode_registry(blob: str) -> dict[str, Any]:
    """Inverse of :func:`encode_registry`.

    Returns the full registry dict — at minimum ``{version,
    trainer_world_layout, tensors}``, plus any extras the publisher
    attached (round-tripped unchanged). Receivers inspect the dict
    directly for the keys they care about; unknown keys are preserved
    so future fields don't require coordinated upgrades.
    """
    parsed = json.loads(blob)
    parsed["tensors"] = [
        TensorDescriptorV2.from_dict(t) for t in parsed.get("tensors", [])
    ]
    return parsed


def encode_expert_set(expert_ids: set[int] | list[int] | tuple[int, ...]) -> str:
    """Compact encoding for an expert id set, used in ``extra_parameters``."""
    return ",".join(str(int(e)) for e in sorted(set(expert_ids)))


def decode_expert_set(s: str | None) -> set[int]:
    if not s:
        return set()
    return {int(p) for p in s.split(",") if p.strip()}


__all__ = [
    "PLACEMENT_PARTIAL",
    "PLACEMENT_REPLICATE",
    "PLACEMENT_SHARD",
    "TensorDescriptorV2",
    "decode_expert_set",
    "decode_registry",
    "describe_tensor",
    "encode_expert_set",
    "encode_registry",
    "even_expert_owner_map",
]
