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

# Canonical compile targets. The string set is open — frameworks can introduce
# new targets without an MX bump — but receivers should treat targets they
# don't recognise as "do not consume". Always pair a non-HF target with
# ``compile_metadata`` describing the engine/kernel/quant choices that drive
# byte-level compatibility.
COMPILE_TARGET_HF_RAW = "hf_raw"
COMPILE_TARGET_VLLM_FUSED = "vllm_fused"
COMPILE_TARGET_DEEPGEMM_FP8 = "deep_gemm_fp8"
COMPILE_TARGET_CUTLASS_FP8 = "cutlass_fp8"
COMPILE_TARGET_TRTLLM = "trtllm"


@dataclasses.dataclass
class TensorDescriptorV2:
    """Per-tensor shape + placement + expert + compile metadata.

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
        compile_target: kernel layout label this tensor's bytes are encoded
            for. ``"hf_raw"`` is the safe default — the trainer's HF
            state-dict view, no post-processing. Other publishers may emit
            ``"deep_gemm_fp8"``, ``"cutlass_fp8"``, ``"vllm_fused"``, etc.
            Receivers filter on this via
            :meth:`MxV2RefitReceiver.discover_v2_sources` so they only consume
            sources whose layout they can decode.
        compile_metadata: free-form key/value blob describing the specific
            compile invocation (e.g. ``{"engine": "DeepGemm", "version":
            "0.1.7", "block_size": 128, "scale_layout": "K-major"}``).
            Receivers should treat a mismatch on any byte-affecting field as
            a hard reject even if ``compile_target`` matches.
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
    compile_target: str = COMPILE_TARGET_HF_RAW
    compile_metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

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
        if self.compile_target != COMPILE_TARGET_HF_RAW:
            d["compile_target"] = self.compile_target
        if self.compile_metadata:
            d["compile_metadata"] = dict(self.compile_metadata)
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
            compile_target=str(d.get("compile_target", COMPILE_TARGET_HF_RAW)),
            compile_metadata=dict(d.get("compile_metadata", {})),
        )


def _dtype_to_str(dtype: torch.dtype) -> str:
    s = str(dtype)
    return s[len("torch.") :] if s.startswith("torch.") else s


@dataclasses.dataclass(frozen=True)
class NonExpertShardSpec:
    """Explicit shard metadata for a plain (non-DTensor) tensor.

    ``describe_tensor`` infers placement from a DTensor's ``placements``,
    but a caller that holds an already-materialized *plain* shard (e.g. a
    post-conversion output buffer that is no longer a DTensor) has no
    placement info to infer from. Passing a ``NonExpertShardSpec`` lets
    such a caller publish its buffer as a ``SHARD`` of a larger logical
    tensor so receivers can pull + reassemble across ranks instead of
    the publisher pre-gathering the full tensor.

    Fields:
        global_shape: the un-sharded full tensor's shape.
        shard_axis: axis along which the tensor is sharded across ranks.
        local_shard_range: ``(start, end)`` along ``shard_axis`` that the
            publisher's rank owns. End-exclusive.
    """

    global_shape: tuple[int, ...]
    shard_axis: int
    local_shard_range: tuple[int, int]


def describe_tensor(
    *,
    name: str,
    tensor: torch.Tensor,
    rank: int,
    fsdp_world_size: int,
    is_expert: bool = False,
    expert_axis: int = 0,
    owned_expert_ids: tuple[int, ...] | set[int] | list[int] = (),
    compile_target: str = COMPILE_TARGET_HF_RAW,
    compile_metadata: dict[str, Any] | None = None,
    shard_spec: NonExpertShardSpec | None = None,
) -> TensorDescriptorV2:
    """Build a ``TensorDescriptorV2`` from a tensor + rank context.

    For a regular ``torch.Tensor`` (no ``placements`` attribute) this yields
    a ``REPLICATE`` descriptor. For a DTensor it inspects ``tensor.placements``
    and emits the matching ``SHARD`` / ``PARTIAL`` descriptor; the local shard
    range is computed assuming an even shard layout (every rank's local size
    is the same). Uneven shards are not supported in v0 — the caller should
    pre-pad or fall back to bucket pack.

    When ``shard_spec`` is provided the tensor is treated as an explicit
    ``SHARD`` of ``shard_spec.global_shape`` regardless of whether it is a
    DTensor — this is the path for publishing a plain, already-materialized
    shard (e.g. a post-conversion output buffer) so receivers can pull and
    reassemble it. The caller owns the shard math; we validate the local
    extent matches the tensor's actual size along ``shard_axis``.

    The returned descriptor refers to the **global** shape: i.e. the
    un-sharded full tensor. ``local_shard_range[0:1]`` describes the slice
    along ``shard_axis`` that this rank owns.
    """
    dtype_str = _dtype_to_str(tensor.dtype)
    metadata = dict(compile_metadata) if compile_metadata else {}

    # Explicit shard spec short-circuit: publish a plain buffer as a SHARD
    # of a larger logical tensor. Used by frameworks whose post-conversion
    # buffers are plain tensors (not DTensors) but represent a per-rank
    # FSDP/TP shard the receiver must reassemble.
    if shard_spec is not None:
        lo, hi = shard_spec.local_shard_range
        axis = shard_spec.shard_axis
        local_extent = int(tensor.shape[axis])
        if hi - lo != local_extent:
            raise ValueError(
                f"describe_tensor({name!r}): shard_spec local_shard_range "
                f"({lo}, {hi}) width {hi - lo} != tensor extent {local_extent} "
                f"on axis {axis}"
            )
        if hi > shard_spec.global_shape[axis] or lo < 0:
            raise ValueError(
                f"describe_tensor({name!r}): shard_spec range ({lo}, {hi}) "
                f"outside global_shape[{axis}]={shard_spec.global_shape[axis]}"
            )
        return TensorDescriptorV2(
            name=name,
            global_shape=tuple(int(s) for s in shard_spec.global_shape),
            dtype=dtype_str,
            placement_kind=PLACEMENT_SHARD,
            shard_axis=axis,
            local_shard_range=(lo, hi),
            is_expert=is_expert,
            expert_axis=expert_axis,
            owned_expert_ids=tuple(sorted(owned_expert_ids)),
            compile_target=compile_target,
            compile_metadata=metadata,
        )

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
            compile_target=compile_target,
            compile_metadata=metadata,
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
            compile_target=compile_target,
            compile_metadata=metadata,
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
            except Exception:  # noqa: BLE001
                # Fallback: assume even sharding.
                local_extent = global_shape[p.dim] // fsdp_world_size
            # ``start = rank * local_extent`` is only correct when every
            # rank's shard along this axis has the same extent. Real
            # DTensors may shard unevenly (e.g. ``Shard(0)`` over a
            # non-divisible axis), which would publish a wrong
            # ``local_shard_range`` for ranks > 0. v0 of the v2 slice
            # planner doesn't support uneven sharding, so refuse to emit
            # a wrong descriptor — callers can either change the layout
            # or pass an explicit ``target_range`` on the receiver side.
            if (
                fsdp_world_size > 0
                and global_shape[p.dim] != local_extent * fsdp_world_size
            ):
                raise RuntimeError(
                    f"tensor {name!r} has uneven DTensor sharding on axis "
                    f"{p.dim}: global_extent={global_shape[p.dim]} but "
                    f"local_extent={local_extent} × world_size="
                    f"{fsdp_world_size}. v2 v0 requires uniform shard "
                    "sizes; pass a custom target_range on the receiver "
                    "side or rebalance the layout."
                )
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
            compile_target=compile_target,
            compile_metadata=metadata,
        )

    if isinstance(p, Partial):
        return TensorDescriptorV2(
            name=name,
            global_shape=tuple(int(s) for s in tensor.shape),
            dtype=dtype_str,
            placement_kind=PLACEMENT_PARTIAL,
            shard_axis=int(p.dim) if hasattr(p, "dim") else 0,
            compile_target=compile_target,
            compile_metadata=metadata,
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
) -> str:
    """Serialize a registry to a string for ``extra_parameters``."""
    payload = {
        "version": int(version),
        "trainer_world_layout": trainer_world_layout,
        "tensors": [d.to_dict() for d in descriptors],
    }
    return json.dumps(payload, separators=(",", ":"))


def decode_registry(blob: str) -> dict[str, Any]:
    """Inverse of ``encode_registry``. Returns ``{version, trainer_world_layout, tensors}``."""
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


def compile_target_matches(
    descriptor: TensorDescriptorV2,
    *,
    allowed_targets: set[str] | frozenset[str] | None,
    required_metadata: dict[str, Any] | None = None,
) -> bool:
    """Return True if ``descriptor`` is acceptable to a receiver.

    Args:
        descriptor: the publisher-side descriptor (its ``compile_target`` and
            ``compile_metadata`` describe how the bytes are laid out).
        allowed_targets: receiver-side whitelist of compile-target strings the
            receiver knows how to consume. ``None`` means "accept everything"
            (back-compat shim — equivalent to the v0 behaviour).
        required_metadata: optional key/value subset the descriptor's
            ``compile_metadata`` must agree with byte-for-byte. Useful for
            pinning e.g. ``{"block_size": 128, "scale_layout": "K-major"}``
            so a Cutlass receiver doesn't accept a DeepGemm-block-256
            publisher's bytes by mistake.
    """
    if allowed_targets is not None and descriptor.compile_target not in allowed_targets:
        return False
    if required_metadata:
        for key, want in required_metadata.items():
            if descriptor.compile_metadata.get(key) != want:
                return False
    return True


__all__ = [
    "COMPILE_TARGET_CUTLASS_FP8",
    "COMPILE_TARGET_DEEPGEMM_FP8",
    "COMPILE_TARGET_HF_RAW",
    "COMPILE_TARGET_TRTLLM",
    "COMPILE_TARGET_VLLM_FUSED",
    "PLACEMENT_PARTIAL",
    "PLACEMENT_REPLICATE",
    "PLACEMENT_SHARD",
    "TensorDescriptorV2",
    "compile_target_matches",
    "decode_expert_set",
    "decode_registry",
    "describe_tensor",
    "encode_expert_set",
    "encode_registry",
    "even_expert_owner_map",
]
