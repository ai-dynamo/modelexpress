# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vendored Megatron-Core → HuggingFace tensor-math helpers for the MX v2
Megatron receiver path (Phase C).

These functions are byte-identical ports of the standalone helpers in
``megatron.bridge.models.conversion.param_mapping`` (Bridge upstream:
https://github.com/NVIDIA-NeMo/Megatron-Bridge). They are vendored here so
the receiver-side translator can use them without taking a hard dependency
on Megatron-Bridge in the inference image (the production Dynamo
VllmDecodeWorker image at ``jwillthomson/dynamo-arm-tokenize-endpoint-...``
does not ship Megatron-Core / Megatron-Bridge — only modelexpress).

Correctness contract: the test ``tests/test_megatron_helpers.py`` runs
each helper here against Bridge's authoritative implementation and
asserts byte-identical outputs on a representative MHA + GQA matrix.
That cross-check runs as part of MX's CI loop, so any drift between
this vendored copy and Bridge's upstream is caught immediately. Bridge
is therefore a *test-time-only* dependency; runtime receivers don't
import it.

Subset rationale: this v0 only vendors the helpers needed for standard
MHA / GQA QKV (`split_qkv_weights`, `split_qkv_biases`) and gated-MLP
chunk. Out of scope for this version (defer to a later vendoring pass
when the production NemoRL trainer hits these layouts):

* ``attention_output_gate=True`` (Megatron-Core gated-attention layout
  ``[Q, gate, K, V]`` per query group)
* ``merge_qkvg_weights`` / ``split_qkvg_weights`` (Q-K-V-Gate fused linear)
* ``merge_kv_weights`` / ``split_kv_weights`` (KV-only fused linear)
* FP8 blockwise scale tensors (the ``feature_dim`` / scale-domain branch
  of Bridge's ``split_qkv_weights``)
* Mamba/GDN linear / conv layouts

For any of those, fall back to "have Bridge installed in the test
environment and paste its output as ground truth"; or extend this
module + its CI cross-check.

The :class:`MegatronTransformerConfig` dataclass below is a minimal
config holder that mirrors the fields the helpers read off
``TransformerConfig``. The trainer-side publisher serializes these
fields into the v2 sidecar (see ``MxV2TrainingPublisher``); the
receiver reconstructs the dataclass before invoking the helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class MegatronTransformerConfig:
    """Minimal subset of Megatron-Core's TransformerConfig used by the
    QKV / gated-MLP helpers. Carried in the v2 sidecar so the receiver
    can reconstruct the fields without a Megatron-Core import.
    """

    num_attention_heads: int
    num_query_groups: int           # equals num_attention_heads for MHA
    kv_channels: int | None         # if None, derived from hidden_size // num_heads
    hidden_size: int

    @property
    def head_size(self) -> int:
        if self.kv_channels:
            return self.kv_channels
        return self.hidden_size // self.num_attention_heads

    def to_dict(self) -> dict[str, int | None]:
        return {
            "num_attention_heads": self.num_attention_heads,
            "num_query_groups": self.num_query_groups,
            "kv_channels": self.kv_channels,
            "hidden_size": self.hidden_size,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MegatronTransformerConfig":
        return cls(
            num_attention_heads=int(d["num_attention_heads"]),
            num_query_groups=int(d["num_query_groups"]),
            kv_channels=int(d["kv_channels"]) if d.get("kv_channels") is not None else None,
            hidden_size=int(d["hidden_size"]),
        )


# ---------------------------------------------------------------------------
# QKV — port of Bridge's split_qkv_biases
# (megatron/bridge/models/conversion/param_mapping.py L3073-3125)
# ---------------------------------------------------------------------------


def split_qkv_biases(
    config: MegatronTransformerConfig,
    qkv: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split Megatron's interleaved QKV bias into separate Q, K, V biases.

    Megatron lays out QKV biases as
    ``[group0: q*ratio, k, v, group1: q*ratio, k, v, ...]`` where
    ``ratio = num_attention_heads / num_query_groups``.

    Args:
        config: Megatron transformer config (head counts + dims).
        qkv: Interleaved QKV bias as a 1-D tensor of shape
            ``(num_attention_heads + 2 * num_query_groups) * head_size``.

    Returns:
        Tuple ``(q, k, v)`` of 1-D bias tensors.
    """
    head_num = config.num_attention_heads
    num_query_groups = config.num_query_groups
    heads_per_group = head_num // num_query_groups
    head_size = config.head_size

    qkv_total_dim = head_num + 2 * num_query_groups
    total_heads_per_group = heads_per_group + 2

    qkv_reshaped = qkv.reshape(qkv_total_dim, head_size)

    q_slice = torch.cat(
        [
            torch.arange(
                total_heads_per_group * i, total_heads_per_group * i + heads_per_group
            )
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(total_heads_per_group - 2, qkv_total_dim, total_heads_per_group)
    v_slice = torch.arange(total_heads_per_group - 1, qkv_total_dim, total_heads_per_group)

    q = qkv_reshaped[q_slice].flatten()
    k = qkv_reshaped[k_slice].flatten()
    v = qkv_reshaped[v_slice].flatten()
    return q, k, v


# ---------------------------------------------------------------------------
# QKV — port of Bridge's split_qkv_weights (basic non-FP8 path)
# (megatron/bridge/models/conversion/param_mapping.py L3243-3354)
# ---------------------------------------------------------------------------


def split_qkv_weights(
    config: MegatronTransformerConfig,
    qkv: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split Megatron's interleaved QKV weight tensor into separate Q, K, V.

    Args:
        config: Megatron transformer config.
        qkv: Interleaved QKV weight tensor with shape
            ``((num_attention_heads + 2 * num_query_groups) * head_size, hidden_size)``.

    Returns:
        Tuple ``(q, k, v)`` of 2-D weight tensors with shapes
        ``(num_attention_heads * head_size, hidden_size)`` and
        ``(num_query_groups * head_size, hidden_size)`` for K, V.
    """
    head_num = config.num_attention_heads
    num_query_groups = config.num_query_groups
    heads_per_group = head_num // num_query_groups
    head_size = config.head_size
    hidden_size = config.hidden_size

    qkv_total_dim = head_num + 2 * num_query_groups
    total_heads_per_group = heads_per_group + 2

    if qkv.ndim == 1:
        # Caller meant biases — defer.
        return split_qkv_biases(config, qkv)

    if qkv.shape[-1] != hidden_size:
        # FP8 blockwise scale tensors compress the trailing dim. Bridge's
        # full implementation handles this; v0 vendored helper rejects it
        # explicitly so the caller knows to upgrade.
        raise NotImplementedError(
            f"split_qkv_weights vendored v0 expects qkv.shape[-1] == hidden_size "
            f"({hidden_size}); got {qkv.shape[-1]}. FP8 blockwise scale support "
            f"requires extending the vendored helper or upgrading to "
            f"megatron.bridge.models.conversion.param_mapping.split_qkv_weights."
        )

    qkv_reshaped = qkv.view(qkv_total_dim, head_size, hidden_size)

    q_slice = torch.cat(
        [
            torch.arange(
                total_heads_per_group * i, total_heads_per_group * i + heads_per_group
            )
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(total_heads_per_group - 2, qkv_total_dim, total_heads_per_group)
    v_slice = torch.arange(total_heads_per_group - 1, qkv_total_dim, total_heads_per_group)

    q = qkv_reshaped[q_slice]
    k = qkv_reshaped[k_slice]
    v = qkv_reshaped[v_slice]

    assert q.numel() + k.numel() + v.numel() == qkv.numel(), (
        f"split_qkv_weights size mismatch: "
        f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} qkv={tuple(qkv.shape)}"
    )

    q = q.reshape(-1, hidden_size)
    k = k.reshape(-1, hidden_size)
    v = v.reshape(-1, hidden_size)
    return q, k, v


# ---------------------------------------------------------------------------
# Inverse helpers — useful in tests and for trainer-side ground-truth
# generation. Direct ports of Bridge's merge_qkv_weights / merge_qkv_biases.
# ---------------------------------------------------------------------------


def merge_qkv_weights(
    config: MegatronTransformerConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Merge separate Q, K, V weight matrices into Megatron's interleaved
    QKV format. Inverse of :func:`split_qkv_weights`."""
    head_num = config.num_attention_heads
    num_query_groups = config.num_query_groups
    heads_per_group = head_num // num_query_groups
    head_size = config.head_size
    hidden_size = config.hidden_size
    is_bias = q.ndim == 1

    if is_bias:
        q_reshaped = q.view(head_num, head_size)
        k_reshaped = k.view(num_query_groups, head_size)
        v_reshaped = v.view(num_query_groups, head_size)
    else:
        q_reshaped = q.view(head_num, head_size, hidden_size)
        k_reshaped = k.view(num_query_groups, head_size, hidden_size)
        v_reshaped = v.view(num_query_groups, head_size, hidden_size)

    qkv_weights = []
    for i in range(num_query_groups):
        qkv_weights.append(q_reshaped[i * heads_per_group : (i + 1) * heads_per_group])
        qkv_weights.append(k_reshaped[i : i + 1])
        qkv_weights.append(v_reshaped[i : i + 1])

    qkv = torch.cat(qkv_weights, dim=0)
    assert q.numel() + k.numel() + v.numel() == qkv.numel(), (
        f"merge_qkv_weights size mismatch: "
        f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} qkv={tuple(qkv.shape)}"
    )
    if is_bias:
        return qkv.reshape(-1)
    return qkv.reshape(-1, hidden_size)


def merge_qkv_biases(
    config: MegatronTransformerConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Inverse of :func:`split_qkv_biases`."""
    return merge_qkv_weights(config, q, k, v)


# ---------------------------------------------------------------------------
# Gated MLP — Megatron stores gate+up fused along dim 0; HF expects them
# split as gate_proj / up_proj. Pure passthrough; trivial helper here for
# clarity + symmetry with the QKV pair.
# ---------------------------------------------------------------------------


def split_gated_mlp(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split Megatron's fused gate+up linear into (gate, up).

    Megatron-Core's convention is ``[gate; up]`` along output dim 0; HF
    materializes them as separate ``mlp.gate_proj.weight`` and
    ``mlp.up_proj.weight``.

    Use this helper when the input was assembled from a SINGLE Megatron
    source rank (matched-TP or per-expert grouped). For inputs assembled
    by concatenating multiple TP-sharded ranks, use
    :func:`split_gated_mlp_tp` instead — the per-rank layout is
    ``[gate_local; up_local]`` so a straight chunk-2 produces a mixed
    ``[gate_r0; up_r0]`` / ``[gate_r1; up_r1]`` split rather than the
    un-interleaved ``[gate_global]`` / ``[up_global]`` HF expects.
    """
    gate, up = tensor.chunk(2, dim=0)
    return gate, up


def split_gated_mlp_tp(
    tensor: torch.Tensor, tp: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Un-interleave a TP-concatenated fused gate+up tensor into (gate, up).

    Megatron-Core's fused gate+up ColumnParallelLinear stores each rank's
    weight as ``[gate_local; up_local]`` (gate-then-up of that rank's
    intermediate slice) along the output axis. When the receiver
    assembles N TP-rank shards via ``concat_dim0``, the resulting tensor
    has the interleaved layout::

        [gate_r0; up_r0; gate_r1; up_r1; ...; gate_r(N-1); up_r(N-1)]

    HF's ``mlp.gate_proj.weight`` and ``mlp.up_proj.weight`` want the
    un-interleaved::

        gate = [gate_r0; gate_r1; ...; gate_r(N-1)]
        up   = [up_r0;   up_r1;   ...; up_r(N-1)]

    This helper does the un-interleave via a single contiguous reshape +
    advanced indexing (no per-rank copies). It's the inverse of the
    Megatron publisher's per-rank ``merge_gated_mlp`` (which produces
    ``[gate_local; up_local]`` per rank) followed by an axis-0 concat.

    When ``tp == 1`` this is exactly :func:`split_gated_mlp`. The
    receiver-side translator dispatches on ``len(plan.sources)`` to
    choose between the two.
    """
    total_rows = tensor.shape[0]
    rest = tensor.shape[1:]
    if total_rows % (2 * tp) != 0:
        raise ValueError(
            f"split_gated_mlp_tp: leading dim {total_rows} not divisible by "
            f"2 * tp = {2 * tp}"
        )
    inter_per_rank = total_rows // (2 * tp)
    # Reshape into (tp, {gate=0, up=1}, inter_per_rank, ...rest).
    reshaped = tensor.view(tp, 2, inter_per_rank, *rest)
    # gate_per_rank is reshaped[:, 0]; up_per_rank is reshaped[:, 1].
    # Flatten the (tp, inter_per_rank) dims back together.
    gate = reshaped[:, 0].reshape(tp * inter_per_rank, *rest).contiguous()
    up = reshaped[:, 1].reshape(tp * inter_per_rank, *rest).contiguous()
    return gate, up


def merge_gated_mlp(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`split_gated_mlp` — concatenate ``[gate; up]``."""
    return torch.cat([gate, up], dim=0)


__all__ = [
    "MegatronTransformerConfig",
    "split_qkv_weights",
    "split_qkv_biases",
    "merge_qkv_weights",
    "merge_qkv_biases",
    "split_gated_mlp",
    "split_gated_mlp_tp",
    "merge_gated_mlp",
]
