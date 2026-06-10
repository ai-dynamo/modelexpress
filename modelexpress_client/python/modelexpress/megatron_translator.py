# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Receiver-side Megatron-MX translator (Phase C of the Megatron-MX path).

Orchestrates the full receiver flow for a Megatron trainer's published
weights:

    1. discover_v2_sources(target_tp_layout)        — Phase B
    2. pick_megatron_slice_plans(target_tensor_specs)
    3. NIXL parallel pull into pre-allocated global tensors
    4. translate Megatron → HF via the vendored helpers
    5. yield (hf_name, hf_tensor) for the caller's `model.load_weights()`

Framework-agnostic — both NemoRL's direct-vLLM path and Dynamo's
``MxRefitWorkerExtension`` can use this. The caller supplies the
``ReceiveSpec`` list (one entry per HF parameter the inference model
needs); steps 1-5 are owned by this module.

See ``temp/NemoRL_Megatron_MX_Design.md`` §6 + §9b and
``temp/NemoRL_Megatron_MX_Phase_C_Handoff.md`` for the full design.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import torch

from .megatron_helpers import (
    MegatronTransformerConfig,
    split_qkv_biases,
    split_qkv_weights,
    split_gated_mlp,
)
from .nemo_rl_v2 import (
    MegatronSlicePlan,
    MegatronSliceSource,
    MegatronTensorSpec,
    MxV2RefitReceiver,
    ROLE_MEGATRON_COLUMN,
    ROLE_MEGATRON_EXPERT_COLUMN,
    ROLE_MEGATRON_EXPERT_ROW,
    ROLE_MEGATRON_GATED_MLP_COLUMN,
    ROLE_MEGATRON_QKV_COLUMN,
    ROLE_MEGATRON_REPLICATED,
    ROLE_MEGATRON_ROW,
    ROLE_MEGATRON_VOCAB_PARALLEL,
    TargetTpLayout,
    V2SourceCandidate,
)

logger = logging.getLogger("modelexpress.megatron_translator")


# Sentinel string for the Megatron transformer-config sidecar key. The
# trainer-side publisher serializes ``MegatronTransformerConfig.to_dict()``
# into the v2 sidecar under this key; the receiver reconstructs the config
# before driving the QKV un-interleave.
SIDECAR_TRANSFORMER_CONFIG_KEY = "megatron_transformer_config"

# Key for the Megatron→HF naming list in the sidecar. The trainer publishes
# a list of ``[megatron_name, [hf_name_1, hf_name_2, ...]]`` so the receiver
# knows which HF names each Megatron tensor should produce. Derived once at
# trainer init via Bridge's ``export_hf_weights`` introspection.
SIDECAR_HF_NAME_MAP_KEY = "megatron_hf_name_map"


# ---------------------------------------------------------------------------
# Receiver-spec dataclass
# ---------------------------------------------------------------------------


@dataclass
class ReceiveSpec:
    """One HF parameter the receiver wants this refit cycle.

    The receiver builds these per-parameter from the inference model's
    own state-dict shape + dtype, plus the trainer's published Megatron
    metadata (role + name map). One ReceiveSpec corresponds to one
    Megatron tensor (e.g. ``decoder.layers.0.self_attention.linear_qkv.weight``)
    that may unfold into multiple HF tensors after translation
    (e.g. ``q_proj``, ``k_proj``, ``v_proj``).

    Fields:
        megatron_name: source-side tensor name (Megatron-shaped, the name
            the trainer published under).
        hf_names: HF-shaped names this Megatron tensor expands into.
            For ``replicated`` / ``column`` / ``row`` / ``vocab_parallel``:
            length 1. For ``qkv_column``: length 3
            (``q_proj.weight``, ``k_proj.weight``, ``v_proj.weight``).
            For ``gated_mlp_column``: length 2
            (``gate_proj.weight``, ``up_proj.weight``).
            For ``expert_column`` / ``expert_row``: length depends on the
            number of local experts on this receiver rank.
        role: Megatron role (one of ``ROLE_MEGATRON_*``).
        target_shape: GLOBAL Megatron-side shape that the receiver should
            assemble the per-rank slices into. After translation, the HF
            outputs may be smaller (e.g. for QKV: each q/k/v is a
            partition of this assembled tensor).
        target_dtype: dtype string ("bfloat16", "float16", "float32").
        shard_axis: axis along which TP sources are tiled.
        pp_rank: which PP stage of the trainer's mesh owns this tensor.
        role_descriptor: per-role extras forwarded to the planner. For
            QKV / gated_mlp this holds the role-specific keys
            (``num_heads_total``, ``head_dim``, ``gated_mlp_order``, ...).
    """

    megatron_name: str
    hf_names: list[str]
    role: str
    target_shape: tuple[int, ...]
    target_dtype: str
    shard_axis: int = 0
    pp_rank: int = 0
    role_descriptor: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# Sidecar parsing — extract Megatron-publisher metadata from a candidate
# ---------------------------------------------------------------------------


def parse_megatron_sidecar(
    candidate: V2SourceCandidate,
) -> tuple[MegatronTransformerConfig | None, dict[str, list[str]]]:
    """Pull Megatron transformer config + Megatron→HF name map from a
    source's v2 metadata.

    The trainer-side publisher serializes both into the v2 sidecar JSON
    so the receiver doesn't need a Bridge import to know per-arch naming
    or attention head config.

    Returns ``(config, hf_name_map)``. Either may be empty/None for sources
    that don't carry the keys (e.g. DTensor publishers, older builds).
    """
    cfg: MegatronTransformerConfig | None = None
    hf_name_map: dict[str, list[str]] = {}

    if candidate.registry is None:
        return cfg, hf_name_map

    # The registry is a JSON-decoded dict. We look for the two
    # Megatron-specific top-level keys; receivers tolerate their absence.
    cfg_blob = candidate.registry.get(SIDECAR_TRANSFORMER_CONFIG_KEY)
    if cfg_blob:
        try:
            if isinstance(cfg_blob, str):
                cfg_blob = json.loads(cfg_blob)
            cfg = MegatronTransformerConfig.from_dict(cfg_blob)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to parse megatron_transformer_config sidecar: %s", exc)

    map_blob = candidate.registry.get(SIDECAR_HF_NAME_MAP_KEY)
    if map_blob:
        try:
            if isinstance(map_blob, str):
                map_blob = json.loads(map_blob)
            for entry in map_blob:
                # Each entry is [megatron_name, [hf_name_1, hf_name_2, ...]]
                if isinstance(entry, (list, tuple)) and len(entry) == 2:
                    hf_name_map[str(entry[0])] = [str(x) for x in entry[1]]
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to parse megatron_hf_name_map sidecar: %s", exc)

    return cfg, hf_name_map


# ---------------------------------------------------------------------------
# Assembly — pre-allocate destination, parallel-pull into sliced views
# ---------------------------------------------------------------------------


def _torch_dtype(s: str) -> torch.dtype:
    s = s.strip()
    if s.startswith("torch."):
        s = s[len("torch."):]
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "float": torch.float32,
    }[s]


def assemble_into_destination(
    plan: MegatronSlicePlan,
    *,
    pull: Callable[[MegatronSliceSource, torch.Tensor], None],
    device: torch.device | str = "cuda",
) -> torch.Tensor | dict[int, torch.Tensor]:
    """Pull all sources for one tensor and assemble into a single buffer.

    For every assembly except ``per_expert``, returns a single global
    tensor with shape ``plan.target_shape``. For ``per_expert``, returns a
    dict ``{expert_id: tensor}`` since each expert is a separate HF output.

    The ``pull`` callback synchronously RDMAs the source's bytes into the
    destination view (or sub-slice for mixed-TP cases). For matched-TP
    matched sources ``source_subslice`` is None and the full source slice
    fills the target view.

    Pre-allocates the global destination once; per-source views point into
    the same buffer to avoid an extra ``torch.cat`` GPU memcpy.
    """
    target_dtype = _torch_dtype(plan.target_dtype)
    if plan.assembly == "per_expert":
        results: dict[int, torch.Tensor] = {}
        for src in plan.sources:
            expert_id = int(src.role_extras.get("expert_id", "-1"))
            if expert_id < 0:
                logger.warning("per_expert source missing expert_id: %r", src)
                continue
            buf = torch.empty(plan.target_shape, dtype=target_dtype, device=device)
            pull(src, buf)
            results[expert_id] = buf
        return results

    dest = torch.empty(plan.target_shape, dtype=target_dtype, device=device)

    if plan.assembly == "passthrough":
        if plan.sources:
            pull(plan.sources[0], dest)
        return dest

    # All other assemblies tile along a single axis (0 for column/qkv/
    # gated_mlp/vocab; 1 for row).
    axis = 1 if plan.assembly == "concat_dim1" else 0
    for src in plan.sources:
        lo, hi = src.target_local_range
        view = dest.narrow(axis, lo, hi - lo)
        pull(src, view)
    return dest


# ---------------------------------------------------------------------------
# Translation — vendored helpers do all the math
# ---------------------------------------------------------------------------


def translate_megatron_to_hf(
    plan: MegatronSlicePlan,
    assembled: torch.Tensor | dict[int, torch.Tensor],
    *,
    transformer_config: MegatronTransformerConfig,
    hf_names: list[str],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(hf_name, hf_tensor)`` pairs for one assembled tensor.

    The caller is responsible for resolving ``hf_names`` from the
    Megatron→HF name map (see :func:`parse_megatron_sidecar`) — this
    function consumes the resolved list and dispatches on
    ``plan.assembly``.

    Args:
        plan: The slice plan returned by Phase B's planner. Used for
            ``role`` / ``assembly`` dispatch and for ``role_descriptor``
            (head counts, gated-mlp order, etc.).
        assembled: For non per-expert: a single global tensor. For
            per_expert: a dict ``{expert_id: tensor}``.
        transformer_config: Megatron transformer config (head counts +
            dims). Required for QKV un-interleave; ignored otherwise.
        hf_names: HF-shaped names this tensor expands into. Length must
            match the role (1 for column/row/replicated/vocab_parallel,
            3 for qkv_column, 2 for gated_mlp_column, N for per_expert).
    """
    role = plan.role

    if role == ROLE_MEGATRON_REPLICATED:
        if not isinstance(assembled, torch.Tensor):
            raise TypeError("replicated assembly expects a single tensor")
        if len(hf_names) != 1:
            raise ValueError(
                f"replicated tensor expects 1 hf_name, got {hf_names}"
            )
        yield hf_names[0], assembled
        return

    if role in (ROLE_MEGATRON_COLUMN, ROLE_MEGATRON_ROW, ROLE_MEGATRON_VOCAB_PARALLEL):
        if not isinstance(assembled, torch.Tensor):
            raise TypeError(f"{role} assembly expects a single tensor")
        if len(hf_names) != 1:
            raise ValueError(f"{role} expects 1 hf_name, got {hf_names}")
        yield hf_names[0], assembled
        return

    if role == ROLE_MEGATRON_QKV_COLUMN:
        if not isinstance(assembled, torch.Tensor):
            raise TypeError("qkv_column assembly expects a single tensor")
        if len(hf_names) != 3:
            raise ValueError(
                f"qkv_column expects 3 hf_names (q, k, v); got {hf_names}"
            )
        if assembled.ndim == 1:
            q, k, v = split_qkv_biases(transformer_config, assembled)
        else:
            q, k, v = split_qkv_weights(transformer_config, assembled)
        yield hf_names[0], q
        yield hf_names[1], k
        yield hf_names[2], v
        return

    if role == ROLE_MEGATRON_GATED_MLP_COLUMN:
        if not isinstance(assembled, torch.Tensor):
            raise TypeError("gated_mlp_column assembly expects a single tensor")
        if len(hf_names) != 2:
            raise ValueError(
                f"gated_mlp_column expects 2 hf_names (gate, up); got {hf_names}"
            )
        gate, up = split_gated_mlp(assembled)
        yield hf_names[0], gate
        yield hf_names[1], up
        return

    if role in (ROLE_MEGATRON_EXPERT_COLUMN, ROLE_MEGATRON_EXPERT_ROW):
        # Two assembly shapes:
        #
        # (a) ``assembly == "passthrough"`` — the **grouped** layout (TE
        #     per-expert ``nn.Parameter`` named ``weight0`` / ``weight1`` /
        #     ...). The planner picks one source per Megatron tensor; the
        #     ``assembled`` tensor is THIS expert's full per-expert
        #     weight. Bridge's name map gives 1 HF name for plain
        #     ``linear_fc2`` (row-parallel down_proj) and 2 HF names for
        #     ``linear_fc1`` (column-parallel fused gate + up — needs
        #     ``split_gated_mlp``).
        #
        # (b) ``assembly == "per_expert"`` — the **leading-axis** layout
        #     (legacy EP>1 single ``.weight`` whose leading axis stacks
        #     experts). ``assembled`` is a ``dict[expert_id, tensor]``;
        #     ``hf_names`` is one-per-expert keyed by position OR routing
        #     map in ``plan.role_descriptor['hf_names_by_expert_id']``.
        if plan.assembly == "passthrough":
            if not isinstance(assembled, torch.Tensor):
                raise TypeError(
                    f"{role} passthrough assembly expects a single tensor"
                )
            if len(hf_names) == 1:
                yield hf_names[0], assembled
                return
            if len(hf_names) == 2:
                # Per-expert fused gate+up: same math as
                # ROLE_MEGATRON_GATED_MLP_COLUMN, just one expert at a
                # time. The vendored helper does the axis-0 split.
                gate, up = split_gated_mlp(assembled)
                yield hf_names[0], gate
                yield hf_names[1], up
                return
            raise ValueError(
                f"{role} passthrough expects 1 or 2 hf_names, got {hf_names}"
            )

        # Legacy per_expert assembly path: stacked-experts dict.
        if not isinstance(assembled, dict):
            raise TypeError(f"{role} per_expert assembly expects dict[expert_id, tensor]")
        # hf_names is keyed by position in this v0; the caller provides
        # the names as ``["experts.0.<sub>.weight", "experts.1.<sub>.weight", ...]``
        # in expert-id order. For routing flexibility, also accept the
        # case where hf_names is empty and the names come from
        # ``plan.role_descriptor['hf_names_by_expert_id']`` (a JSON-encoded
        # mapping, populated by the trainer's name-map publish path).
        names_by_id: dict[int, str] = {}
        if hf_names:
            # Position-keyed: caller already resolved ids → names. Trust it.
            for pos, name in enumerate(hf_names):
                names_by_id[pos] = name
        else:
            blob = (plan.role_descriptor or {}).get("hf_names_by_expert_id", "")
            if blob:
                try:
                    names_by_id = {int(k): str(v) for k, v in json.loads(blob).items()}
                except Exception as exc:  # noqa: BLE001
                    logger.warning("failed to parse hf_names_by_expert_id: %s", exc)

        for expert_id, tensor in assembled.items():
            name = names_by_id.get(expert_id)
            if name is None:
                logger.warning(
                    "no HF name for expert_id=%d; skipping", expert_id
                )
                continue
            yield name, tensor
        return

    raise ValueError(f"unsupported plan.role: {role}")


# ---------------------------------------------------------------------------
# High-level orchestrator — discover + plan + assemble + translate
# ---------------------------------------------------------------------------


@dataclass
class MegatronReceiverContext:
    """Per-receiver context carried across refit cycles. Built once at
    receiver init from the first cycle's source metadata + the inference
    model's HF state-dict shapes.
    """

    target_tp_layout: TargetTpLayout
    transformer_config: MegatronTransformerConfig
    # megatron_name → list of hf_names. Derived from the trainer's sidecar.
    hf_name_map: dict[str, list[str]]
    # Receive specs for the inference model's params, keyed by megatron_name.
    receive_specs: dict[str, ReceiveSpec]


def discover_megatron_context(
    candidates: list[V2SourceCandidate],
) -> tuple[MegatronTransformerConfig | None, dict[str, list[str]]]:
    """Inspect candidates for the first Megatron source and pull its
    config + name map. Returns ``(None, {})`` if no Megatron source.
    """
    for c in candidates:
        if c.megatron_meta is None:
            continue
        cfg, name_map = parse_megatron_sidecar(c)
        if cfg is not None or name_map:
            return cfg, name_map
    return None, {}


def run_refit_cycle(
    receiver: MxV2RefitReceiver,
    *,
    candidates: list[V2SourceCandidate],
    context: MegatronReceiverContext,
    pull: Callable[[MegatronSliceSource, torch.Tensor], None],
    device: torch.device | str = "cuda",
    pre_assembled_buffers: dict[str, torch.Tensor] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """One refit cycle: discover → plan → assemble → translate → yield.

    Args:
        receiver: a fully-initialized :class:`MxV2RefitReceiver`.
        candidates: discovered v2 sources for this cycle (typically
            ``receiver.discover_v2_sources(model_name=..., target_tp_layout=...)``
            has already been called by the orchestrator).
        context: receiver-side per-cycle context (target layout +
            Megatron config + name map + receive specs).
        pull: synchronous NIXL pull callback. Implementations should
            register the destination view with NIXL and complete the
            transfer before returning.
        device: where to pre-allocate destination tensors.
        pre_assembled_buffers: optional ``{megatron_name: tensor}`` map
            of pre-filled buffers. When set, the orchestrator skips
            ``assemble_into_destination`` for matched names and uses the
            pre-filled tensor directly. The matched-TP fast path
            registers buffers under each Megatron name with NIXL once
            and calls a bulk ``receive_weights`` per cycle; the translator
            then walks the filled buffers via this argument and
            ``pull`` becomes a no-op.

    Yields ``(hf_name, hf_tensor)`` ready for ``vllm.model.load_weights()``.
    """
    target_specs: dict[str, MegatronTensorSpec] = {}
    for megatron_name, rs in context.receive_specs.items():
        target_specs[megatron_name] = MegatronTensorSpec(
            role=rs.role,
            target_shape=rs.target_shape,
            target_dtype=rs.target_dtype,
            shard_axis=rs.shard_axis,
            pp_rank=rs.pp_rank,
            role_descriptor=dict(rs.role_descriptor or {}),
        )

    plans = receiver.pick_megatron_slice_plans(
        candidates,
        target_tp_layout=context.target_tp_layout,
        target_tensor_specs=target_specs,
    )

    pre = pre_assembled_buffers or {}
    for plan in plans:
        if not plan.sources and plan.tensor_name not in pre:
            logger.warning(
                "no sources for tensor %s (cycle skipped)", plan.tensor_name
            )
            continue
        rs = context.receive_specs[plan.tensor_name]
        if plan.tensor_name in pre and plan.assembly != "per_expert":
            # Pre-pulled buffer path: bypass per-source assembly. The buffer
            # already holds the receiver-side view of the assembled tensor.
            assembled: torch.Tensor | dict[int, torch.Tensor] = pre[plan.tensor_name]
        else:
            assembled = assemble_into_destination(plan, pull=pull, device=device)
        yield from translate_megatron_to_hf(
            plan,
            assembled,
            transformer_config=context.transformer_config,
            hf_names=rs.hf_names,
        )


__all__ = [
    "MegatronReceiverContext",
    "ReceiveSpec",
    "SIDECAR_HF_NAME_MAP_KEY",
    "SIDECAR_TRANSFORMER_CONFIG_KEY",
    "assemble_into_destination",
    "discover_megatron_context",
    "parse_megatron_sidecar",
    "run_refit_cycle",
    "translate_megatron_to_hf",
]
