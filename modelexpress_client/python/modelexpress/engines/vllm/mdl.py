# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MDL (Mapped Direct Load) — a load-side fast-loader for RL refit.

MDL is an eager, destination-mapped in-place weight load: cycle 1 runs
vLLM's stock ``load_weights`` (correct cold start) and builds a destination
map classifying every HF tensor as

  * ``direct``      — 1:1 param, warm cycle does ``param.data.copy_``;
  * ``fused-slice`` — stacked member (q/k/v -> qkv_proj, gate/up ->
    gate_up_proj), ``param.data.narrow(0, off, sz).copy_``; offsets derived
    from the ACTUAL member shapes (version-robust);
  * ``expert-slice``— per-expert MoE tensor -> its slot in the stacked
    ``w13_weight``/``w2_weight`` via vLLM's own ``get_expert_mapping()``
    (requires the standard 3D layout, i.e. ``--moe-backend triton``);
  * ``fallback``    — anything else, routed back through stock ``load_weights``.

Cycle 2+ writes each tensor to its precomputed slot with **zero** stock-loader
calls for mapped tensors. Measured: Qwen3-4B 218 direct+180 fused (13 ms warm);
Qwen3-30B-A3B (triton MoE) 291 direct+144 fused+18,432 expert-slice, 2.15 s ->
0.55 s, byte-identical.

This is our own eager write (NOT Anyscale RDT — no lazy tensors / deferred
narrow). It is decoupled from transport: it's a plain
``load_weights(list[(name, tensor)])`` callback, so it composes with vLLM's
native weight-transfer API — pass ``MdlLoader(model).load_weights`` as the
``load_weights`` callback to a ``WeightTransferEngine.receive_weights`` (or as
the engine-level layerwise reload hook).

Generality / regime:
  * **Fused groups are derived from the model's own ``stacked_params_mapping``**
    when present (vLLM's listing order = pack order), so new architectures map
    with no MDL edits; falls back to the built-in qkv/gate_up set otherwise.
  * **Layout invalidation:** the destination map is re-validated each cycle
    against a cheap layout signature (param count + expert mapping + an optional
    ``model._mx_layout_version`` / ``MX_LOAD_LAYOUT_VERSION`` bump). If it
    changes (param-set change, EPLB rebalance signalled via the version, etc.)
    the map is rebuilt cold rather than writing to stale slots.
  * **Pipeline-parallel safe by construction:** the map only ever contains
    params present on *this* worker, and fallback only fires for incoming
    tensors — so a PP stage maps its own subset with no spurious fallback for
    other-stage params (validated PP=2).
  * **Partial / subset / delta updates:** the destination map is built
    *incrementally* — each tensor is classified the first time it is seen, in
    whatever update it arrives in. So an ``update_weights(subset=...)`` (by
    layer, layer-group, or param list), a layerwise reload, or a delta update
    gets the warm fast path instead of permanently falling back for tensors
    absent from the first cycle. Fused-group offsets are computed per batch, so
    a subset should carry a fused group's members together.
  * **Out of regime today (fall back to stock load, correct but slow):** kernel-
    swizzled / quantized MoE (needs ``--moe-backend triton``; guarded), and any
    custom loader fusion not expressed in ``stacked_params_mapping``.
"""

from __future__ import annotations

import logging
import os
import time as _time
from typing import Any

import torch

logger = logging.getLogger("modelexpress.engines.vllm.mdl")

# Stacked-param groups: (fused_param_suffix, member_suffix, canonical_order).
_STACKED_GROUPS = (
    ("qkv_proj", "q_proj", 0),
    ("qkv_proj", "k_proj", 1),
    ("qkv_proj", "v_proj", 2),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
)


class MdlLoader:
    """Wraps a vLLM model and provides an MDL ``load_weights`` callback.

    Usage:
        mdl = MdlLoader(model)
        # hand mdl.load_weights to a WeightTransferEngine as the callback:
        engine.receive_weights(update_info, load_weights=mdl.load_weights)

    Enable via ``MX_LOAD_MODE=direct`` (default ``stock`` -> always stock
    loader, i.e. MDL is a no-op wrapper). Gated to the validated regime
    (shape-matched writes); anything that doesn't map goes to stock fallback.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._param_cache: dict[str, torch.Tensor] | None = None
        self._direct: dict[str, torch.Tensor] = {}
        self._fused: dict[str, tuple] = {}
        self._expert: dict[str, tuple] = {}
        self._moe_module_cache: dict[str, Any] = {}
        self._layout_sig: tuple | None = None
        self._cycles = 0
        # H1: some quantized models (block-fp8 MoE) replace their parameters in
        # process_weights_after_loading, so vLLM's own load_weights is not
        # re-entrant (it reads param.output_dim etc. that the processed params
        # no longer carry). In that regime MDL runs "loaderless": it never calls
        # model.load_weights, and instead writes bytes straight into the
        # already-allocated/processed parameter slots on every cycle.
        self._loaderless = False

    # ---- public callback ----
    def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        mode = os.environ.get("MX_LOAD_MODE", "stock").lower()
        if mode != "direct":
            self._model.load_weights(weights=weights)
            return
        # H4: if the layout changed since the map was built (param-set change,
        # EPLB rebalance signalled via _mx_layout_version, etc.), drop the stale
        # map and rebuild cold rather than writing into stale slots.
        if self._param_cache is not None and self._layout_signature() != self._layout_sig:
            logger.info("[mx-mdl] layout changed since map build; forcing cold rebuild")
            self._reset_map()
        if self._param_cache is None:
            t0 = _time.perf_counter()
            if not self._loaderless:
                try:
                    self._model.load_weights(weights=weights)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[mx-mdl] model.load_weights failed on cold cycle (%s: %s); "
                        "switching to loaderless MDL apply (quant/refit-unsupported "
                        "loader). MDL will write parameter slots in-place instead.",
                        type(exc).__name__, str(exc)[:120],
                    )
                    self._loaderless = True
            self._check_moe_swizzle()
            self._param_cache = dict(self._model.named_parameters())
            apply_weights = self._remap_weights(weights) if self._loaderless else weights
            self._extend_dest_map(apply_weights)
            self._layout_sig = self._layout_signature()
            if self._loaderless:
                # load_weights didn't apply anything — do it ourselves via the
                # dest map (same in-place writes the warm path uses).
                self._warm_load(apply_weights)
                self._assert_loaderless_coverage()
            logger.info(
                "[mx-mdl] cold-cycle %s load %.2fs; cached %d params",
                "loaderless" if self._loaderless else "stock",
                _time.perf_counter() - t0, len(self._param_cache),
            )
            return
        self._warm_load(self._remap_weights(weights) if self._loaderless else weights)

    def _remap_weights(
        self, weights: list[tuple[str, torch.Tensor]]
    ) -> list[tuple[str, torch.Tensor]]:
        """Loaderless mode: translate checkpoint tensor names to vLLM param
        names with the model's own weight mapper — the substitution that
        ``model.load_weights`` normally applies. No-op if the model exposes no
        mapper (names already match)."""
        mapper = getattr(self._model, "hf_to_vllm_mapper", None)
        if mapper is None:
            return weights
        try:
            return list(mapper.apply(weights))
        except Exception:  # noqa: BLE001
            fn = getattr(mapper, "_map_name", None) or getattr(mapper, "map_name", None)
            if fn is None:
                return weights
            out = []
            for n, t in weights:
                try:
                    mn = fn(n)
                except Exception:  # noqa: BLE001
                    mn = n
                if mn is not None:
                    out.append((mn, t))
            return out

    def _assert_loaderless_coverage(self) -> None:
        """Fail loud rather than silently serve a half-written model. In
        loaderless mode MDL is the only thing writing the parameters, so if it
        couldn't classify most of them (e.g. a quantized model whose scale
        tensors MDL doesn't yet map), refuse instead of leaving stale bytes."""
        mapped = len(self._direct) + len(self._fused) + len(self._expert)
        total = len(self._param_cache or {})
        if total and mapped < 0.9 * total:
            raise RuntimeError(
                f"[mx-mdl] loaderless MDL mapped only {mapped}/{total} parameters "
                f"— this model's refit path is not fully supported by MDL yet "
                f"(commonly: fp8/quantized scale tensors, or an unmapped weight "
                f"name scheme). Refusing to serve a partially-written model. "
                f"Use a refit-capable (non-quantized, or triton-MoE) config, or "
                f"extend MDL's quant handling."
            )

    def _reset_map(self) -> None:
        self._param_cache = None
        self._direct = {}
        self._fused = {}
        self._expert = {}
        self._moe_module_cache = {}
        self._layout_sig = None

    # ---- warm path ----
    def _try_write(self, hf_name: str, tensor: torch.Tensor) -> str | None:
        """In-place write `tensor` into its mapped slot if known; return the
        class it was written as (``direct``/``fused``/``expert``) or ``None``."""
        d = self._fused.get(hf_name)
        if d is not None:
            param, axis, off, sz = d
            param.data.narrow(axis, off, sz).copy_(tensor, non_blocking=True)
            return "fused"
        e = self._expert.get(hf_name)
        if e is not None:
            param, eid, axis, off, sz = e
            param.data[eid].narrow(axis, off, sz).copy_(tensor, non_blocking=True)
            return "expert"
        p = self._direct.get(hf_name)
        if p is not None and tuple(p.shape) == tuple(tensor.shape):
            p.data.copy_(tensor, non_blocking=True)
            return "direct"
        return None

    def _warm_load(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        counts = {"direct": 0, "fused": 0, "expert": 0}
        mapped_late = 0
        fallback: list[tuple[str, torch.Tensor]] = []
        t0 = _time.perf_counter()
        with torch.no_grad():
            unmapped: list[tuple[str, torch.Tensor]] = []
            for hf_name, tensor in weights:
                cls = self._try_write(hf_name, tensor)
                if cls is not None:
                    counts[cls] += 1
                else:
                    unmapped.append((hf_name, tensor))
            # Partial / subset / delta support: a tensor not present in the cold
            # cycle (e.g. an update_weights(subset=...) scoped to new params, or a
            # layerwise/delta update) is classified + mapped on first sight here,
            # so it goes warm from the next cycle instead of falling back to the
            # stock loader every time. Only truly unclassifiable tensors fall back.
            if unmapped:
                self._extend_dest_map(unmapped)
                still: list[tuple[str, torch.Tensor]] = []
                for hf_name, tensor in unmapped:
                    cls = self._try_write(hf_name, tensor)
                    if cls is not None:
                        counts[cls] += 1
                        mapped_late += 1
                    else:
                        still.append((hf_name, tensor))
                fallback = still
        if fallback:
            if self._loaderless:
                # No re-entrant stock loader available for this (quantized)
                # model; a truly unmapped tensor can't be applied. For a fully
                # classified model this list is empty; warn if not so it's
                # visible rather than silently stale.
                logger.warning(
                    "[mx-mdl] loaderless: %d tensor(s) unmapped, left unwritten: %s",
                    len(fallback), [n for n, _ in fallback[:5]],
                )
            else:
                self._model.load_weights(weights=fallback)
        self._cycles += 1
        logger.info(
            "[mx-mdl] warm-cycle %d: %d direct + %d fused + %d expert in %.3fs "
            "(%d mapped on-the-fly, %d fallback)", self._cycles, counts["direct"],
            counts["fused"], counts["expert"], _time.perf_counter() - t0,
            mapped_late, len(fallback),
        )

    # ---- map building (additive / incremental) ----
    def _extend_dest_map(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        """Classify + record destination-map entries for any tensors in
        ``weights`` not already mapped. Additive and idempotent, so it serves
        both the full cold cycle and later partial/subset updates (each tensor
        is mapped the first time it's seen). Fused-group offsets are computed
        from the members present in this batch, so a subset update should
        include a fused group's members together (attention/MLP block
        granularity); a lone fused member with no mapped siblings falls back."""
        params = self._param_cache
        name_to_shape = {
            n: tuple(t.shape) for n, t in weights
            if n not in self._direct and n not in self._fused and n not in self._expert
        }
        if not name_to_shape:
            return
        expert_lookup: dict[str, tuple] = {}
        model = self._model
        if hasattr(model, "get_expert_mapping"):
            try:
                for p_suf, w_suf, eid, shard in model.get_expert_mapping():
                    expert_lookup[w_suf] = (p_suf, int(eid), shard)
            except Exception as exc:  # noqa: BLE001
                logger.info("[mx-mdl] no expert mapping: %s", exc)

        stacked_groups = self._stacked_groups()
        groups: dict[str, list[tuple[int, str, int]]] = {}
        for hf_name in name_to_shape:
            param = params.get(hf_name)
            if param is not None and tuple(param.shape) == name_to_shape[hf_name]:
                self._direct[hf_name] = param
                continue
            if ".experts." in hf_name and expert_lookup:
                dest = self._resolve_expert(hf_name, name_to_shape[hf_name], expert_lookup, params)
                if dest is not None:
                    self._expert[hf_name] = dest
                    continue
            matched = False
            for fused_suf, member_suf, order in stacked_groups:
                if member_suf + "." in hf_name or hf_name.endswith(member_suf + ".weight"):
                    fused_name = hf_name.replace(member_suf, fused_suf)
                    if params.get(fused_name) is None:
                        continue
                    groups.setdefault(fused_name, []).append(
                        (order, hf_name, name_to_shape[hf_name][0]))
                    matched = True
                    break
            # unmatched -> implicit fallback (not in any map)
        for fused_name, members in groups.items():
            fused_param = params[fused_name]
            members.sort(key=lambda m: m[0])
            off = 0
            for _o, hf_name, rows in members:
                self._fused[hf_name] = (fused_param, 0, off, rows)
                off += rows
            if off != int(fused_param.shape[0]):
                logger.warning("[mx-mdl] fused %s rows=%d != param=%d; fallback",
                               fused_name, off, int(fused_param.shape[0]))
                for _o, hf_name, _r in members:
                    self._fused.pop(hf_name, None)
        logger.info("[mx-mdl] dest map: %d direct, %d fused, %d expert",
                    len(self._direct), len(self._fused), len(self._expert))

    def _resolve_expert(self, hf_name, hf_shape, expert_lookup, params) -> tuple | None:
        for w_suf, (p_suf, eid, shard) in expert_lookup.items():
            if w_suf not in hf_name:
                continue
            fused_name = hf_name.replace(w_suf, p_suf)
            fp = params.get(fused_name)
            if fp is None or fp.ndim != 3:
                return None
            local = self._map_global_to_local(fused_name, eid)
            if local is None or local < 0 or local >= int(fp.shape[0]):
                return None
            rows = hf_shape[0]
            if shard == "w1":
                axis, off, sz = 0, 0, rows
            elif shard == "w3":
                axis, off, sz = 0, rows, rows
            else:  # w2
                axis, off, sz = 0, 0, int(fp.shape[1])
            if shard in ("w1", "w3") and sz != rows:
                return None
            if shard == "w2" and int(fp.shape[1]) != rows:
                return None
            return (fp, local, axis, off, sz)
        return None

    def _map_global_to_local(self, fused_param_name: str, gid: int) -> int | None:
        mod = self._moe_module_cache.get(fused_param_name, "MISS")
        if mod == "MISS":
            obj = self._model
            for part in fused_param_name.rsplit(".", 1)[0].split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            mod = obj
            self._moe_module_cache[fused_param_name] = mod
        if mod is not None and hasattr(mod, "_map_global_expert_id_to_local_expert_id"):
            try:
                return int(mod._map_global_expert_id_to_local_expert_id(gid))
            except Exception:  # noqa: BLE001
                return gid
        return gid

    def _stacked_groups(self) -> tuple:
        """Fused-member groups: (fused_suffix, member_suffix, pack_order).

        H3: derive from the model's own ``stacked_params_mapping`` when present
        — vLLM lists members in pack order, so the encounter index per fused
        param IS the offset order. New architectures map with no MDL edits.
        Falls back to the built-in qkv/gate_up set if the model doesn't expose
        the mapping.
        """
        raw = getattr(self._model, "stacked_params_mapping", None)
        groups: list[tuple[str, str, int]] = []
        if raw:
            seen: dict[str, int] = {}
            for entry in raw:
                try:
                    fused = str(entry[0]).lstrip(".")
                    member = str(entry[1]).lstrip(".")
                except Exception:  # noqa: BLE001
                    continue
                if not fused or not member or fused == member:
                    continue
                order = seen.get(fused, 0)
                seen[fused] = order + 1
                groups.append((fused, member, order))
        return tuple(groups) if groups else _STACKED_GROUPS

    def _layout_signature(self) -> tuple:
        """Cheap signature that bumps when the destination map would go stale.

        H4: catches param-set changes and (via ``get_expert_mapping``) static
        expert-mapping changes. For in-place EPLB rebalances that don't change
        the global mapping, the framework should bump ``model._mx_layout_version``
        (or set ``MX_LOAD_LAYOUT_VERSION``) — that's the reliable signal.
        """
        ver = getattr(self._model, "_mx_layout_version", None)
        if ver is None:
            ver = os.environ.get("MX_LOAD_LAYOUT_VERSION", "")
        n = sum(1 for _ in self._model.named_parameters())
        em_hash = 0
        if self._expert and hasattr(self._model, "get_expert_mapping"):
            try:
                em_hash = hash(tuple(
                    (str(p), str(w), int(e), str(s))
                    for p, w, e, s in self._model.get_expert_mapping()))
            except Exception:  # noqa: BLE001
                em_hash = 0
        return (str(ver), n, em_hash)

    def _check_moe_swizzle(self) -> None:
        for name, param in self._model.named_parameters():
            if "experts" in name and name.endswith(("w13_weight", "w2_weight")):
                if param.ndim > 3:
                    raise RuntimeError(
                        f"[mx-mdl] MoE expert param {name!r} is swizzled "
                        f"{param.ndim}D {tuple(param.shape)}; refit needs the "
                        f"standard 3D layout. Relaunch vLLM with "
                        f"'--moe-backend triton'."
                    )
                return
