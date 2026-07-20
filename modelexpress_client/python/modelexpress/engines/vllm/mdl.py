# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MDL (Mapped Direct Load) — a load-side fast-loader for RL refit.

MDL is an eager, destination-mapped in-place weight load. The first full load
uses vLLM's stock ``load_weights`` where it is safe and builds a destination
map classifying every inference-format tensor as

  * ``direct``      — 1:1 param, warm cycle does ``param.data.copy_``;
  * ``fused-slice`` — stacked member (q/k/v -> qkv_proj, gate/up ->
    gate_up_proj), ``param.data.narrow(0, off, sz).copy_``; offsets derived
    from the ACTUAL member shapes (version-robust);
  * ``expert-slice``— per-expert MoE tensor -> its slot in the stacked
    ``w13_weight``/``w2_weight`` via vLLM's own ``get_expert_mapping()``
    (requires the standard 3D layout, i.e. ``--moe-backend triton``);
  * ``fallback``    — anything else, routed back through stock ``load_weights``.

Later updates write mapped tensors directly into their precomputed slots,
avoiding repeated stock-loader dispatch.

MDL is an eager installer, not a lazy-tensor or deferred-narrow mechanism. It
is decoupled from transport: it is a plain
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
  * **FP8 loaderless refit:** standard Qwen3 MoE ``weight_scale[_inv]`` and
    optional activation/input scales map across direct, qkv/gate-up fused, and
    w13/w2 expert layouts. Loaderless updates fail closed unless every incoming
    tensor and every local FP8/scale parameter has an explicit destination.
  * **Out of regime today:** kernel-swizzled MoE (needs ``--moe-backend
    triton``; guarded), and custom loader fusion not expressed in
    ``stacked_params_mapping``.
"""

from __future__ import annotations

import logging
import os
import time as _time
from typing import Any

import torch

from modelexpress.refit_timing import current_refit_timing

logger = logging.getLogger("modelexpress.engines.vllm.mdl")

# Stacked-param groups: (fused_param_suffix, member_suffix, canonical_order).
_STACKED_GROUPS = (
    ("qkv_proj", "q_proj", 0),
    ("qkv_proj", "k_proj", 1),
    ("qkv_proj", "v_proj", 2),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
)

_SCALE_NAME_ALIASES = (
    ("weight_scale_inv", "weight_scale"),
    ("input_scale", "activation_scale"),
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
        self._fused: dict[str, torch.Tensor] = {}
        self._expert: dict[str, torch.Tensor] = {}
        self._moe_module_cache: dict[str, Any] = {}
        self._covered_params: set[str] = set()
        self._layout_sig: tuple | None = None
        self._cycles = 0
        self._tp_size = 1
        self._tp_rank = 0
        try:
            from vllm.distributed.parallel_state import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            self._tp_size = int(get_tensor_model_parallel_world_size())
            self._tp_rank = int(get_tensor_model_parallel_rank())
        except Exception:  # noqa: BLE001
            pass
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
        timing = current_refit_timing()
        if mode != "direct":
            if timing is not None:
                timing.set_cold(True)
            self._model.load_weights(weights=weights)
            return
        # H4: if the layout changed since the map was built (param-set change,
        # EPLB rebalance signalled via _mx_layout_version, etc.), drop the stale
        # map and rebuild cold rather than writing into stale slots.
        if (
            self._param_cache is not None
            and self._layout_signature() != self._layout_sig
        ):
            logger.info("[mx-mdl] layout changed since map build; forcing cold rebuild")
            self._reset_map()
        if timing is not None:
            timing.set_cold(self._param_cache is None)
        if self._param_cache is None:
            t0 = _time.perf_counter()
            # vLLM's stock FP8 load_weights is not re-entrant for some processed
            # quantized models and can block at TP>1. The guarded loaderless
            # path writes only through validated destinations.
            if not self._loaderless and self._should_force_fp8_loaderless():
                logger.info(
                    "[mx-mdl] FP8 params detected with tp_size=%d; using loaderless "
                    "MDL apply from the cold cycle (skipping non-re-entrant stock "
                    "FP8 load_weights).",
                    self._tp_size,
                )
                self._loaderless = True
            if not self._loaderless:
                self._model.load_weights(weights=weights)
            self._check_moe_swizzle()
            self._param_cache = dict(self._model.named_parameters())
            apply_weights = (
                self._remap_weights(weights) if self._loaderless else weights
            )
            self._extend_dest_map(apply_weights)
            self._layout_sig = self._layout_signature()
            if self._loaderless:
                applied = False
                try:
                    self._assert_loaderless_coverage(apply_weights)
                    self._warm_load(apply_weights)
                    applied = True
                finally:
                    if not applied:
                        self._reset_map()
            logger.info(
                "[mx-mdl] cold-cycle %s load %.2fs; cached %d params",
                "loaderless" if self._loaderless else "stock",
                _time.perf_counter() - t0,
                len(self._param_cache),
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

    def _assert_loaderless_coverage(
        self, weights: list[tuple[str, torch.Tensor]]
    ) -> None:
        """Require exact checkpoint and local FP8/scale destination coverage.

        A percentage threshold is unsafe for MoE: thousands of expert source
        tensors can hide a missing fused scale destination. Loaderless mode has
        no stock-loader safety net, so every incoming tensor and every local
        float8/scale parameter must have an explicit destination.
        """
        mapped_names = set(self._direct) | set(self._fused) | set(self._expert)
        missing_inputs = [name for name, _ in weights if name not in mapped_names]
        required_params = {
            name
            for name, param in (self._param_cache or {}).items()
            if param.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            or self._is_scale_name(name)
        }
        missing_params = sorted(required_params - self._covered_params)
        if missing_inputs or missing_params:
            raise RuntimeError(
                "[mx-mdl] loaderless coverage incomplete; refusing a "
                "partially-written model. "
                f"unmapped checkpoint tensors={missing_inputs[:10]}"
                f"{' ...' if len(missing_inputs) > 10 else ''}; "
                f"uncovered local FP8/scale parameters={missing_params[:10]}"
                f"{' ...' if len(missing_params) > 10 else ''}"
            )

    def _should_force_fp8_loaderless(self) -> bool:
        """Whether to use loaderless mode before the stock FP8 cold load.

        Default: force loaderless when the model carries FP8 parameters and
        ``tp_size > 1`` — the regime where vLLM's stock FP8 ``load_weights`` can
        hang instead of raising. Focused TP1/TP2 tests cover block-FP8 weights,
        W13 gate/up scale placement, and fail-closed behavior. Override with
        ``MX_FP8_LOADERLESS``: ``1`` forces it on for any TP size, while ``0``
        disables the guard and falls back to the stock loader.
        """
        override = os.environ.get("MX_FP8_LOADERLESS")
        if override is not None:
            override = override.strip()
            if override == "0":
                return False
            force_any_tp = override == "1"
        else:
            force_any_tp = False
        if self._tp_size <= 1 and not force_any_tp:
            return False
        try:
            for _name, param in self._model.named_parameters():
                if param.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    return True
        except Exception:  # noqa: BLE001
            return False
        return False

    def _reset_map(self) -> None:
        self._param_cache = None
        self._direct = {}
        self._fused = {}
        self._expert = {}
        self._moe_module_cache = {}
        self._covered_params = set()
        self._layout_sig = None

    # ---- warm path ----
    def _tp_local_tensor(
        self,
        tensor: torch.Tensor,
        expected_shape: tuple[int, ...],
    ) -> torch.Tensor | None:
        """Return this TP rank's contiguous shard when ``tensor`` is global."""
        if tuple(tensor.shape) == expected_shape:
            return tensor
        if self._tp_size <= 1 or tensor.ndim != len(expected_shape):
            return None
        candidates = []
        for axis, (loaded, expected) in enumerate(
            zip(tensor.shape, expected_shape, strict=True)
        ):
            if int(loaded) != int(expected) * self._tp_size:
                continue
            if all(
                dim == axis or int(tensor.shape[dim]) == int(expected_shape[dim])
                for dim in range(tensor.ndim)
            ):
                candidates.append(axis)
        if len(candidates) != 1:
            return None
        axis = candidates[0]
        local = int(expected_shape[axis])
        return tensor.narrow(axis, self._tp_rank * local, local).contiguous()

    def _tp_shape_compatible(
        self,
        loaded_shape: tuple[int, ...],
        expected_shape: tuple[int, ...],
    ) -> bool:
        if loaded_shape == expected_shape:
            return True
        if self._tp_size <= 1 or len(loaded_shape) != len(expected_shape):
            return False
        mismatches = [
            index
            for index, (loaded, expected) in enumerate(
                zip(loaded_shape, expected_shape, strict=True)
            )
            if int(loaded) != int(expected)
        ]
        return (
            len(mismatches) == 1
            and int(loaded_shape[mismatches[0]])
            == int(expected_shape[mismatches[0]]) * self._tp_size
        )

    def _resolve_write(
        self, hf_name: str, tensor: torch.Tensor
    ) -> tuple[str, torch.Tensor, torch.Tensor] | None:
        """Resolve one source tensor to a destination and local source view."""
        d = self._fused.get(hf_name)
        if d is not None:
            dest = d
            local = self._tp_local_tensor(tensor, tuple(dest.shape))
            if local is None:
                return None
            return "fused", dest, local
        e = self._expert.get(hf_name)
        if e is not None:
            dest = e
            local = self._tp_local_tensor(tensor, tuple(dest.shape))
            if local is None:
                return None
            return "expert", dest, local
        p = self._direct.get(hf_name)
        if p is not None:
            local = self._tp_local_tensor(tensor, tuple(p.shape))
            if local is not None:
                return "direct", p.data, local
        return None

    def _warm_load(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        counts = {"direct": 0, "fused": 0, "expert": 0}
        mapped_late = 0
        fallback: list[tuple[str, torch.Tensor]] = []
        writes: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        t0 = _time.perf_counter()
        unresolved: list[tuple[str, torch.Tensor]] = []
        for hf_name, tensor in weights:
            resolved = self._resolve_write(hf_name, tensor)
            if resolved is None:
                unresolved.append((hf_name, tensor))
            else:
                writes.append(resolved)

        # Partial / subset / delta support: classify tensors not present in the
        # cold cycle on first sight. Complete this pass before any in-place copy
        # so an unmapped loaderless batch cannot partially modify the model.
        if unresolved:
            self._extend_dest_map(unresolved)
            for hf_name, tensor in unresolved:
                resolved = self._resolve_write(hf_name, tensor)
                if resolved is None:
                    fallback.append((hf_name, tensor))
                else:
                    writes.append(resolved)
                    mapped_late += 1

        if fallback:
            if self._loaderless:
                raise RuntimeError(
                    "[mx-mdl] loaderless update contains unmapped tensor(s); "
                    "refusing to leave stale bytes: "
                    f"{[name for name, _ in fallback[:10]]}"
                )
        with torch.no_grad():
            if fallback:
                self._model.load_weights(weights=fallback)
            for kind, destination, source in writes:
                destination.copy_(source, non_blocking=True)
                counts[kind] += 1

        self._cycles += 1
        logger.info(
            "[mx-mdl] warm-cycle %d: %d direct + %d fused + %d expert in %.3fs "
            "(%d mapped on-the-fly, %d fallback)",
            self._cycles,
            counts["direct"],
            counts["fused"],
            counts["expert"],
            _time.perf_counter() - t0,
            mapped_late,
            len(fallback),
        )

    # ---- map building (additive / incremental) ----
    @staticmethod
    def _is_scale_name(name: str) -> bool:
        return name.endswith(
            (
                "weight_scale",
                "weight_scale_inv",
                "input_scale",
                "activation_scale",
            )
        )

    @staticmethod
    def _param_candidates(name: str) -> tuple[str, ...]:
        candidates = [name]
        for left, right in _SCALE_NAME_ALIASES:
            if left in name:
                candidates.append(name.replace(left, right))
            if right in name:
                candidates.append(name.replace(right, left))
        return tuple(dict.fromkeys(candidates))

    def _find_param(
        self, name: str, params: dict[str, torch.Tensor]
    ) -> tuple[str, torch.Tensor] | None:
        for candidate in self._param_candidates(name):
            param = params.get(candidate)
            if param is not None:
                return candidate, param
        return None

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
            n: tuple(t.shape)
            for n, t in weights
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
        groups: dict[str, list[tuple[int, str, tuple[int, ...]]]] = {}
        for hf_name in name_to_shape:
            direct = self._find_param(hf_name, params)
            if direct is not None and self._tp_shape_compatible(
                name_to_shape[hf_name],
                tuple(direct[1].shape),
            ):
                param_name, param = direct
                self._direct[hf_name] = param
                self._covered_params.add(param_name)
                continue
            if ".experts." in hf_name and expert_lookup:
                resolved = self._resolve_expert(
                    hf_name, name_to_shape[hf_name], expert_lookup, params
                )
                if resolved is not None:
                    dest, param_name = resolved
                    self._expert[hf_name] = dest
                    self._covered_params.add(param_name)
                    continue
            for fused_suf, member_suf, order in stacked_groups:
                if member_suf + "." in hf_name or hf_name.endswith(
                    member_suf + ".weight"
                ):
                    requested_name = hf_name.replace(member_suf, fused_suf)
                    fused = self._find_param(requested_name, params)
                    if fused is None:
                        continue
                    fused_name, _ = fused
                    groups.setdefault(fused_name, []).append(
                        (order, hf_name, name_to_shape[hf_name])
                    )
                    break
            # unmatched -> implicit fallback (not in any map)
        for fused_name, members in groups.items():
            fused_param = params[fused_name]
            members.sort(key=lambda m: m[0])
            destinations = self._fused_destinations(fused_param, members)
            if destinations is None:
                logger.warning(
                    "[mx-mdl] fused %s cannot represent member shapes %s in %s; "
                    "fallback",
                    fused_name,
                    [shape for _order, _name, shape in members],
                    tuple(fused_param.shape),
                )
                continue
            self._fused.update(destinations)
            self._covered_params.add(fused_name)
        logger.info(
            "[mx-mdl] dest map: %d direct, %d fused, %d expert",
            len(self._direct),
            len(self._fused),
            len(self._expert),
        )

    def _fused_destinations(
        self,
        fused_param: torch.Tensor,
        members: list[tuple[int, str, tuple[int, ...]]],
    ) -> dict[str, torch.Tensor] | None:
        """Return views for qkv/gate-up weights and their scale layouts."""
        data = fused_param.data
        if all(not shape for _order, _name, shape in members):
            if data.numel() == 1:
                return {name: data.reshape(()) for _order, name, _shape in members}
            if data.numel() == len(members):
                flat = data.reshape(-1)
                return {
                    name: flat[index].reshape(())
                    for index, (_order, name, _shape) in enumerate(members)
                }
            return None

        if not members or any(len(shape) != data.ndim for _o, _n, shape in members):
            return None
        candidates = []
        for axis in range(data.ndim):
            if not all(
                all(
                    dim == axis or int(shape[dim]) == int(data.shape[dim])
                    for dim in range(data.ndim)
                )
                for _order, _name, shape in members
            ):
                continue
            total = sum(int(shape[axis]) for _order, _name, shape in members)
            if total == int(data.shape[axis]):
                candidates.append((axis, 1))
            elif (
                self._tp_size > 1
                and total == int(data.shape[axis]) * self._tp_size
                and all(
                    int(shape[axis]) % self._tp_size == 0 for _o, _n, shape in members
                )
            ):
                candidates.append((axis, self._tp_size))
        if not candidates:
            return None
        axis, divisor = candidates[0]
        offset = 0
        out = {}
        for _order, name, shape in members:
            size = int(shape[axis]) // divisor
            out[name] = data.narrow(axis, offset, size)
            offset += size
        return out

    def _resolve_expert(
        self, hf_name, hf_shape, expert_lookup, params
    ) -> tuple[torch.Tensor, str] | None:
        for w_suf, (p_suf, eid, shard) in expert_lookup.items():
            if w_suf not in hf_name:
                continue
            requested_name = hf_name.replace(w_suf, p_suf)
            fused = self._find_param(requested_name, params)
            if fused is None:
                return None
            fused_name, fp = fused
            if fp.ndim < 1:
                return None
            local = self._map_global_to_local(fused_name, eid)
            if local is None or local < 0 or local >= int(fp.shape[0]):
                return None
            expert_data = fp.data[local]
            if shard == "w2":
                if self._tp_shape_compatible(hf_shape, tuple(expert_data.shape)):
                    return expert_data, fused_name
                return None
            if not hf_shape:
                if expert_data.numel() == 1:
                    return expert_data.reshape(()), fused_name
                if expert_data.numel() == 2:
                    index = 0 if shard == "w1" else 1
                    return expert_data.reshape(-1)[index].reshape(()), fused_name
                return None
            if len(hf_shape) != expert_data.ndim:
                return None
            # w13 stacks gate (w1) and up (w3) in its first dimension. At TP2,
            # the local stacked destination can have the same shape as one
            # global HF member (e.g. 768 global gate rows == 2 * 384 local
            # rows). Treating that as an exact match maps both gate and up to
            # the entire w13 slot. Always select the member half first, then
            # let _resolve_write/_tp_local_tensor choose this TP rank's source
            # shard. The same collision occurs for block-FP8 scale rows.
            if shard in ("w1", "w3") and int(expert_data.shape[0]) % 2 == 0:
                local_size = int(expert_data.shape[0]) // 2
                offset = 0 if shard == "w1" else local_size
                destination = expert_data.narrow(0, offset, local_size)
                if self._tp_shape_compatible(hf_shape, tuple(destination.shape)):
                    return destination, fused_name
            for axis in range(expert_data.ndim):
                if not all(
                    dim == axis or int(hf_shape[dim]) == int(expert_data.shape[dim])
                    for dim in range(expert_data.ndim)
                ):
                    continue
                source = int(hf_shape[axis])
                dest = int(expert_data.shape[axis])
                if dest == source * 2:
                    local_size = source
                elif self._tp_size > 1 and dest * self._tp_size == source * 2:
                    local_size = dest // 2
                else:
                    continue
                offset = 0 if shard == "w1" else local_size
                return expert_data.narrow(axis, offset, local_size), fused_name
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
                return None
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
        """Describe parameter storage and expert mapping used by destinations."""
        ver = getattr(self._model, "_mx_layout_version", None)
        if ver is None:
            ver = os.environ.get("MX_LOAD_LAYOUT_VERSION", "")
        parameter_layout = tuple(
            (
                name,
                id(param),
                int(param.data_ptr()),
                tuple(param.shape),
                tuple(param.stride()),
                int(param.storage_offset()),
                str(param.dtype),
                str(param.device),
            )
            for name, param in self._model.named_parameters()
        )
        expert_mapping: tuple = ()
        get_expert_mapping = getattr(self._model, "get_expert_mapping", None)
        if callable(get_expert_mapping):
            try:
                expert_mapping = tuple(
                    (str(param), str(weight), int(expert), str(shard))
                    for param, weight, expert, shard in get_expert_mapping()
                )
            except Exception as exc:  # noqa: BLE001
                # Some vLLM model adapters expose the method before their
                # expert mapping is available. Parameter storage and the
                # explicit layout version still participate in the signature;
                # changing from unavailable to available also invalidates it.
                expert_mapping = (("<unavailable>", type(exc).__qualname__),)
        return str(ver), parameter_layout, expert_mapping

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
