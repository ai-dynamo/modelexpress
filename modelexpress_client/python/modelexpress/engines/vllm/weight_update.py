# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier-2: the vLLM-specific weight-update layer.

Owns generator-side **geometry discovery** (reading the live vLLM engine's
target layout, fused-param and expert mappings) and the **update/load process**
(receive + translate + place), exposing the base lifecycle:

  * ``initialize_weight_update_setup(init_info)``  — one-time setup.
  * ``was_weight_update_setup_initialized()``      — idempotency query.
  * ``start_weight_update(version)``               — open a refit cycle.
  * ``update_weights(update_info, load_weights, subset=None)`` — pull + convert
    + load, optionally scoped to a subset of parameters.
  * ``finish_weight_update(version)``              — close the cycle.

Layering:
  * **Tier 1** — :class:`modelexpress.MxV2RefitReceiver` is a *generic*,
    engine-agnostic receiver (source discovery, slice planning, NIXL pull,
    sidecar-driven translate). It knows nothing about vLLM and is used here.
  * **Tier 2 (this module)** — the vLLM-specific logic. All the target-layout
    discovery, buffer allocation/registration, EP filtering, and byte-verify
    live here, on top of tier 1.
  * **Tier 3** — :class:`modelexpress.engines.vllm.weight_transfer.MxWeightTransferEngine`
    is a thin adapter conforming to vLLM's ``WeightTransferEngine`` ABC over
    this tier. A framework may instead drive this tier directly via its own RPC.

The model *load* stays in vLLM's ``load_weights`` callback (which may be an MDL
fast-loader) — this tier hands it the ``[(name, tensor)]`` list.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from modelexpress.refit_timing import (
    RefitTimingRecorder,
    refit_span,
    use_refit_timing,
)
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)

logger = logging.getLogger("modelexpress.engines.vllm.weight_update")


# ---------------------------------------------------------------------------
# Info / config objects (shared with the tier-3 backend)
#
# These subclass vLLM's ABC info bases: this is the vLLM-specific layer, so the
# coupling belongs here (tier-1 stays vLLM-free). The tier-3 backend imports
# these directly.
# ---------------------------------------------------------------------------


@dataclass
class MxInitInfo(WeightTransferInitInfo):
    """One-time init for the MX receiver (per inference worker)."""

    mx_server_url: str = "modelexpress-server:8001"
    model_name: str = ""
    worker_rank: int = 0
    device_id: int = 0
    nic_pin: str = "auto"
    same_rank_only: bool = True
    tree_scale_out: bool = False


@dataclass
class MxUpdateInfo(WeightTransferUpdateInfo):
    """Per-refit update info + MX feature toggles."""

    version: int = 0
    min_version: int = 1
    timeout_seconds: float = 300.0
    # EP filter
    moe_expert_filter: bool = False
    ep_world_size: int = 1
    ep_rank: int = 0
    num_experts: int = 0
    expert_placement: str = "linear"
    # Receiver tensor-parallel identity. Checkpoint-format weights remain in
    # full HF shape for vLLM's model-specific loaders to shard correctly, but
    # the identity is required for slice planning, coverage, and direct-dest
    # optimizations.
    tp_world_size: int = 1
    tp_rank: int = 0
    # Phase 0.5 / arena (fall back to env if unset here)
    buffer_loc: str | None = None
    use_arena: bool | None = None
    # verify
    verify_gt_path: str | None = None
    # Partial / subset refit selectors (map onto WeightSubset). Any set => the
    # warm cycles pull and install only the matching tensors. The cold cycle is
    # always applied in full so the destination map covers every parameter.
    subset_param_names: list[str] | None = None
    subset_layers: list[int] | None = None
    subset_layer_groups: list[str] | None = None


@dataclass
class WeightSubset:
    """Scopes an ``update_weights`` call to part of the model.

    Selectors (a tensor is kept if it matches ANY provided selector; all-None =
    full update):
      * ``param_names`` — exact HF/Megatron tensor names.
      * ``layers`` — transformer layer indices; matches ``.layers.<n>.``.
      * ``layer_groups`` — name fragments for coarse groups, e.g.
        ``["self_attn", "mlp"]`` or ``["experts"]``.
    """

    param_names: list[str] | None = None
    layers: list[int] | None = None
    layer_groups: list[str] | None = None

    def is_empty(self) -> bool:
        return not (self.param_names or self.layers or self.layer_groups)

    def matches(self, *names: str) -> bool:
        """True if any of ``names`` satisfies any provided selector."""
        ns = [n for n in names if n]
        if self.param_names:
            want = set(self.param_names)
            if any(n in want for n in ns):
                return True
        if self.layers:
            tags = tuple(f".layers.{i}." for i in self.layers)
            if any(t in n for n in ns for t in tags):
                return True
        if self.layer_groups:
            if any(g in n for n in ns for g in self.layer_groups):
                return True
        return False


@dataclass
class MxTrainerSendArgs:
    """Trainer-side send args (kept for API symmetry). MX publishes via
    ``MxV2TrainingPublisher`` in the NeMo-RL Megatron policy worker
    (``stream_weights_via_mx``), not through this layer."""

    model_name: str = ""
    version: int = 0
    mx_server_url: str = "modelexpress-server:8001"


# ---------------------------------------------------------------------------
# Tier-2 weight-update layer
# ---------------------------------------------------------------------------


class MxVllmWeightUpdater:
    """vLLM-specific weight-update layer over the generic MX receiver."""

    def __init__(self) -> None:
        self._receiver = None
        self._init_info: MxInitInfo | None = None
        self._buffers: dict[str, torch.Tensor] | None = None
        self._arena = None  # kept alive if arena registration is used

    # ---- lifecycle: initialize ----
    def initialize_weight_update_setup(
        self, init_info: MxInitInfo, existing_receiver: Any | None = None
    ) -> None:
        """One-time setup: create (or adopt) the generic tier-1 receiver and pin NICs.

        vLLM geometry discovery (target layout, fused-param / expert mappings)
        is done lazily per update from the live model + source registry, so this
        stays cheap and idempotent.

        ``existing_receiver``: when a host (e.g. NeMo-RL's vLLM worker extension)
        already owns an :class:`MxV2RefitReceiver`, pass it in so we reuse the
        one NIXL agent instead of creating a second one in the same process."""
        from modelexpress import ucx_utils

        self._init_info = init_info
        if existing_receiver is not None:
            self._receiver = existing_receiver
            logger.info("[mx-wt] setup: adopted existing receiver (model=%s)",
                        init_info.model_name)
            return

        from modelexpress import MxV2RefitReceiver

        ucx_utils.apply_nic_pin_for_device(init_info.device_id)
        rank = init_info.worker_rank
        self._receiver = MxV2RefitReceiver(
            agent_name=f"mx-wt-{socket.gethostname()}-r{rank}",
            device_id=init_info.device_id,
            mx_server_url=init_info.mx_server_url,
            worker_rank=rank,
        )
        self._receiver.initialize(model_tensors=None)
        logger.info(
            "[mx-wt] setup: receiver rank=%d device=%d model=%s",
            rank, init_info.device_id, init_info.model_name,
        )

    def was_weight_update_setup_initialized(self) -> bool:
        return self._receiver is not None

    # ---- lifecycle: start (open cycle) ----
    def start_weight_update(self, version: int) -> None:
        """Open a refit cycle. Discovery + pull happen in ``update_weights``;
        this is the hook for pause/prepare and is a no-op today."""
        if self._receiver is None:
            raise RuntimeError("initialize_weight_update_setup must be called first")

    # ---- lifecycle: update (pull + convert + load) ----
    def update_weights(
        self,
        update_info: MxUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], Any],
        subset: WeightSubset | None = None,
        target_model: Any | None = None,
    ) -> None:
        if self._receiver is None:
            raise RuntimeError("initialize_weight_update_setup must be called first")
        cycle_start = time.perf_counter()
        timing = RefitTimingRecorder(
            backend="vllm",
            version=int(update_info.version),
            rank=int(self._init_info.worker_rank),
            tp_rank=int(update_info.tp_rank),
            tp_size=int(update_info.tp_world_size),
            ep_rank=int(update_info.ep_rank),
            ep_size=int(update_info.ep_world_size),
            cold=self._buffers is None,
        )

        with use_refit_timing(timing):
            try:
                model_name = self._init_info.model_name
                with timing.span("control_discovery"):
                    cands = self._receiver.discover_v2_sources(
                        model_name=model_name,
                        min_version=update_info.min_version,
                        same_rank_only=self._init_info.same_rank_only,
                        include_replicas=False,
                    )
                if not cands:
                    raise RuntimeError(
                        f"[mx-wt] no MX source for {model_name} "
                        f"v>={update_info.min_version}"
                    )

                with timing.span("source_preparation"):
                    is_megatron = any(
                        getattr(c, "megatron_meta", None) is not None for c in cands
                    )
                if is_megatron:
                    weights = self._receive_megatron(cands, update_info, subset)
                else:
                    timing.mark_not_applicable(
                        "transformation",
                        reason="DTensor source names already match loader input",
                    )
                    weights = self._receive_dtensor(cands, update_info, subset)
                receive_elapsed = time.perf_counter() - cycle_start

                if (
                    update_info.verify_gt_path
                    or os.environ.get("MX_VERIFY_BYTE_IDENTITY")
                ):
                    self._verify(
                        weights,
                        update_info.verify_gt_path
                        or os.environ["MX_VERIFY_BYTE_IDENTITY"],
                    )

                load_start = time.perf_counter()
                with timing.span(
                    "installation",
                    metadata={"incoming_tensors": len(weights)},
                ):
                    if (
                        not is_megatron
                        and update_info.tp_world_size > 1
                        and os.environ.get("MX_LOAD_MODE", "stock").lower()
                        == "stock"
                    ):
                        self._validate_stock_tp_weights(
                            weights,
                            update_info,
                            target_model
                            or getattr(load_weights, "__self__", None),
                        )
                    loaded = load_weights(weights)
                if torch.cuda.is_available():
                    with timing.span(
                        "post_install",
                        metadata={"operation": "cuda_synchronize"},
                    ):
                        torch.cuda.synchronize(self._init_info.device_id)
                load_elapsed = time.perf_counter() - load_start
                loaded_count = len(loaded) if isinstance(loaded, set) else None
                logger.info(
                    "[mx-wt] phase timing: receive_translate=%.3fs load=%.3fs "
                    "total=%.3fs incoming_hf=%d loaded_params=%s",
                    receive_elapsed,
                    load_elapsed,
                    time.perf_counter() - cycle_start,
                    len(weights),
                    loaded_count if loaded_count is not None else "unknown",
                )

                if self._init_info.tree_scale_out:
                    try:
                        with timing.span(
                            "post_install",
                            metadata={"operation": "tree_scale_out_publish"},
                        ):
                            self._receiver.publish_self_as_source(
                                version=int(update_info.version), model_name=model_name
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "[mx-wt] tree_scale_out republish failed: %s", exc
                        )
                elif not timing.has_measurements("post_install"):
                    timing.mark_not_applicable(
                        "post_install",
                        reason=(
                            "no CUDA device synchronization or tree scale-out "
                            "publishing was required"
                        ),
                    )
                timing.mark_not_applicable(
                    "rollout_readiness",
                    reason="vLLM engine owns pause/resume and KV-cache readiness",
                )
            finally:
                timing.emit(logger)

    # ---- lifecycle: finish (close cycle) ----
    def finish_weight_update(self, version: int) -> None:
        """Close the refit cycle. Hook for cache-invalidate/resume; the vLLM
        backend already handles KV-cache reset around this, so it's a no-op."""
        return

    def shutdown(self) -> None:
        self._receiver = None
        self._buffers = None
        self._arena = None

    # ---- helpers (vLLM-specific) ----
    def _buffer_device(self, upd: MxUpdateInfo) -> torch.device:
        loc = (upd.buffer_loc or os.environ.get("MX_MEGATRON_BUFFER_LOC", "device")).lower()
        if loc == "host":
            return torch.device("cpu")
        return torch.device(f"cuda:{self._init_info.device_id}")

    def _arena_enabled(self, upd: MxUpdateInfo) -> bool:
        if upd.use_arena is not None:
            return bool(upd.use_arena)
        return os.environ.get("MX_MEGATRON_ARENA", "0") == "1"

    @staticmethod
    def _apply_subset(specs: dict, subset: WeightSubset | None) -> None:
        """Prune receive specs to a subset (in place), by param name, layer index,
        or layer-group fragment. A spec is kept if its Megatron name or any of its
        HF names matches any provided selector."""
        if subset is None or subset.is_empty():
            return
        keep = {
            m: s for m, s in specs.items()
            if subset.matches(m, *(getattr(s, "hf_names", []) or []))
        }
        removed = len(specs) - len(keep)
        specs.clear()
        specs.update(keep)
        logger.info("[mx-wt] subset: kept %d specs, pruned %d", len(keep), removed)

    def _receive_megatron(
        self, cands: list[Any], upd: MxUpdateInfo, subset: WeightSubset | None = None
    ) -> list[tuple[str, torch.Tensor]]:
        from modelexpress.nemo_rl_v2 import TargetTpLayout, ROLE_MEGATRON_VOCAB_PARALLEL
        from modelexpress.megatron_translator import (
            ReceiveSpec, MegatronReceiverContext, discover_megatron_context,
            run_refit_cycle,
        )

        device = torch.device(f"cuda:{self._init_info.device_id}")
        sidecar_cfg, name_map = discover_megatron_context(cands)

        # EP-gather path: an EP-sharded trainer (sources spanning >1 ep_rank)
        # publishing to a non-EP / lower-EP rollout must gather experts across
        # sources and remap each source's local expert names to global. The
        # matched/TP-assembly path below (single ep_rank) is unchanged.
        ep_ranks = {c.megatron_meta.ep_rank for c in cands if c.megatron_meta is not None}
        if len(ep_ranks) > 1:
            return self._receive_megatron_ep_gather(
                cands, upd, subset, device, sidecar_cfg, name_map
            )

        specs: dict[str, ReceiveSpec] = {}
        cand = cands[0]
        for td in (cand.registry.get("tensors", []) if cand.registry else []):
            if not td.megatron_role or td.name.startswith("__mx_"):
                continue
            specs[td.name] = ReceiveSpec(
                megatron_name=td.name,
                hf_names=list(name_map.get(td.name, [td.name])),
                role=td.megatron_role,
                target_shape=tuple(int(s) for s in td.global_shape),
                target_dtype=td.dtype or "bfloat16",
                shard_axis=int(td.shard_axis),
                pp_rank=cand.megatron_meta.pp_rank if cand.megatron_meta else 0,
                role_descriptor=dict(td.megatron_extras or {}),
            )
        # Subset scoping (param-name level) before planning.
        self._apply_subset(specs, subset)
        # EP filter: prune expert specs to this rank's local set.
        if upd.moe_expert_filter and upd.num_experts:
            self._apply_ep_filter(specs, upd)

        ctx = MegatronReceiverContext(
            target_tp_layout=TargetTpLayout(tp_size=1, tp_rank=0),
            transformer_config=sidecar_cfg,
            hf_name_map=name_map,
            receive_specs=specs,
        )
        buf_dev = self._buffer_device(upd)
        dt_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        buffers: dict[str, torch.Tensor] = {}
        source_tp = next((c.megatron_meta.tp_size for c in cands
                          if c.megatron_meta and c.megatron_meta.tp_size > 0), 1)

        def _alloc():
            for spec in specs.values():
                if spec.role.startswith("expert_"):
                    continue
                shape = list(spec.target_shape)
                if spec.role == ROLE_MEGATRON_VOCAB_PARALLEL:
                    shape[int(spec.shard_axis)] *= int(source_tp)
                buffers[spec.megatron_name] = torch.empty(
                    shape, dtype=dt_map.get(spec.target_dtype, torch.bfloat16), device=buf_dev
                )

        nixl = self._receiver._receiver._nixl
        if self._arena_enabled(upd) and buf_dev.type == "cuda":
            from modelexpress.vmm import (
                VmmArena, CudaVmmBackend, use_arena, install_pluggable_allocator,
            )
            install_pluggable_allocator()
            arena = VmmArena(total_bytes=80 * (1024 ** 3), device=self._init_info.device_id,
                             backend=CudaVmmBackend(device=self._init_info.device_id))
            with refit_span(
                "setup_registration",
                metadata={"buffer_cache": "cold", "registration": "arena"},
            ):
                with use_arena(arena, device):
                    _alloc()
                nixl.register_arena(arena, buffers)
            self._arena = arena
            logger.info("[mx-wt] arena-registered %d megatron buffers (1 region)", len(buffers))
        else:
            with refit_span(
                "setup_registration",
                metadata={"buffer_cache": "cold", "registration": "per_tensor"},
            ):
                _alloc()
                nixl.register_tensors(buffers)
            logger.info("[mx-wt] per-tensor-registered %d megatron buffers on %s",
                        len(buffers), buf_dev.type)

        matched = next((c for c in cands if c.megatron_meta
                        and c.megatron_meta.tp_rank == 0), None)
        if matched is None:
            raise RuntimeError("[mx-wt] no matched-TP source (target_tp=1)")
        nixl.rebind_tensors(buffers)
        for _n, _t in self._receiver.receive_from(matched, timeout_seconds=upd.timeout_seconds):
            pass

        def _noop_pull(_src: Any, _dest: torch.Tensor) -> None:
            return

        weights = list(run_refit_cycle(
            self._receiver, candidates=cands, context=ctx,
            pull=_noop_pull, device=device, pre_assembled_buffers=dict(buffers),
        ))
        self._buffers = buffers
        return weights

    def _receive_megatron_ep_gather(
        self, cands: list[Any], upd: MxUpdateInfo, subset: WeightSubset | None,
        device: torch.device, sidecar_cfg: Any, name_map: dict,
    ) -> list[tuple[str, torch.Tensor]]:
        """Gather an EP-sharded Megatron trainer -> non-EP / lower-EP rollout.

        Each EP source names its grouped experts LOCALLY (``weight<L>`` ->
        ``experts.<L>.``), identical across ranks; the publisher advertises the
        GLOBAL id in ``extras['expert_id']`` (``local_expert_id`` keeps the local
        one). We process each source with its expert HF names rewritten
        ``experts.<L>. -> experts.<G>.`` so the gathered experts land in distinct,
        correct slots (vLLM's loader expects global expert names). Replicated/dense
        tensors are identical across sources and dedupe by name. The EP filter keeps
        only the rollout rank's wanted global experts. Delivers full HF weights
        (target TP1); vLLM slices to the rollout's TP at load.
        """
        import re
        from modelexpress.nemo_rl_v2 import TargetTpLayout
        from modelexpress.megatron_translator import (
            ReceiveSpec, MegatronReceiverContext, run_refit_cycle,
        )
        from modelexpress.rl_expert_layout import compute_local_expert_ids

        layout = TargetTpLayout(tp_size=1, tp_rank=0)
        wanted: set[int] | None = None
        if upd.moe_expert_filter and upd.num_experts:
            placement = (upd.expert_placement
                         if upd.expert_placement in ("linear", "round_robin") else "linear")
            wanted = set(compute_local_expert_ids(
                ep_rank=upd.ep_rank, ep_world_size=upd.ep_world_size,
                num_experts=int(upd.num_experts), placement=placement))

        _EXP = re.compile(r"(experts\.)(\d+)(\.)")

        def _globalize(hf: str, local_id: int, global_id: int) -> str:
            return _EXP.sub(
                lambda m: (f"{m.group(1)}{global_id}{m.group(3)}"
                           if int(m.group(2)) == local_id else m.group(0)),
                hf,
            )

        inner = self._receiver._receiver
        hf_results: dict[str, torch.Tensor] = {}
        expected_hf: set[str] = set()
        # Dedupe by ep_rank keeping the MOST RECENT source: the MX server can
        # retain stale registrations from prior (torn-down) trainers with the same
        # model name; connecting to a dead agent yields NIXL_ERR_REMOTE_DISCONNECT.
        by_ep: dict[int, Any] = {}
        for c in sorted(cands, key=lambda c: -(getattr(c, "updated_at", 0) or 0)):
            ep = c.megatron_meta.ep_rank if c.megatron_meta else 0
            by_ep.setdefault(ep, c)
        ordered = [by_ep[e] for e in sorted(by_ep)]
        logger.info("[mx-wt] EP-gather: %d candidates -> %d live EP sources (by ep_rank, newest)",
                    len(cands), len(ordered))
        inner.prune_scratch_remote_agents({c.ref.worker_id for c in ordered})
        # Non-expert / replicated tensors (attention, embeddings, norms, router
        # gate) are IDENTICAL across every EP source. Pull them from ONE source
        # only (the first) instead of pulling N copies over the wire and deduping
        # on keep. Expert tensors are distinct per source and always pulled.
        # NOTE: assumes non-expert coverage is complete on a single source (true
        # for PP=1; a PP-split trainer would need per-pp_rank non-expert handling).
        seen_nonexpert: set[str] = set()
        for cand in ordered:
            specs: dict[str, ReceiveSpec] = {}
            nm_c = dict(name_map)
            shapes: dict[str, tuple[int, ...]] = {}
            for td in (cand.registry.get("tensors", []) if cand.registry else []):
                if not td.megatron_role or td.name.startswith("__mx_"):
                    continue
                extras = dict(td.megatron_extras or {})
                lookup_name = (
                    td.name[len("module.") :]
                    if td.name.startswith("module.")
                    else td.name
                )
                is_expert = td.megatron_role.startswith("expert_")
                gid = int(extras.get("expert_id", -1)) if is_expert else -1
                lid = (
                    int(extras.get("local_expert_id", gid))
                    if is_expert
                    else -1
                )
                # Older publishers advertise local ids in both fields. Recover
                # the global id from the source's EP coordinate and the live
                # rollout's global expert count. Newer publishers already
                # provide a distinct global id and skip this compatibility path.
                source_ep_size = int(
                    cand.megatron_meta.ep_size
                    if cand.megatron_meta is not None
                    else len(ordered)
                )
                source_ep_rank = int(
                    cand.megatron_meta.ep_rank
                    if cand.megatron_meta is not None
                    else 0
                )
                if (
                    is_expert
                    and gid == lid
                    and source_ep_size > 1
                    and upd.num_experts > 0
                ):
                    local_count = int(upd.num_experts) // source_ep_size
                    gid = source_ep_rank * local_count + lid
                    extras["expert_id"] = str(gid)
                # Bridge introspection on an EP rank may key the sidecar by
                # global weight index (weight64) while the published local
                # parameter is named weight0. Prefer the global key derived
                # from descriptor extras, then fall back to the local key for
                # older sidecars.
                global_lookup = lookup_name
                if is_expert and gid >= 0 and lid >= 0:
                    global_lookup = re.sub(
                        rf"weight{lid}$",
                        f"weight{gid}",
                        lookup_name,
                    )
                hfs = list(
                    name_map.get(
                        global_lookup,
                        name_map.get(
                            lookup_name,
                            name_map.get(td.name, [td.name]),
                        ),
                    )
                )
                if is_expert:
                    if wanted is not None and gid not in wanted:
                        continue
                    hfs = [_globalize(h, lid, gid) for h in hfs]
                else:
                    # replicated/dense: only pull from the first source that has it
                    if td.name in seen_nonexpert:
                        continue
                    seen_nonexpert.add(td.name)
                nm_c[td.name] = hfs
                specs[td.name] = ReceiveSpec(
                    megatron_name=td.name, hf_names=hfs, role=td.megatron_role,
                    target_shape=tuple(int(s) for s in td.global_shape),
                    target_dtype=td.dtype or "bfloat16", shard_axis=int(td.shard_axis),
                    pp_rank=cand.megatron_meta.pp_rank if cand.megatron_meta else 0,
                    role_descriptor=extras,
                )
                expected_hf.update(hfs)
                shapes[td.name] = tuple(int(s) for s in td.global_shape)
            self._apply_subset(specs, subset)
            if not specs:
                continue
            ctx = MegatronReceiverContext(
                target_tp_layout=layout, transformer_config=sidecar_cfg,
                hf_name_map=nm_c, receive_specs=specs,
            )
            pre: dict[str, torch.Tensor] = {}
            # include_names prunes the RDMA pull to exactly this source's kept
            # tensors (experts owned by this rollout rank after the EP filter +
            # non-expert only from the first source), instead of the source's
            # full published set.
            for n, t in inner.receive_weights_scratch(
                cand.ref, timeout_seconds=upd.timeout_seconds,
                tensor_shapes=shapes, include_names=set(specs.keys()),
            ):
                if n in specs:
                    pre[n] = t
            for hf_name, hf_t in run_refit_cycle(
                self._receiver, candidates=[cand], context=ctx,
                pull=lambda _s, _d: None, device=device, pre_assembled_buffers=pre,
            ):
                if hf_name not in hf_results:
                    # Translation outputs are commonly views into the
                    # persistent scratch buffers. The next EP source reuses
                    # and overwrites those buffers, so retain an owned copy.
                    # CPU staging bounds rollout HBM and matches the proven
                    # legacy EP-gather path; TP-local/direct destinations can
                    # replace this staging in the optimized path.
                    hf_results[hf_name] = hf_t.detach().to("cpu")
        def _load_order(item: tuple[str, torch.Tensor]) -> tuple[int, int, str]:
            name = item[0]
            match = re.search(r"\.layers\.(\d+)\.", name)
            if match:
                return (1, int(match.group(1)), name)
            if "embed_tokens" in name:
                return (0, 0, name)
            return (2, 0, name)

        # vLLM's layerwise reload wrapper only releases temporary module
        # buffers when a parent layer is complete. EP-source insertion order
        # interleaves every layer and can retain ~model-size extra HBM; natural
        # layer order bounds the temporary working set to one layer.
        weights = sorted(hf_results.items(), key=_load_order)
        missing = sorted(expected_hf.difference(hf_results))
        if missing:
            sample = ", ".join(missing[:8])
            raise RuntimeError(
                "[mx-wt] incomplete Megatron EP-gather coverage: "
                f"missing {len(missing)}/{len(expected_hf)} expected HF tensors "
                f"(sample: {sample})"
            )
        logger.info(
            "[mx-wt] megatron EP-gather: %d/%d expected HF tensors from "
            "%d EP sources%s (target_tp=%d rank=%d)",
            len(weights), len(expected_hf), len(cands),
            f", EP-filtered to {len(wanted)} experts" if wanted is not None else "",
            upd.tp_world_size, upd.tp_rank,
        )
        return weights

    def _receive_dtensor(
        self, cands: list[Any], upd: MxUpdateInfo, subset: WeightSubset | None = None
    ) -> list[tuple[str, torch.Tensor]]:
        """HF/DTensor path: restore checkpoint shapes in scratch buffers.

        ``receive_weights_scratch`` allocates flat buffers from byte counts.
        vLLM's stock loaders require checkpoint-rank shapes so they can select
        their TP shard (notably VocabParallelEmbedding checks dimension 0).
        The v2 registry is authoritative for that reshape.
        """
        active = subset is not None and not subset.is_empty()
        registry = cands[0].registry or {}
        descriptors = {
            td.name: td
            for td in registry.get("tensors", [])
            if not td.name.startswith("__mx_")
        }
        if not descriptors:
            raise RuntimeError(
                "[mx-wt] DTensor/HF source has no v2 shape registry; refusing "
                "to pass flat scratch tensors to the stock vLLM loader"
            )

        tensor_shapes: dict[str, tuple[int, ...]] = {}
        for name, td in descriptors.items():
            shape = [int(dim) for dim in td.global_shape]
            if td.placement_kind == "SHARD":
                if td.local_shard_range is None:
                    raise RuntimeError(
                        f"[mx-wt] sharded tensor {name!r} has no local_shard_range"
                    )
                axis = int(td.shard_axis)
                if not 0 <= axis < len(shape):
                    raise RuntimeError(
                        f"[mx-wt] sharded tensor {name!r} has invalid axis {axis}"
                    )
                lo, hi = (int(v) for v in td.local_shard_range)
                if lo < 0 or hi <= lo or hi > shape[axis]:
                    raise RuntimeError(
                        f"[mx-wt] sharded tensor {name!r} has invalid range "
                        f"({lo}, {hi}) for global shape {tuple(shape)}"
                    )
                shape[axis] = hi - lo
            tensor_shapes[name] = tuple(shape)

        out: list[tuple[str, torch.Tensor]] = []
        for name, t in self._receiver._receiver.receive_weights_scratch(
            cands[0].ref,
            timeout_seconds=upd.timeout_seconds,
            tensor_shapes=tensor_shapes,
        ):
            if not active or subset.matches(name):
                expected = tensor_shapes.get(name)
                if expected is None:
                    raise RuntimeError(
                        f"[mx-wt] received tensor {name!r} is absent from the "
                        "v2 shape registry"
                    )
                if tuple(t.shape) != expected:
                    raise RuntimeError(
                        f"[mx-wt] tensor {name!r} shape {tuple(t.shape)} does "
                        f"not match registry shape {expected}"
                    )
                out.append((name, t))
        return out

    @staticmethod
    def _validate_stock_tp_weights(
        weights: list[tuple[str, torch.Tensor]],
        upd: MxUpdateInfo,
        model: Any | None,
    ) -> None:
        """Validate the global-HF contract consumed by stock vLLM TP loaders.

        The stock loader owns installation and name fusion. We only prove that
        exact-name vocab and ordinary TP parameters can produce this rank's
        local slice; passing an already-local shard would make stock slice it
        a second time, so that case fails loudly.
        """
        if model is None or not hasattr(model, "named_parameters"):
            logger.warning(
                "[mx-wt] cannot validate TP stock-load shapes without live "
                "vLLM model metadata"
            )
            return

        params = dict(model.named_parameters())
        vocab_modules: dict[str, Any] = {}
        param_modules: dict[str, Any] = {}
        for module_name, module in model.named_modules():
            prefix = f"{module_name}." if module_name else ""
            for param_name, _param in module.named_parameters(recurse=False):
                param_modules[prefix + param_name] = module
            if not hasattr(module, "org_vocab_size") or not hasattr(
                module, "shard_indices"
            ):
                continue
            for param_name, _param in module.named_parameters(recurse=False):
                vocab_modules[prefix + param_name] = module

        tp_size = int(upd.tp_world_size)
        tp_rank = int(upd.tp_rank)
        for name, tensor in weights:
            param = params.get(name)
            if param is None or tensor.ndim != param.ndim:
                continue

            local_shape = tuple(int(dim) for dim in param.shape)
            loaded_shape = tuple(int(dim) for dim in tensor.shape)
            vocab = vocab_modules.get(name)
            if vocab is not None:
                axis = int(getattr(param, "output_dim", 0))
                expected_vocab = int(vocab.org_vocab_size)
                if loaded_shape[axis] != expected_vocab:
                    raise RuntimeError(
                        f"[mx-wt] {name!r} has vocab extent "
                        f"{loaded_shape[axis]}, expected global HF vocab "
                        f"{expected_vocab}; registry/source layout is invalid"
                    )
                indices = vocab.shard_indices
                start = int(indices.org_vocab_start_index)
                end = int(indices.org_vocab_end_index)
                local = tensor.narrow(axis, start, end - start)
                if local.shape[axis] > local_shape[axis] or any(
                    int(local.shape[dim]) != local_shape[dim]
                    for dim in range(tensor.ndim)
                    if dim != axis
                ):
                    raise RuntimeError(
                        f"[mx-wt] {name!r} TP{tp_size} rank {tp_rank} vocab "
                        f"slice {tuple(local.shape)} cannot load into "
                        f"{local_shape}"
                    )
                continue

            if hasattr(param, "packed_dim") or hasattr(param, "pack_factor"):
                continue
            axes = []
            for attr in ("output_dim", "input_dim"):
                value = getattr(param, attr, None)
                if isinstance(value, int) and value not in axes:
                    axes.append(value)
            if not axes:
                continue
            matching = [
                axis
                for axis in axes
                if loaded_shape[axis] == local_shape[axis] * tp_size
                and all(
                    dim == axis or loaded_shape[dim] == local_shape[dim]
                    for dim in range(tensor.ndim)
                )
            ]
            if len(matching) == 1:
                axis = matching[0]
                local = tensor.narrow(
                    axis, tp_rank * local_shape[axis], local_shape[axis]
                )
                if tuple(local.shape) != local_shape:
                    raise RuntimeError(
                        f"[mx-wt] {name!r} produced invalid TP-local shape "
                        f"{tuple(local.shape)}, expected {local_shape}"
                    )
            elif loaded_shape == local_shape:
                module = param_modules.get(name)
                if module is not None and "Replicated" in type(module).__name__:
                    continue
                raise RuntimeError(
                    f"[mx-wt] {name!r} is already TP-local ({loaded_shape}); "
                    "stock vLLM expects a global HF tensor and would shard it "
                    "again. Use a replicated/global source or MX_LOAD_MODE=direct."
                )
            elif loaded_shape != local_shape:
                raise RuntimeError(
                    f"[mx-wt] {name!r} global shape {loaded_shape} is not a "
                    f"valid TP{tp_size} source for local shape {local_shape}"
                )

    def _apply_ep_filter(self, specs: dict, upd: MxUpdateInfo) -> None:
        from modelexpress.rl_expert_layout import compute_local_expert_ids
        placement = upd.expert_placement if upd.expert_placement in ("linear", "round_robin") else "linear"
        local = compute_local_expert_ids(
            ep_rank=upd.ep_rank, ep_world_size=upd.ep_world_size,
            num_experts=int(upd.num_experts), placement=placement,
        )
        local_str = ",".join(str(e) for e in local)
        n = 0
        for spec in specs.values():
            if spec.role.startswith("expert_"):
                rd = dict(spec.role_descriptor or {})
                rd["local_expert_ids"] = local_str
                spec.role_descriptor = rd
                n += 1
        logger.info("[mx-wt] EP filter: ep=%d/%d -> %d local experts on %d specs",
                    upd.ep_rank, upd.ep_world_size, len(local), n)

    def _verify(self, weights: list[tuple[str, torch.Tensor]], gt_path: str) -> None:
        gt = torch.load(gt_path, map_location="cpu", weights_only=False, mmap=True)
        gt = gt.get("hf_weights", gt) if isinstance(gt, dict) else gt
        got = dict(weights)
        missing = sorted(set(gt).difference(got))
        extra = sorted(set(got).difference(gt))
        drift: list[str] = []
        ok = 0
        for name in sorted(set(got).intersection(gt)):
            value = got[name].detach().cpu()
            expected = gt[name].detach().cpu()
            if value.shape == expected.shape and torch.equal(value, expected):
                ok += 1
            else:
                drift.append(name)
        logger.info(
            "[mx-wt-verify] byte-identity: %d ok / %d drift / %d missing / "
            "%d extra / %d GT",
            ok,
            len(drift),
            len(missing),
            len(extra),
            len(gt),
        )
        if missing or drift or extra:
            raise RuntimeError(
                "[mx-wt-verify] byte-identity failed: "
                f"drift={drift[:5]} missing={missing[:5]} extra={extra[:5]}"
            )
