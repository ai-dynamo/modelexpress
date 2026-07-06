# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelExpress backend for vLLM's native weight-transfer API.

Implements the RDMA backend (`RDMA[TODO]` in vLLM's
``WeightTransferEngine`` docstring) on top of ModelExpress + NIXL. Plugs
MX into vLLM's pluggable 4-phase weight-sync protocol
(https://docs.vllm.ai/en/latest/training/weight_transfer/):

  * ``init_transfer_engine``  -> create + initialize the MxV2 receiver.
  * ``receive_weights``       -> discover the trainer source, RDMA-pull its
    native (Megatron or DTensor) shards, translate to HF layout, then hand
    the ``[(name, tensor)]`` list to vLLM's ``load_weights`` callback.
  * ``trainer_send_weights``  -> (optional) publish via MxV2TrainingPublisher.
  * ``shutdown``.

The engine owns **transport + format conversion** (the parts vLLM's docs
say are framework-customized); the actual model load stays in vLLM's
``load_weights`` callback (which may itself be an MDL fast-loader). This
keeps MX transport-only and version-robust.

Feature toggles carried on ``MxUpdateInfo`` / env:
  * EP filter (``moe_expert_filter``): pull only this rank's experts.
  * Phase-0.5 host staging (``buffer_loc="host"`` / ``MX_MEGATRON_BUFFER_LOC``).
  * VMM-arena registration (``use_arena`` / ``MX_MEGATRON_ARENA``): 1 NIXL
    region instead of N (needs ``UCX_CUDA_COPY_REG_WHOLE_ALLOC=off``).
  * Byte-identity verify (``verify_gt_path`` / ``MX_VERIFY_BYTE_IDENTITY``).
"""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import torch

from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)

logger = logging.getLogger("modelexpress.engines.vllm.weight_transfer")

_MX_ENGINE_NAME = "mx"


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
class MxTrainerSendArgs:
    """Trainer-side send args. MX publishes via ``MxV2TrainingPublisher`` in
    the NeMo-RL Megatron policy worker (`stream_weights_via_mx`), so this is a
    thin descriptor kept for API symmetry with other backends."""

    model_name: str = ""
    version: int = 0
    mx_server_url: str = "modelexpress-server:8001"


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
    # Phase 0.5 / arena (fall back to env if unset here)
    buffer_loc: str | None = None
    use_arena: bool | None = None
    # verify
    verify_gt_path: str | None = None


class MxWeightTransferEngine(WeightTransferEngine[MxInitInfo, MxUpdateInfo]):
    """ModelExpress / NIXL RDMA backend for vLLM weight transfer."""

    init_info_cls = MxInitInfo
    update_info_cls = MxUpdateInfo

    def __init__(self, config, parallel_config) -> None:
        super().__init__(config, parallel_config)
        self._receiver = None
        self._init_info: MxInitInfo | None = None
        self._megatron_ctx = None
        self._buffers: dict[str, torch.Tensor] | None = None
        self._vocab_buffers: dict[str, torch.Tensor] | None = None
        self._arena = None  # kept alive if arena registration is used

    # ---- helpers ----
    def _buffer_device(self, upd: MxUpdateInfo) -> torch.device:
        loc = (upd.buffer_loc or os.environ.get("MX_MEGATRON_BUFFER_LOC", "device")).lower()
        if loc == "host":
            return torch.device("cpu")
        return torch.device(f"cuda:{self._init_info.device_id}")

    def _arena_enabled(self, upd: MxUpdateInfo) -> bool:
        if upd.use_arena is not None:
            return bool(upd.use_arena)
        return os.environ.get("MX_MEGATRON_ARENA", "0") == "1"

    # ---- phase 1: init ----
    def init_transfer_engine(self, init_info: MxInitInfo) -> None:
        from modelexpress import MxV2RefitReceiver
        from modelexpress import ucx_utils

        self._init_info = init_info
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
            "[mx-wt] initialized receiver rank=%d device=%d model=%s",
            rank, init_info.device_id, init_info.model_name,
        )

    # ---- phase 3: receive ----
    def receive_weights(
        self,
        update_info: MxUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        if self._receiver is None:
            raise RuntimeError("init_transfer_engine must be called first")
        from modelexpress.nemo_rl_v2 import ROLE_TRAINER

        model_name = self._init_info.model_name
        cands = self._receiver.discover_v2_sources(
            model_name=model_name,
            min_version=update_info.min_version,
            same_rank_only=self._init_info.same_rank_only,
            include_replicas=False,
        )
        if not cands:
            raise RuntimeError(f"[mx-wt] no MX source for {model_name} v>={update_info.min_version}")

        is_megatron = any(getattr(c, "megatron_meta", None) is not None for c in cands)
        if is_megatron:
            weights = self._receive_megatron(cands, update_info)
        else:
            weights = self._receive_dtensor(cands, update_info)

        if update_info.verify_gt_path or os.environ.get("MX_VERIFY_BYTE_IDENTITY"):
            self._verify(weights, update_info.verify_gt_path
                         or os.environ["MX_VERIFY_BYTE_IDENTITY"])

        load_weights(weights)

        if self._init_info.tree_scale_out:
            try:
                self._receiver.publish_self_as_source(
                    version=int(update_info.version), model_name=model_name
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[mx-wt] tree_scale_out republish failed: %s", exc)

    def _receive_megatron(
        self, cands: list[Any], upd: MxUpdateInfo
    ) -> list[tuple[str, torch.Tensor]]:
        from modelexpress.nemo_rl_v2 import TargetTpLayout, ROLE_MEGATRON_VOCAB_PARALLEL
        from modelexpress.megatron_translator import (
            ReceiveSpec, MegatronReceiverContext, discover_megatron_context,
            run_refit_cycle,
        )

        device = torch.device(f"cuda:{self._init_info.device_id}")
        sidecar_cfg, name_map = discover_megatron_context(cands)
        # Build receive specs from the source registry.
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
        # EP filter: prune expert specs to this rank's local set.
        if upd.moe_expert_filter and upd.num_experts:
            self._apply_ep_filter(specs, upd)

        ctx = MegatronReceiverContext(
            target_tp_layout=TargetTpLayout(tp_size=1, tp_rank=0),
            transformer_config=sidecar_cfg,
            hf_name_map=name_map,
            receive_specs=specs,
        )
        # Allocate + register receive buffers (device/host, per-tensor or arena).
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
            with use_arena(arena, device):
                _alloc()
            nixl.register_arena(arena, buffers)
            self._arena = arena
            logger.info("[mx-wt] arena-registered %d megatron buffers (1 region)", len(buffers))
        else:
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

    def _receive_dtensor(
        self, cands: list[Any], upd: MxUpdateInfo
    ) -> list[tuple[str, torch.Tensor]]:
        """HF/DTensor path: pull by name into scratch, yield (name, tensor)."""
        out: list[tuple[str, torch.Tensor]] = []
        for name, t in self._receiver._receiver.receive_weights_scratch(
            cands[0].ref, timeout_seconds=upd.timeout_seconds
        ):
            out.append((name, t))
        return out

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
        ok = sum(1 for k, v in weights
                 if k in gt and torch.equal(v.cpu(), gt[k].cpu()))
        logger.info("[mx-wt-verify] byte-identity: %d/%d match", ok, len(weights))

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | Any,
    ) -> None:
        raise NotImplementedError(
            "MX trainer send is driven by MxV2TrainingPublisher in the NeMo-RL "
            "Megatron policy worker (stream_weights_via_mx), not this hook."
        )

    def shutdown(self) -> None:
        self._receiver = None
        self._buffers = None
        self._arena = None


def register() -> None:
    """Register the MX backend with vLLM's WeightTransferEngineFactory."""
    from vllm.distributed.weight_transfer import WeightTransferEngineFactory
    WeightTransferEngineFactory.register_engine(_MX_ENGINE_NAME, MxWeightTransferEngine)


# Best-effort auto-register on import (safe: no-op if vLLM factory absent).
try:
    register()
except Exception:  # noqa: BLE001
    pass
