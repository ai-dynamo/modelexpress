# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V2 NemoRL helpers built on top of MxTrainingPublisher / MxRefitReceiver.

This module implements the design from
``pensieve/RL/NemoRL/04_design_v2_moe_rank_to_rank.md`` as a Python-only
shim that doesn't require proto/Rust changes. The shim:

1. Encodes per-tensor shape + placement + expert metadata into
   ``SourceIdentity.extra_parameters`` (JSON document under key
   ``shape_registry``). See :mod:`modelexpress.shape_descriptors`.

2. Defaults to **same-rank-only transfers** (lesson from PrimeRL on
   GB200; cross-subnet full-mesh fails on multi-NIC fabrics). Each
   inference rank N pulls only from trainer rank N (or another
   inference rank N that's already received via tree fan-out).

3. Implements **tree fan-out / pipeline replication** by having
   inference receivers republish themselves with NIXL after
   receiving — subsequent receivers can pull from them. Source
   selection prefers the trainer first, then any peer that's
   ahead of us at the same ``worker_rank``.

4. Encodes **owned / needed expert IDs** into ``extra_parameters``
   so a receiver in EP mode can skip non-owned experts entirely.

5. Wraps :class:`HeartbeatThread` so v2 publishers / receivers come
   with liveness signaling out of the box. The MX-side reaper can
   correctly distinguish quiet-but-alive workers from dead ones.

This is a **prototype-grade** shim: the eventual production answer is
new RPCs (PickSource, GetShapeRegistry, SetDirtyExperts, ...) on the
MX server, with full TopologyScheduler logic in Rust. See
``pensieve/RL/NemoRL/05_mx_helpers_needed.md`` for the proto migration.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Iterator

import torch

from . import p2p_pb2
from .metadata.heartbeat import HeartbeatThread
from .refit_receiver import MxRefitReceiver, SourceRef
from .shape_descriptors import (
    PLACEMENT_SHARD,
    TensorDescriptorV2,
    decode_expert_set,
    decode_registry,
    describe_tensor,
    encode_expert_set,
    encode_registry,
)
from .training_publisher import MxTrainingPublisher

logger = logging.getLogger("modelexpress.nemo_rl_v2")


# Role string written into ``extra_parameters["role"]``. Matches the
# convention adopted by PR #2389. Receivers filter on it to disambiguate.
ROLE_TRAINER = "trainer"
ROLE_INFERENCE = "inference"
ROLE_INFERENCE_REPLICA = "inference_replica"


# Synthetic tensor descriptor used as a v2 metadata sidecar. The current
# Rust MX server drops most string fields (agent_name, extra_parameters,
# metadata_endpoint, etc.) when echoing a WorkerMetadata back via
# GetMetadata, but it preserves tensor descriptors. So we abuse a
# zero-size, magic-named TensorDescriptor as the transport: the JSON v2
# payload goes in the ``dtype`` field, which is a freeform proto3 string
# the server stores verbatim. Receivers look for this marker and pull
# v2 fields from it.
_V2_SIDECAR_NAME = "__mx_v2_meta__"


# Megatron role enum. Each Megatron parameter classifies into exactly one of
# these on the publisher side; planner + assembler dispatch on the role string.
# See temp/NemoRL_Megatron_MX_Design.md §3 for the full semantics table.
ROLE_MEGATRON_QKV_COLUMN = "qkv_column"
ROLE_MEGATRON_GATED_MLP_COLUMN = "gated_mlp_column"
ROLE_MEGATRON_COLUMN = "column"
ROLE_MEGATRON_ROW = "row"
ROLE_MEGATRON_VOCAB_PARALLEL = "vocab_parallel"
ROLE_MEGATRON_REPLICATED = "replicated"
ROLE_MEGATRON_EXPERT_COLUMN = "expert_column"
ROLE_MEGATRON_EXPERT_ROW = "expert_row"

_MEGATRON_ROLE_SET = frozenset({
    ROLE_MEGATRON_QKV_COLUMN,
    ROLE_MEGATRON_GATED_MLP_COLUMN,
    ROLE_MEGATRON_COLUMN,
    ROLE_MEGATRON_ROW,
    ROLE_MEGATRON_VOCAB_PARALLEL,
    ROLE_MEGATRON_REPLICATED,
    ROLE_MEGATRON_EXPERT_COLUMN,
    ROLE_MEGATRON_EXPERT_ROW,
})


def _extract_megatron_meta(extra: dict[str, str]) -> "MegatronSourceMeta | None":
    """Pull source-level Megatron rank metadata out of extra_parameters.

    Detection is by ``publisher_kind == "megatron"`` (set on the source
    identity) and / or presence of ``tp_size`` + ``tp_rank`` keys.
    Per-tensor role lives in the source's shape_registry, not here.
    Returns ``None`` for non-Megatron sources (DTensor, PrimeRL).
    """
    if extra.get("publisher_kind") != "megatron" and not (
        "tp_size" in extra and "tp_rank" in extra
    ):
        return None

    def _i(key: str, default: int = 0) -> int:
        try:
            return int(extra.get(key, str(default)))
        except (TypeError, ValueError):
            return default

    return MegatronSourceMeta(
        tp_rank=_i("tp_rank"),
        tp_size=_i("tp_size", 1),
        pp_rank=_i("pp_rank"),
        pp_size=_i("pp_size", 1),
        ep_rank=_i("ep_rank"),
        ep_size=_i("ep_size", 1),
    )


# Inference-side desired layout. Constructed by the receiver and passed into
# the slice planner so it can decide matched-TP / mixed-TP plans.
@dataclass(frozen=True)
class TargetTpLayout:
    """Inference-side parallelism the receiver wants to assemble into.

    Set ``ep_size`` > 1 for MoE deployments where inference EP differs from
    trainer EP. PP on the inference side is rare; left at 1 for now.
    """

    tp_size: int = 1
    ep_size: int = 1
    tp_rank: int = 0  # this receiver's TP rank within the inference mesh
    ep_rank: int = 0  # this receiver's EP rank within the inference mesh


# Trainer world layout descriptor. Receivers can sanity-check that the
# layout they expect matches what the trainer actually published.
@dataclass(frozen=True)
class TrainerWorldLayout:
    """Compact descriptor for a trainer's parallelism layout."""

    fsdp_world_size: int = 1
    tp_world_size: int = 1
    pp_world_size: int = 1
    ep_world_size: int = 1

    def encode(self) -> str:
        return (
            f"fsdp:{self.fsdp_world_size},tp:{self.tp_world_size},"
            f"pp:{self.pp_world_size},ep:{self.ep_world_size}"
        )

    @classmethod
    def decode(cls, s: str) -> "TrainerWorldLayout":
        kv = {p.split(":")[0]: int(p.split(":")[1]) for p in s.split(",") if ":" in p}
        return cls(
            fsdp_world_size=kv.get("fsdp", 1),
            tp_world_size=kv.get("tp", 1),
            pp_world_size=kv.get("pp", 1),
            ep_world_size=kv.get("ep", 1),
        )


class MxV2TrainingPublisher:
    """v2 trainer-side publisher.

    Wraps :class:`MxTrainingPublisher` and adds:

    - **Shape registry**: per-tensor placement + expert info, JSON-encoded
      and stashed in ``extra_parameters["shape_registry"]``.
    - **Rank-to-rank semantics**: every rank publishes its OWN local shard;
      no allgather, no bucket pack.
    - **Heartbeat**: started automatically by :meth:`mark_ready`.
    - **MoE expert metadata**: per-tensor ``owned_expert_ids`` propagated
      to the receiver via the registry.

    Args:
        agent_name: Unique NIXL agent name (e.g. ``"nemo-rl-trainer-r3"``).
        device_id: CUDA device index.
        mx_server_url: MX gRPC URL.
        worker_rank: Global rank within the trainer's parallelism group.
            For FSDP-only this is the FSDP rank; for FSDP+TP+EP it should
            map to the receiver's rank index in the same coord system.
        world_layout: Total parallelism layout — receivers use it to
            sanity-check expected shape.
        listen_port: Optional NIXL listen port.
        heartbeat: Whether to start a background heartbeat after
            ``mark_ready``. Default True.
    """

    def __init__(
        self,
        *,
        agent_name: str,
        device_id: int,
        mx_server_url: str,
        worker_rank: int,
        world_layout: TrainerWorldLayout,
        listen_port: int | None = None,
        heartbeat: bool = True,
    ):
        self._publisher = MxTrainingPublisher(
            agent_name=agent_name,
            device_id=device_id,
            mx_server_url=mx_server_url,
            listen_port=listen_port,
        )
        self._worker_rank = worker_rank
        self._world_layout = world_layout
        self._heartbeat_enabled = heartbeat
        self._heartbeat: HeartbeatThread | None = None

        self._registry: list[TensorDescriptorV2] = []
        self._registered_tensors: dict[str, torch.Tensor] = {}
        self._initialized = False

        # Megatron rank position within the TP × PP × EP mesh. Receivers
        # use these to route per-rank pulls. Set via
        # :meth:`set_megatron_mesh_position` before publish() when this
        # publisher is being used by a Megatron-Core trainer; defaults
        # (0, 0, 0) are correct for the FSDP / DTensor case where the
        # sub-mesh axes are all size 1.
        self._megatron_tp_rank: int = 0
        self._megatron_pp_rank: int = 0
        self._megatron_ep_rank: int = 0

        # Optional Megatron-Bridge-derived sidecar payload (transformer
        # config + Megatron→HF name map). Set once at trainer init via
        # :meth:`set_megatron_sidecar`; merged into the v2 sidecar JSON
        # at every :meth:`publish` so receivers can do role-aware
        # assembly without importing Bridge.
        self._megatron_sidecar: dict[str, Any] = {}

    @property
    def worker_rank(self) -> int:
        return self._worker_rank

    @property
    def mx_source_id(self) -> str | None:
        return self._publisher.mx_source_id

    @property
    def worker_id(self) -> str:
        return self._publisher.worker_id

    def initialize(self, *, model_name: str, dtype: str = "bfloat16") -> None:
        """Initialize the underlying NIXL agent + MX gRPC client."""
        self._publisher.initialize(
            model_name=model_name,
            tensor_parallel_size=self._world_layout.tp_world_size,
            pipeline_parallel_size=self._world_layout.pp_world_size,
            expert_parallel_size=self._world_layout.ep_world_size,
            dtype=dtype,
            training_framework="nemo_rl",
        )
        self._initialized = True

    def set_megatron_sidecar(self, sidecar: dict[str, Any]) -> None:
        """Set the Megatron-Bridge-derived sidecar payload.

        ``sidecar`` is a dict with the keys
        ``"megatron_transformer_config"`` and ``"megatron_hf_name_map"``
        (per :mod:`modelexpress.megatron_translator`). The publisher
        merges these into the v2 sidecar JSON at every :meth:`publish`,
        so receivers can do role-aware assembly without importing
        Megatron-Bridge themselves.

        Call once at trainer init after Bridge introspection. Empty
        dict (the default) is a no-op — receivers fall back to deriving
        head config / name map independently.
        """
        self._megatron_sidecar = dict(sidecar) if sidecar else {}

    def set_megatron_mesh_position(
        self,
        *,
        tp_rank: int,
        pp_rank: int = 0,
        ep_rank: int = 0,
    ) -> None:
        """Set this publisher's rank position in the Megatron TP × PP × EP mesh.

        Call before :meth:`publish` if any tensor in the in-flight registry
        has ``megatron_role`` set. The rank-position metadata is stamped
        onto the source identity so receivers can route per-rank pulls
        (the per-tensor role drives assembly; rank position drives source
        selection). Default (0, 0, 0) is correct for FSDP / DTensor
        publishers where these mesh axes have size 1.
        """
        self._megatron_tp_rank = int(tp_rank)
        self._megatron_pp_rank = int(pp_rank)
        self._megatron_ep_rank = int(ep_rank)
        logger.info(
            "MxV2TrainingPublisher initialized: rank=%d layout=%s",
            self._worker_rank,
            self._world_layout.encode(),
        )

    def add_tensor(
        self,
        *,
        name: str,
        tensor: torch.Tensor,
        is_expert: bool = False,
        expert_axis: int = 0,
        owned_expert_ids: tuple[int, ...] | set[int] | list[int] = (),
        megatron_role: str | None = None,
        megatron_extras: dict[str, str] | None = None,
    ) -> None:
        """Register a tensor for publication.

        Each call appends the tensor and its descriptor to the in-flight
        registry. Call :meth:`publish` once all tensors are added; that
        single publish call registers everything with NIXL (once) and
        emits one ``WorkerMetadata`` row.

        Args:
            name: tensor's qualified state-dict name.
            tensor: GPU tensor to publish. May be a DTensor or plain
                tensor. **Must NOT be a materialized full tensor** —
                pass ``tensor.to_local()`` for DTensors. The whole
                point of v2 is to avoid the allgather.
            is_expert: whether the tensor's leading axis is the MoE
                expert axis (used for expert filtering).
            expert_axis: axis index for the expert dimension.
            owned_expert_ids: which expert IDs this rank holds. Pass
                only when ``is_expert == True``.
            megatron_role: per-tensor Megatron role string (one of
                ``ROLE_MEGATRON_*``). Stashed on the per-tensor
                ``TensorDescriptorV2`` in the registry blob — the
                receiver-side slice planner reads it from there. Leave
                ``None`` for DTensor / PrimeRL publishes.
            megatron_extras: per-tensor Megatron descriptor extras
                (head counts for ``qkv_column``, ``gated_mlp_order``
                for ``gated_mlp_column``, etc.). Same lifetime as
                ``megatron_role``.
        """
        if not self._initialized:
            raise RuntimeError("call initialize() before add_tensor()")
        if not tensor.is_cuda:
            raise RuntimeError(
                f"tensor {name!r} is not on CUDA; v2 publish requires GPU residency"
            )

        descriptor = describe_tensor(
            name=name,
            tensor=tensor,
            rank=self._worker_rank,
            fsdp_world_size=self._world_layout.fsdp_world_size,
            is_expert=is_expert,
            expert_axis=expert_axis,
            owned_expert_ids=tuple(sorted(owned_expert_ids)),
        )
        if megatron_role is not None:
            descriptor.megatron_role = megatron_role
        if megatron_extras:
            descriptor.megatron_extras = dict(megatron_extras)
        self._registry.append(descriptor)
        # Use a key that's unique per descriptor (including any potential
        # name collisions from layer publishing). For v2 we publish all
        # tensors at once, so the name is sufficient.
        self._registered_tensors[name] = tensor

    def publish(self, *, version: int) -> str:
        """Publish all added tensors as one ``WorkerMetadata`` row.

        Returns the ``mx_source_id`` (16-hex hash) assigned by the server.
        """
        if not self._initialized:
            raise RuntimeError("call initialize() before publish()")
        if not self._registered_tensors:
            raise RuntimeError(
                "no tensors added; call add_tensor() before publish()"
            )

        # Build the registry blob; merge in the Megatron sidecar (if any)
        # via encode_registry's extras so the receiver sees
        # transformer_config + name_map at the top level of
        # candidate.registry. parse_megatron_sidecar consumes them from
        # there.
        registry_extras: dict[str, Any] = {}
        if self._megatron_sidecar:
            for key in ("megatron_transformer_config", "megatron_hf_name_map"):
                if key in self._megatron_sidecar:
                    registry_extras[key] = self._megatron_sidecar[key]
        registry_blob = encode_registry(
            self._registry,
            version=version,
            trainer_world_layout=self._world_layout.encode(),
            extras=registry_extras or None,
        )

        # Fold the v2 metadata into the underlying publisher's
        # extra_parameters via a monkey-patched _build_identity (the
        # forward-compatible path) AND attach a synthetic
        # ``TensorDescriptor(name=_V2_SIDECAR_NAME, dtype=<json>)`` to the
        # outgoing WorkerMetadata (the path that survives the current
        # Rust server's GetMetadata field-dropping). Receivers look at
        # both: identity.extra_parameters first, then the sidecar
        # descriptor.
        original_build_identity = self._publisher._build_identity

        def _build_identity_with_v2(step: int) -> p2p_pb2.SourceIdentity:
            ident = original_build_identity(step)
            ident.extra_parameters["role"] = ROLE_TRAINER
            ident.extra_parameters["mx_v2"] = "1"
            ident.extra_parameters["worker_rank"] = str(self._worker_rank)
            ident.extra_parameters["shape_registry"] = registry_blob
            ident.extra_parameters["world_layout"] = self._world_layout.encode()
            # If any registered tensor carries a megatron_role, stamp the
            # source as Megatron-shaped so receivers can route through the
            # Megatron slice planner. Per-source rank-position metadata
            # (tp_rank, tp_size, pp_rank, pp_size, ep_rank, ep_size) is
            # filled from world_layout — same source publishes all roles.
            if any(d.megatron_role is not None for d in self._registry):
                ident.extra_parameters["publisher_kind"] = "megatron"
                wl = self._world_layout
                ident.extra_parameters["tp_rank"] = str(self._megatron_tp_rank)
                ident.extra_parameters["tp_size"] = str(wl.tp_world_size)
                ident.extra_parameters["pp_rank"] = str(self._megatron_pp_rank)
                ident.extra_parameters["pp_size"] = str(wl.pp_world_size)
                ident.extra_parameters["ep_rank"] = str(self._megatron_ep_rank)
                ident.extra_parameters["ep_size"] = str(wl.ep_world_size)
            return ident

        # Build the v2 sidecar payload (preserves all the same data as
        # extra_parameters but in a transport the server actually echoes).
        # ``shape_registry`` is intentionally embedded as a nested JSON string
        # inside this JSON document — receivers parse the outer JSON with
        # decode_registry's matching call to handle the inner blob.
        sidecar_dict: dict[str, Any] = {
            "mx_v2": "1",
            "role": ROLE_TRAINER,
            "worker_rank": int(self._worker_rank),
            "training_step": int(version),
            "world_layout": self._world_layout.encode(),
            "framework": "nemo_rl",
            "shape_registry": registry_blob,
        }
        if any(d.megatron_role is not None for d in self._registry):
            wl = self._world_layout
            sidecar_dict["publisher_kind"] = "megatron"
            sidecar_dict["tp_rank"] = int(self._megatron_tp_rank)
            sidecar_dict["tp_size"] = int(wl.tp_world_size)
            sidecar_dict["pp_rank"] = int(self._megatron_pp_rank)
            sidecar_dict["pp_size"] = int(wl.pp_world_size)
            sidecar_dict["ep_rank"] = int(self._megatron_ep_rank)
            sidecar_dict["ep_size"] = int(wl.ep_world_size)
            # Merge in the Bridge-derived transformer config + Megatron→HF
            # name map (set once at trainer init via set_megatron_sidecar).
            # Receivers consume these via
            # modelexpress.megatron_translator.parse_megatron_sidecar.
            for key in ("megatron_transformer_config", "megatron_hf_name_map"):
                if key in self._megatron_sidecar:
                    sidecar_dict[key] = self._megatron_sidecar[key]
        sidecar_payload = json.dumps(sidecar_dict, separators=(",", ":"))

        # Wrap the agent_name with v2 markers (legacy-server fallback path 2).
        original_agent_name = self._publisher._agent_name
        self._publisher._agent_name = (
            f"mx_v2|{ROLE_TRAINER}|rank={self._worker_rank}|"
            f"version={int(version)}|orig={original_agent_name}"
        )
        self._publisher._build_identity = _build_identity_with_v2  # type: ignore[method-assign]

        # Wrap _build_tensor_protos to append the sidecar descriptor.
        original_build_tensor_protos = self._publisher._build_tensor_protos

        def _build_tensor_protos_with_sidecar(descriptors):
            protos = original_build_tensor_protos(descriptors)
            sidecar = p2p_pb2.TensorDescriptor(
                name=_V2_SIDECAR_NAME,
                addr=0,
                size=0,
                device_id=0,
                dtype=sidecar_payload,
            )
            protos.append(sidecar)
            return protos

        self._publisher._build_tensor_protos = _build_tensor_protos_with_sidecar  # type: ignore[method-assign]

        try:
            mx_source_id = self._publisher.publish_weights(
                named_tensors=self._registered_tensors,
                step=int(version),
                worker_rank=self._worker_rank,
            )
        finally:
            self._publisher._build_identity = original_build_identity  # type: ignore[method-assign]
            self._publisher._agent_name = original_agent_name
            self._publisher._build_tensor_protos = original_build_tensor_protos  # type: ignore[method-assign]

        logger.info(
            "MxV2 publish: rank=%d version=%d tensors=%d mx_source_id=%s",
            self._worker_rank,
            version,
            len(self._registered_tensors),
            mx_source_id,
        )
        return mx_source_id

    def mark_ready(self) -> bool:
        """Mark this source as READY. Starts heartbeat if enabled."""
        ok = self._publisher.mark_ready(worker_rank=self._worker_rank)
        if ok and self._heartbeat_enabled and self._heartbeat is None:
            self._start_heartbeat()
        return ok

    def _start_heartbeat(self) -> None:
        if self._publisher._client is None or self._publisher._nixl is None:
            logger.warning("cannot start heartbeat: publisher not initialized")
            return
        self._heartbeat = HeartbeatThread(
            mx_client=self._publisher._client,
            mx_source_id=self._publisher.mx_source_id or "",
            worker_id=self._publisher.worker_id,
            worker_rank=self._worker_rank,
            nixl_manager=self._publisher._nixl,
        )
        self._heartbeat.start()

    def shutdown(self) -> None:
        """Stop heartbeat (marks STALE) and tear down the publisher."""
        if self._heartbeat is not None:
            self._heartbeat.stop()
            self._heartbeat = None
        self._publisher.shutdown()
        self._initialized = False


@dataclass
class MegatronSourceMeta:
    """Per-SOURCE Megatron-publish metadata extracted from extra_parameters.

    Populated only when the source's v2 metadata signals a Megatron
    publisher (the per-tensor registry carries ``megatron_role`` keys).
    Absent (``None`` on the candidate) for DTensor and PrimeRL
    publishers; the planner short-circuits to the existing pickers in
    that case.

    Per-source metadata is intentionally **rank-position only**
    (tp_rank, tp_size, pp_rank, ep_rank, etc). The per-tensor role and
    role-specific descriptor extras live on the source's
    :class:`shape_descriptors.TensorDescriptorV2` registry entries —
    one source publishes many tensors with different roles, so role is
    not a source-level attribute.
    """

    tp_rank: int
    tp_size: int
    pp_rank: int = 0
    pp_size: int = 1
    ep_rank: int = 0
    ep_size: int = 1
    is_megatron: bool = True


@dataclass
class V2SourceCandidate:
    """A discovered source with v2 metadata parsed."""

    ref: SourceRef
    role: str  # "trainer" | "inference_replica"
    worker_rank: int
    registry: dict | None  # decoded registry; None for inference_replica
    owned_experts_per_layer: dict[int, set[int]]  # layer_idx → expert IDs
    updated_at: int  # ms epoch
    megatron_meta: MegatronSourceMeta | None = None  # set iff publisher is Megatron


@dataclass
class MegatronSliceSource:
    """One source's contribution to a Megatron slice plan.

    A plan covers one HF parameter; it has one or more sources whose
    target_local_range together tile the global tensor along the role's
    shard axis. Replicated and per-expert plans use a single source each.
    """

    mx_source_id: str
    worker_id: str
    source_rank: int                       # tp_rank for tp-sharded; ep_rank for expert_*; 0 for replicated/PP-only
    source_pp_rank: int                    # 0 if PP=1
    # The slice of the global tensor this source contributes.
    # For matched-TP, target_local_range == source's natural shard range.
    # For mixed-TP, target_local_range is the receiver-side range and
    # source_subslice is set to extract a partial range from the source.
    target_local_range: tuple[int, int]
    source_subslice: tuple[int, int] | None = None
    # Verbatim copy of the source's role-specific descriptor extras
    # (num_heads_local, head_dim, qkv_interleave, etc.). Receiver uses these
    # for QKV un-interleave / gated-MLP split assembly.
    role_extras: dict[str, str] = field(default_factory=dict)


@dataclass
class MegatronTensorSpec:
    """Receiver's per-tensor input to the Megatron slice planner.

    The receiver tells the planner: "for this Megatron-shaped tensor name,
    I want a slice plan covering this global shape, this dtype, sharded on
    this axis, with this role". The planner figures out which sources
    contribute and how their slices map into the receiver's target window.

    For ``replicated``: ``shard_axis`` is unused; ``target_shape`` is the
    full global shape.

    For tp-sharded roles (``column``, ``row``, ``vocab_parallel``,
    ``qkv_column``, ``gated_mlp_column``): ``target_shape`` is the GLOBAL
    shape; ``shard_axis`` is the dim along which the publishers tile;
    the planner computes the receiver's per-rank window inside that.

    For ``expert_column`` / ``expert_row``: ``target_shape`` is the shape
    of ONE expert's slice; the receiver passes the local expert IDs via
    ``role_descriptor['local_expert_ids']`` (comma-separated) and the
    layer id via ``role_descriptor['layer_id']``.
    """

    role: str
    target_shape: tuple[int, ...]
    target_dtype: str
    shard_axis: int = 0
    pp_rank: int = 0  # which PP stage owns this tensor
    role_descriptor: dict[str, str] = field(default_factory=dict)


def _role_extras_from_meta(mm: "MegatronSourceMeta") -> dict[str, str]:
    """Stub — returns empty dict.

    Per-tensor role extras live on the source's
    :class:`shape_descriptors.TensorDescriptorV2` registry entries, not on
    the per-source :class:`MegatronSourceMeta`. The receiver's per-tensor
    :class:`MegatronTensorSpec.role_descriptor` is authoritative for
    assembly. This function is kept (returning empty) so callers that
    forwarded its output to ``MegatronSliceSource.role_extras`` continue
    to compile.
    """
    return {}


@dataclass
class MegatronSlicePlan:
    """Receiver-side coverage plan for one HF-named parameter.

    Constructed by ``MxV2RefitReceiver.pick_megatron_slice_plans`` from a
    candidate set + a ``TargetTpLayout``. Each plan tells the assembler:
    (a) what shape to pre-allocate, (b) what slice-views to register with
    NIXL for parallel pulls, (c) what assembly transform to apply
    post-pull, (d) the Megatron role descriptor for receiver-side
    translation (QKV un-interleave, gate||up split, etc.).
    """

    tensor_name: str                        # source-side (Megatron) name
    role: str                               # one of the 7 Megatron roles
    target_shape: tuple[int, ...]
    target_dtype: str
    sources: list[MegatronSliceSource]
    assembly: str                           # "concat_dim0" | "concat_dim1"
                                            # | "qkv_uninterleave" | "gated_mlp_split"
                                            # | "per_expert" | "passthrough"
    # Aggregated descriptor across sources. For qkv_column this is the
    # SUM of per-source num_heads_local / num_kv_heads_local (i.e. the
    # global head count), plus the shared head_dim. Receiver translator
    # consumes these directly.
    role_descriptor: dict[str, str] = field(default_factory=dict)


class MxV2RefitReceiver:
    """v2 inference-side receiver.

    Wraps :class:`MxRefitReceiver` and adds:

    - **Same-rank source selection**: by default, picks a candidate with
      ``worker_rank == self.worker_rank``. Falls back to other ranks only
      if explicitly requested.

    - **Freshest-first dedup**: when multiple candidates match the rank
      filter, picks the one with the latest ``updated_at``. (Same fix
      as PrimeRL's runtime patch — applied as the default here.)

    - **Tree fan-out**: after a successful receive, optionally calls
      :meth:`publish_self_as_source` to make this rank's buffers
      available to subsequent receivers.

    - **Expert filtering**: when ``my_owned_experts_per_layer`` is set,
      receives only the slices of expert tensors that this rank actually
      uses.
    """

    def __init__(
        self,
        *,
        agent_name: str,
        device_id: int,
        mx_server_url: str,
        worker_rank: int,
        listen_port: int | None = None,
    ):
        self._receiver = MxRefitReceiver(
            agent_name=agent_name,
            device_id=device_id,
            mx_server_url=mx_server_url,
            listen_port=listen_port,
        )
        self._worker_rank = worker_rank
        self._initialized = False
        self._registered_buffers: dict[str, torch.Tensor] = {}

    @property
    def worker_rank(self) -> int:
        return self._worker_rank

    def initialize(
        self,
        *,
        model_tensors: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Initialize NIXL agent + MX client. Optionally register receive buffers."""
        self._receiver.initialize(model_tensors=model_tensors)
        if model_tensors:
            self._registered_buffers = dict(model_tensors)
        self._initialized = True
        logger.info(
            "MxV2RefitReceiver initialized: rank=%d buffers=%d",
            self._worker_rank,
            len(self._registered_buffers),
        )

    def discover_v2_sources(
        self,
        *,
        model_name: str,
        min_version: int = 0,
        same_rank_only: bool = True,
        include_replicas: bool = True,
    ) -> list[V2SourceCandidate]:
        """List candidate v2 sources, filtering and sorting per the v2 rules.

        Args:
            model_name: model name to filter on.
            min_version: only return sources whose ``version`` (== training
                step) is at least this.
            same_rank_only: if True (default), only return candidates whose
                ``worker_rank`` equals this receiver's rank.
            include_replicas: whether to include other inference ranks that
                have already received and republished. Combined with
                ``same_rank_only``, this means "same-rank trainer + any
                same-rank inference replica".

        Returns:
            Candidates sorted by freshness (largest ``updated_at`` first).
            Empty list if none matched.
        """
        if not self._initialized:
            raise RuntimeError("call initialize() before discover_v2_sources()")

        client = self._receiver._client
        assert client is not None, "_receiver._client must be set after initialize()"
        try:
            response = client.list_sources(
                status_filter=p2p_pb2.SOURCE_STATUS_READY,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("list_sources failed: %s", e)
            return []

        candidates: list[V2SourceCandidate] = []
        for instance in response.instances:
            if instance.model_name != model_name:
                continue

            # Resolve the full identity to read v2 metadata.
            try:
                meta = client.get_metadata(
                    instance.mx_source_id, instance.worker_id
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "get_metadata failed for %s: %s", instance.worker_id, e
                )
                continue
            if not getattr(meta, "found", False):
                continue

            # Read v2 metadata. We try three transports in order:
            #   (a) SourceIdentity.extra_parameters (the cleanest path; works
            #       once the Rust server populates GetMetadataResponse.identity).
            #   (b) Synthetic TensorDescriptor sidecar named ``__mx_v2_meta__``
            #       (preserved by the current Rust server; the path the
            #       prototype actually uses today).
            #   (c) WorkerMetadata.agent_name string-encoded marker (legacy).
            identity = getattr(meta, "identity", None)
            extra: dict[str, str] = (
                dict(identity.extra_parameters)
                if identity is not None and identity.extra_parameters
                else {}
            )
            if not extra:
                # Sidecar transport: scan tensors for the magic marker.
                for td in meta.worker.tensors:
                    if td.name == _V2_SIDECAR_NAME and td.dtype:
                        try:
                            sidecar = json.loads(td.dtype)
                            if isinstance(sidecar, dict):
                                for k, v in sidecar.items():
                                    extra[k] = str(v)
                        except (json.JSONDecodeError, TypeError):
                            pass
                        break
            if not extra:
                # Agent-name transport: "mx_v2|<role>|rank=N|version=K|orig=...".
                agent_name = getattr(meta.worker, "agent_name", "") or ""
                if agent_name.startswith("mx_v2|"):
                    parts = agent_name.split("|")
                    if len(parts) >= 4:
                        extra["mx_v2"] = "1"
                        extra["role"] = parts[1]
                        for piece in parts[2:]:
                            if "=" in piece:
                                k, v = piece.split("=", 1)
                                if k == "rank":
                                    extra["worker_rank"] = v
                                elif k == "version":
                                    extra["training_step"] = v
            if extra.get("mx_v2") != "1":
                # Not a v2 source; ignore.
                continue
            role = extra.get("role", "")
            if role == ROLE_TRAINER and not include_replicas and False:
                pass  # always include trainer
            if role not in (ROLE_TRAINER, ROLE_INFERENCE_REPLICA):
                continue
            if role == ROLE_INFERENCE_REPLICA and not include_replicas:
                continue

            try:
                src_rank = int(extra.get("worker_rank", "-1"))
            except ValueError:
                continue
            if same_rank_only and src_rank != self._worker_rank:
                continue

            try:
                version = int(extra.get("training_step", "0"))
            except ValueError:
                continue
            if version < min_version:
                continue

            registry_blob = extra.get("shape_registry", "")
            registry = decode_registry(registry_blob) if registry_blob else None

            owned_blob = extra.get("owned_experts_per_layer", "")
            owned_experts_per_layer: dict[int, set[int]] = {}
            if owned_blob:
                # encoding: "L0:0,1,2|L1:3,4,5"
                for chunk in owned_blob.split("|"):
                    if ":" not in chunk:
                        continue
                    lid, ids = chunk.split(":", 1)
                    owned_experts_per_layer[int(lid.lstrip("L"))] = decode_expert_set(
                        ids
                    )

            updated_at = int(getattr(meta.worker, "updated_at", 0) or 0)

            megatron_meta = _extract_megatron_meta(extra)

            candidates.append(
                V2SourceCandidate(
                    ref=SourceRef(
                        mx_source_id=instance.mx_source_id,
                        worker_id=instance.worker_id,
                        model_name=instance.model_name,
                        worker_rank=src_rank,
                        training_step=version,
                    ),
                    role=role,
                    worker_rank=src_rank,
                    registry=registry,
                    owned_experts_per_layer=owned_experts_per_layer,
                    updated_at=updated_at,
                    megatron_meta=megatron_meta,
                )
            )

        # Topology score: prefer trainer over inference_replica (trainer is
        # always authoritative); within that, prefer freshest.
        candidates.sort(
            key=lambda c: (
                0 if c.role == ROLE_TRAINER else 1,
                -c.updated_at,
            )
        )
        return candidates

    def pick_best_source(
        self,
        candidates: list[V2SourceCandidate],
        *,
        needed_experts_per_layer: dict[int, set[int]] | None = None,
    ) -> V2SourceCandidate | None:
        """Pick the best candidate. Optionally requires expert coverage.

        If ``needed_experts_per_layer`` is set, the candidate must own a
        superset of the requested experts (or be a trainer with full info).
        """
        if not candidates:
            return None
        if needed_experts_per_layer is None:
            return candidates[0]

        for cand in candidates:
            if cand.role == ROLE_TRAINER:
                # Trainer publishes its rank's owned set in the registry; if
                # we need experts the trainer doesn't own, no single source
                # has them and the caller has to multi-source. v0 punts.
                return cand
            covers_all = all(
                needed.issubset(cand.owned_experts_per_layer.get(layer, set()))
                for layer, needed in needed_experts_per_layer.items()
            )
            if covers_all:
                return cand
        return None

    def pick_megatron_slice_plans(
        self,
        candidates: list[V2SourceCandidate],
        *,
        target_tp_layout: TargetTpLayout,
        target_tensor_specs: dict[str, "MegatronTensorSpec"],
    ) -> list[MegatronSlicePlan]:
        """Build per-tensor slice-coverage plans from a Megatron candidate set.

        Args:
            candidates: As returned by :meth:`discover_v2_sources`. Mixed
                Megatron + non-Megatron candidate lists are supported; the
                planner only consumes those whose ``megatron_meta`` is set.
            target_tp_layout: The receiver's desired TP × EP layout.
            target_tensor_specs: For each HF parameter the receiver needs,
                a :class:`MegatronTensorSpec` describing the source-side
                tensor name (Megatron-shaped), global shape, dtype, and
                — for fused QKV / gated MLP — the role-specific descriptor
                that the assembler will need (head counts, etc).

        Returns:
            One :class:`MegatronSlicePlan` per entry in
            ``target_tensor_specs``. The planner does not attempt to detect
            missing source coverage; the caller is responsible for handling
            the empty-sources case (a returned plan with ``sources=[]``
            indicates discovery hasn't caught up yet, callers should retry).

        Backwards compatibility: if no candidate carries Megatron metadata,
        callers should use :meth:`pick_best_source` (the FSDP / EP-experts
        picker). This method always returns a plan list — possibly with
        empty ``sources`` entries — so callers can detect that case
        uniformly.
        """
        megatron_cands = [c for c in candidates if c.megatron_meta is not None]
        plans: list[MegatronSlicePlan] = []

        for source_name, spec in target_tensor_specs.items():
            plan = self._plan_one_megatron_tensor(
                source_name=source_name,
                spec=spec,
                candidates=megatron_cands,
                target_tp_layout=target_tp_layout,
            )
            plans.append(plan)
        return plans

    def _plan_one_megatron_tensor(
        self,
        *,
        source_name: str,
        spec: "MegatronTensorSpec",
        candidates: list[V2SourceCandidate],
        target_tp_layout: TargetTpLayout,
    ) -> MegatronSlicePlan:
        """Build a single MegatronSlicePlan for one source-side tensor name."""
        role = spec.role

        # Filter candidates to those covering this PP stage. Per-tensor
        # role determines assembly (set by the receiver in spec.role); it
        # is NOT a source-level filter — every Megatron rank publishes
        # tensors of all roles in one source.
        relevant: list[V2SourceCandidate] = []
        for c in candidates:
            mm = c.megatron_meta
            assert mm is not None  # filtered above
            if mm.pp_rank != spec.pp_rank:
                continue
            relevant.append(c)

        if role == ROLE_MEGATRON_REPLICATED:
            # One source — the rank-0 publisher. Pick the freshest if
            # multiple races exist.
            relevant.sort(key=lambda c: -c.updated_at)
            sources: list[MegatronSliceSource] = []
            if relevant:
                c = relevant[0]
                sources.append(
                    MegatronSliceSource(
                        mx_source_id=c.ref.mx_source_id,
                        worker_id=c.ref.worker_id,
                        source_rank=0,
                        source_pp_rank=spec.pp_rank,
                        target_local_range=(0, spec.target_shape[0]),
                        role_extras={},
                    )
                )
            return MegatronSlicePlan(
                tensor_name=source_name,
                role=role,
                target_shape=spec.target_shape,
                target_dtype=spec.target_dtype,
                sources=sources,
                assembly="passthrough",
                role_descriptor=dict(spec.role_descriptor),
            )

        if role in (ROLE_MEGATRON_EXPERT_COLUMN, ROLE_MEGATRON_EXPERT_ROW):
            # Two MoE layouts in the wild:
            #
            # 1. ``expert_layout="grouped"`` — Megatron-Core TE-grouped
            #    linears (``TEColumnParallelGroupedLinear``,
            #    ``TERowParallelGroupedLinear``) expose **each local
            #    expert's slice as a separate ``nn.Parameter``** named
            #    ``weight0`` / ``weight1`` / .... The classifier emits one
            #    Megatron-tensor-name per expert. The slice plan is
            #    therefore identity / passthrough — one source per
            #    Megatron tensor, no per-expert dict assembly.
            # 2. ``expert_layout="leading_axis"`` (legacy / EP>1 single
            #    ``.weight``) — one Megatron tensor holds ``ep_size``
            #    experts stacked along axis 0. The legacy
            #    ``_plan_per_expert`` covers this: one source per local
            #    expert id, the assembler emits a ``dict[expert_id,
            #    tensor]``.
            #
            # The per-source ``expert_layout`` lives in
            # :attr:`MegatronTensorSpec.role_descriptor` so the receiver
            # can dispatch. Default to ``leading_axis`` for backwards
            # compatibility with v0 callers that don't set it.
            expert_layout = spec.role_descriptor.get(
                "expert_layout", "leading_axis"
            )
            if expert_layout == "grouped":
                return self._plan_grouped_per_expert(
                    source_name=source_name, spec=spec,
                    relevant=relevant, target_tp_layout=target_tp_layout,
                )
            return self._plan_per_expert(
                source_name=source_name, spec=spec,
                relevant=relevant, target_tp_layout=target_tp_layout,
            )

        # tp-sharded roles: column / row / vocab_parallel / qkv_column /
        # gated_mlp_column. Compute the receiver's slice along the role's
        # shard axis and tile sources to cover it.
        return self._plan_tp_sharded(
            source_name=source_name, spec=spec,
            relevant=relevant, target_tp_layout=target_tp_layout,
        )

    def _plan_tp_sharded(
        self,
        *,
        source_name: str,
        spec: "MegatronTensorSpec",
        relevant: list[V2SourceCandidate],
        target_tp_layout: TargetTpLayout,
    ) -> MegatronSlicePlan:
        """Plan a column/row/vocab_parallel/qkv/gated_mlp tensor.

        Decides matched-TP vs mixed-TP and produces one MegatronSliceSource
        per source range that overlaps the receiver's target range.
        """
        role = spec.role
        shard_axis = spec.shard_axis
        global_extent = spec.target_shape[shard_axis]
        target_tp = target_tp_layout.tp_size
        target_rank = target_tp_layout.tp_rank

        # The receiver wants [target_lo, target_hi) along the shard axis.
        per_target = global_extent // target_tp
        target_lo = target_rank * per_target
        target_hi = (
            global_extent if target_rank == target_tp - 1 else (target_rank + 1) * per_target
        )

        # Sort sources by tp_rank so we walk them in slice order.
        relevant_sorted = sorted(
            relevant,
            key=lambda c: (c.megatron_meta.tp_rank, -c.updated_at),
        )
        # Dedup per-tp_rank to the freshest.
        seen: dict[int, V2SourceCandidate] = {}
        for c in relevant_sorted:
            if c.megatron_meta.tp_rank not in seen:
                seen[c.megatron_meta.tp_rank] = c
        ordered = [seen[r] for r in sorted(seen.keys())]

        if not ordered:
            return self._empty_plan(source_name, spec)

        source_tp = ordered[0].megatron_meta.tp_size
        per_source = global_extent // source_tp

        sources: list[MegatronSliceSource] = []
        contributing: list[V2SourceCandidate] = []
        for c in ordered:
            mm = c.megatron_meta
            src_lo = mm.tp_rank * per_source
            src_hi = (
                global_extent if mm.tp_rank == source_tp - 1 else (mm.tp_rank + 1) * per_source
            )
            # Overlap with target window?
            ov_lo = max(src_lo, target_lo)
            ov_hi = min(src_hi, target_hi)
            if ov_lo >= ov_hi:
                continue
            sub: tuple[int, int] | None = None
            if (ov_lo, ov_hi) != (src_lo, src_hi):
                # Mixed-TP: pull only a sub-range of the source's slice.
                sub = (ov_lo - src_lo, ov_hi - src_lo)
            sources.append(
                MegatronSliceSource(
                    mx_source_id=c.ref.mx_source_id,
                    worker_id=c.ref.worker_id,
                    source_rank=mm.tp_rank,
                    source_pp_rank=mm.pp_rank,
                    target_local_range=(ov_lo - target_lo, ov_hi - target_lo),
                    source_subslice=sub,
                    role_extras=_role_extras_from_meta(mm),
                )
            )
            contributing.append(c)

        # Aggregate role descriptor across the CONTRIBUTING sources only.
        # For QKV, this gives the head count of the receiver-side assembled
        # buffer (not the global trainer total) — which is what
        # _uninterleave_qkv consumes.
        agg = self._aggregate_role_descriptor(role, contributing, base=spec.role_descriptor)
        assembly = self._assembly_for_role(role)

        # Receiver-side target shape after assembly is the per-target-rank
        # slice along the shard axis, full extent on the others.
        target_shape = list(spec.target_shape)
        target_shape[shard_axis] = target_hi - target_lo
        return MegatronSlicePlan(
            tensor_name=source_name,
            role=role,
            target_shape=tuple(target_shape),
            target_dtype=spec.target_dtype,
            sources=sources,
            assembly=assembly,
            role_descriptor=agg,
        )

    def _plan_per_expert(
        self,
        *,
        source_name: str,
        spec: "MegatronTensorSpec",
        relevant: list[V2SourceCandidate],
        target_tp_layout: TargetTpLayout,
    ) -> MegatronSlicePlan:
        """v0: one source per local expert id, picked from the EP rank that owns it.

        ``spec.target_shape`` is interpreted as the SHAPE of one expert
        (the assembler emits per-expert tensors, not a stacked one).
        ``role_descriptor['local_expert_ids']`` lists which experts the
        receiver needs.
        """
        # Inference rank's owned experts come from the layout. v0 expects
        # the caller to pass them via spec.role_descriptor['local_expert_ids']
        # (encoded as a comma-separated string of ints).
        ids_str = spec.role_descriptor.get("local_expert_ids", "")
        wanted = [int(x) for x in ids_str.split(",") if x.strip()]
        sources: list[MegatronSliceSource] = []
        # Layer id is part of the spec for expert selection.
        layer_id = int(spec.role_descriptor.get("layer_id", "0"))

        for expert_id in wanted:
            owner = next(
                (
                    c for c in relevant
                    if expert_id in c.owned_experts_per_layer.get(layer_id, set())
                ),
                None,
            )
            if owner is None:
                continue
            sources.append(
                MegatronSliceSource(
                    mx_source_id=owner.ref.mx_source_id,
                    worker_id=owner.ref.worker_id,
                    source_rank=owner.megatron_meta.ep_rank,
                    source_pp_rank=owner.megatron_meta.pp_rank,
                    target_local_range=(expert_id, expert_id + 1),
                    role_extras={"expert_id": str(expert_id)},
                )
            )

        return MegatronSlicePlan(
            tensor_name=source_name,
            role=spec.role,
            target_shape=spec.target_shape,
            target_dtype=spec.target_dtype,
            sources=sources,
            assembly="per_expert",
            role_descriptor=dict(spec.role_descriptor),
        )

    def _plan_grouped_per_expert(
        self,
        *,
        source_name: str,
        spec: "MegatronTensorSpec",
        relevant: list[V2SourceCandidate],
        target_tp_layout: TargetTpLayout,
    ) -> MegatronSlicePlan:
        """Plan one grouped TE per-expert parameter.

        The publisher emits one Megatron tensor per (layer, expert) pair
        (``...linear_fc1.weight0`` etc.), each carrying a single expert's
        weights. So a slice plan is just identity: pick one fresh source
        that publishes this tensor name and pass through. Bridge's name
        map handles the per-expert HF naming (1 HF name for ``linear_fc2``
        / down_proj, 2 HF names for ``linear_fc1`` / fused gate+up — the
        translator splits the latter via ``split_gated_mlp``).

        Mixed-TP within the expert's own TP dimension (each expert's
        weight is column- or row-parallel across TP ranks) is a v1
        extension. v0 treats grouped per-expert tensors as if TP=1 on the
        expert axis — fine for ``expert_*`` roles on the standard
        Megatron-Core grouped layout where each ``weight<N>`` is the
        full per-expert tensor.
        """
        sources: list[MegatronSliceSource] = []
        if relevant:
            relevant.sort(key=lambda c: -c.updated_at)
            c = relevant[0]
            mm = c.megatron_meta
            sources.append(
                MegatronSliceSource(
                    mx_source_id=c.ref.mx_source_id,
                    worker_id=c.ref.worker_id,
                    source_rank=mm.ep_rank,
                    source_pp_rank=mm.pp_rank,
                    target_local_range=(0, spec.target_shape[0]),
                    role_extras=dict(spec.role_descriptor),
                )
            )
        return MegatronSlicePlan(
            tensor_name=source_name,
            role=spec.role,
            target_shape=spec.target_shape,
            target_dtype=spec.target_dtype,
            sources=sources,
            assembly="passthrough",
            role_descriptor=dict(spec.role_descriptor),
        )

    @staticmethod
    def _empty_plan(source_name: str, spec: "MegatronTensorSpec") -> MegatronSlicePlan:
        return MegatronSlicePlan(
            tensor_name=source_name,
            role=spec.role,
            target_shape=spec.target_shape,
            target_dtype=spec.target_dtype,
            sources=[],
            assembly=MxV2RefitReceiver._assembly_for_role(spec.role),
            role_descriptor=dict(spec.role_descriptor),
        )

    @staticmethod
    def _assembly_for_role(role: str) -> str:
        if role == ROLE_MEGATRON_QKV_COLUMN:
            return "qkv_uninterleave"
        if role == ROLE_MEGATRON_GATED_MLP_COLUMN:
            return "gated_mlp_split"
        if role == ROLE_MEGATRON_ROW:
            return "concat_dim1"
        if role in (
            ROLE_MEGATRON_COLUMN,
            ROLE_MEGATRON_VOCAB_PARALLEL,
        ):
            return "concat_dim0"
        if role == ROLE_MEGATRON_REPLICATED:
            return "passthrough"
        if role in (ROLE_MEGATRON_EXPERT_COLUMN, ROLE_MEGATRON_EXPERT_ROW):
            return "per_expert"
        raise ValueError(f"unknown megatron role: {role}")

    @staticmethod
    def _aggregate_role_descriptor(
        role: str,
        sources: list[V2SourceCandidate],
        *,
        base: dict[str, str],
    ) -> dict[str, str]:
        """Forward the receiver-supplied descriptor (per-tensor extras).

        Per-tensor descriptor extras (head counts for ``qkv_column``,
        ``gated_mlp_order`` for ``gated_mlp_column``, etc.) are
        receiver-owned: the receiver's ``MegatronTensorSpec.role_descriptor``
        carries them (typically derived from the HF model config).

        The publisher's per-tensor ``megatron_extras`` blob — visible in
        the source's shape_registry — is available for the receiver to
        cross-check, but it's not required to drive assembly. Returning
        ``base`` unchanged means the receiver's spec is authoritative.
        """
        return dict(base)

    def receive_from(
        self,
        candidate: V2SourceCandidate,
        *,
        timeout_seconds: float = 300.0,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Pull the candidate's tensors via NIXL RDMA into our pre-registered buffers.

        Wraps :meth:`MxRefitReceiver.receive_weights`. Yielded tensors are
        the same buffers that were registered at ``initialize`` time.
        """
        yield from self._receiver.receive_weights(
            candidate.ref, timeout_seconds=timeout_seconds
        )

    def publish_self_as_source(
        self,
        *,
        version: int,
        model_name: str,
    ) -> str | None:
        """Make this receiver's buffers available to other receivers.

        Implements the TensorHub pipeline-replication trick: after we've
        successfully received a version, we publish ourselves as an
        ``inference_replica`` source so that any rank N receiver who hasn't
        yet pulled can pull from us instead of contending on the trainer.

        Emits v2 metadata via all three transports the trainer's
        :meth:`MxV2TrainingPublisher.publish` uses, in parallel:

        1. ``identity.extra_parameters`` — clean path; works once the
           Rust server round-trips ``SourceIdentity`` on
           ``GetMetadataResponse`` (server PR #162571f and later).
        2. A ``__mx_v2_meta__`` synthetic ``TensorDescriptor``
           (``size=0``, JSON payload in ``dtype``) — the path that
           survives the older Rust server's identity-strip behaviour.
        3. A ``mx_v2|<role>|rank=N|version=K|orig=...`` ``agent_name``
           marker on the outgoing ``WorkerMetadata`` — legacy fallback
           if the sidecar is missing.

        Earlier versions of this method emitted only transport #1, which
        meant replicas were invisible to ``discover_v2_sources`` on any
        server that strips ``identity.extra_parameters``. Emitting all
        three matches the trainer's own publish path and makes
        receiver-to-receiver tree fan-out actually discoverable.

        Returns:
            The ``mx_source_id`` of the published replica entry, or
            ``None`` if the receiver isn't initialized / has no
            registered buffers. Server-side errors are re-raised so
            silent failures don't mask a broken catalog.
        """
        if not self._registered_buffers:
            logger.warning(
                "publish_self_as_source: no registered buffers; skipping"
            )
            return None
        client = self._receiver._client
        nixl = self._receiver._nixl
        if client is None or nixl is None:
            logger.warning(
                "publish_self_as_source: receiver not initialized; skipping"
            )
            return None

        # Transport #2 sidecar payload. shape_registry / world_layout are
        # intentionally omitted — the trainer's registry is authoritative;
        # receivers don't republish a different one.
        sidecar_payload = json.dumps(
            {
                "mx_v2": "1",
                "role": ROLE_INFERENCE_REPLICA,
                "worker_rank": int(self._worker_rank),
                "training_step": int(version),
                "framework": "nemo_rl",
            },
            separators=(",", ":"),
        )

        # Transport #1: identity.extra_parameters.
        identity = p2p_pb2.SourceIdentity(
            model_name=model_name,
            mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
            backend_framework=p2p_pb2.BACKEND_FRAMEWORK_UNKNOWN,
            tensor_parallel_size=0,
            pipeline_parallel_size=0,
            expert_parallel_size=0,
            dtype="bfloat16",  # not load-bearing for replica; receivers ignore
            quantization="",
            extra_parameters={
                "role": ROLE_INFERENCE_REPLICA,
                "mx_v2": "1",
                "worker_rank": str(self._worker_rank),
                "training_step": str(int(version)),
                "training_framework": "nemo_rl",
            },
        )

        # Transport #2: append a __mx_v2_meta__ sidecar TensorDescriptor.
        descriptors = nixl.tensor_descriptors  # populated at register time
        tensor_protos = [
            p2p_pb2.TensorDescriptor(
                name=d.name,
                addr=d.addr,
                size=d.size,
                device_id=d.device_id,
                dtype=d.dtype,
            )
            for d in descriptors
        ]
        tensor_protos.append(
            p2p_pb2.TensorDescriptor(
                name=_V2_SIDECAR_NAME,
                addr=0,
                size=0,
                device_id=0,
                dtype=sidecar_payload,
            )
        )

        # Transport #3: mx_v2|... marker on the outgoing agent_name.
        agent_name_marker = (
            f"mx_v2|{ROLE_INFERENCE_REPLICA}|rank={self._worker_rank}|"
            f"version={int(version)}|orig={self._receiver._agent_name}"
        )

        worker_meta = p2p_pb2.WorkerMetadata(
            worker_rank=self._worker_rank,
            nixl_metadata=nixl.nixl_metadata,
            tensors=tensor_protos,
            status=p2p_pb2.SOURCE_STATUS_READY,
            agent_name=agent_name_marker,
        )

        # MxRefitReceiver.initialize assigns _worker_id eagerly, so this
        # is always populated. The hasattr-fallback is retained for
        # forward compatibility with caller-provided receivers that
        # might not run through initialize().
        worker_id = (
            self._receiver._worker_id
            if hasattr(self._receiver, "_worker_id") and self._receiver._worker_id
            else f"{self._receiver._agent_name}-replica-{int(version)}"
        )

        try:
            mx_source_id = client.publish_metadata(
                identity=identity,
                worker=worker_meta,
                worker_id=worker_id,
            )
        except Exception:
            # Re-raise rather than silently returning None. Earlier
            # versions swallowed this; the result was that
            # ``publish_self_as_source`` looked like it was succeeding
            # (returning a str-ish from the caller's perspective) while
            # the catalog stayed empty. Surfacing the exception lets
            # the caller's retry logic handle it explicitly.
            logger.error(
                "publish_self_as_source failed for rank=%d version=%d",
                self._worker_rank,
                version,
                exc_info=True,
            )
            raise
        logger.info(
            "Published self as inference_replica: rank=%d version=%d mx_source_id=%s",
            self._worker_rank,
            version,
            mx_source_id,
        )
        return mx_source_id

    def shutdown(self) -> None:
        # MxRefitReceiver has no shutdown method in the existing code; the
        # NIXL transfer manager and MxClient are torn down by Python's gc.
        # Future: when refit_receiver gains a shutdown(), call it here.
        self._initialized = False


__all__ = [
    "MegatronSlicePlan",
    "MegatronSliceSource",
    "MegatronSourceMeta",
    "MegatronTensorSpec",
    "MxV2RefitReceiver",
    "MxV2TrainingPublisher",
    "ROLE_INFERENCE",
    "ROLE_INFERENCE_REPLICA",
    "ROLE_MEGATRON_COLUMN",
    "ROLE_MEGATRON_EXPERT_COLUMN",
    "ROLE_MEGATRON_EXPERT_ROW",
    "ROLE_MEGATRON_GATED_MLP_COLUMN",
    "ROLE_MEGATRON_QKV_COLUMN",
    "ROLE_MEGATRON_REPLICATED",
    "ROLE_MEGATRON_ROW",
    "ROLE_MEGATRON_VOCAB_PARALLEL",
    "ROLE_TRAINER",
    "TargetTpLayout",
    "TrainerWorldLayout",
    "V2SourceCandidate",
]
