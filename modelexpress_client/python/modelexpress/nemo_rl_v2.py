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
from dataclasses import dataclass
from typing import Iterator

import torch

from . import p2p_pb2
try:
    # Modern modelexpress layout (post-v0.6 / post-PR-#349): heartbeat
    # lives under a metadata sub-package.
    from .metadata.heartbeat import HeartbeatThread
except ImportError:
    # Older modelexpress layout (v0.5.x, the prime-rl-mx-on-nixl image
    # in use during the post-#2389 cluster work): heartbeat is at the
    # package root. Same class, different import path.
    from .heartbeat import HeartbeatThread
from .refit_receiver import MxRefitReceiver, SourceRef
from .shape_descriptors import (
    COMPILE_TARGET_HF_RAW,
    PLACEMENT_SHARD,
    NonExpertShardSpec,
    TensorDescriptorV2,
    compile_target_matches,
    decode_expert_set,
    decode_registry,
    describe_tensor,
    encode_expert_set,
    encode_registry,
)  # COMPILE_TARGET_HF_RAW is re-exported for callers passing it to add_tensor().
from .training_publisher import MxTrainingPublisher

logger = logging.getLogger("modelexpress.nemo_rl_v2")


# Role string written into ``extra_parameters["role"]``. Matches the
# convention adopted by PR #2389. Receivers filter on it to disambiguate.
ROLE_TRAINER = "trainer"
ROLE_INFERENCE = "inference"
ROLE_INFERENCE_REPLICA = "inference_replica"


def _slice_along_axis(
    tensor: torch.Tensor, axis: int, rng: tuple[int, int]
) -> torch.Tensor:
    """View ``tensor[..., rng[0]:rng[1], ...]`` along ``axis``.

    Phase-4 helper: lifts a publisher's local-shard bytes into the
    receiver's destination slice. Returns a view (no copy) when ``tensor``
    is already contiguous along ``axis``; otherwise yields a contiguous
    clone so subsequent ``torch.cat`` is well-defined.
    """
    start, end = rng
    if tensor.ndim == 0 or start == 0 and end == tensor.shape[axis]:
        return tensor
    idx: list[slice] = [slice(None)] * tensor.ndim
    idx[axis] = slice(start, end)
    out = tensor[tuple(idx)]
    return out.contiguous() if not out.is_contiguous() else out


# Synthetic tensor descriptor used as a v2 metadata sidecar. The current
# Rust MX server drops most string fields (agent_name, extra_parameters,
# metadata_endpoint, etc.) when echoing a WorkerMetadata back via
# GetMetadata, but it preserves tensor descriptors. So we abuse a
# zero-size, magic-named TensorDescriptor as the transport: the JSON v2
# payload goes in the ``dtype`` field, which is a freeform proto3 string
# the server stores verbatim. Receivers look for this marker and pull
# v2 fields from it.
_V2_SIDECAR_NAME = "__mx_v2_meta__"


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
        compile_target: str = COMPILE_TARGET_HF_RAW,
        compile_metadata: dict[str, object] | None = None,
        shard_spec: "NonExpertShardSpec | None" = None,
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
            compile_target: Phase-3a tag identifying the kernel layout
                the bytes are encoded for. Defaults to ``"hf_raw"`` —
                plain HF state-dict bytes, no kernel-specific layout.
                Callers should pass the resolved ``ConversionEntry.compile_target``
                from their conversion registry (e.g. ``"cutlass_fp8"``
                for cutlass per-channel FP8, ``"deep_gemm_fp8"`` for
                DeepGemm 128x128 blockwise).
            compile_metadata: free-form key/value blob describing the
                byte-affecting compile choices (block size, scale
                layout, kernel version, etc.). Receivers filter on this
                via :meth:`MxV2RefitReceiver.discover_v2_sources`
                ``required_compile_metadata=`` so a Cutlass receiver
                won't accidentally consume DeepGemm-block-256 bytes.
        """
        if not self._initialized:
            raise RuntimeError("call initialize() before add_tensor()")

        descriptor = describe_tensor(
            name=name,
            tensor=tensor,
            rank=self._worker_rank,
            fsdp_world_size=self._world_layout.fsdp_world_size,
            is_expert=is_expert,
            expert_axis=expert_axis,
            owned_expert_ids=tuple(sorted(owned_expert_ids)),
            compile_target=compile_target,
            compile_metadata=compile_metadata,
            shard_spec=shard_spec,
        )

        # `describe_tensor` infers the descriptor from a DTensor's
        # placements + global shape, but NIXL publication needs the local
        # CUDA shard. Unwrap DTensors here so callers don't have to patch
        # `_registered_tensors` privately after `add_tensor`.
        registered_tensor = tensor
        if hasattr(tensor, "placements") and hasattr(tensor, "to_local"):
            registered_tensor = tensor.to_local()
        if not registered_tensor.is_cuda:
            raise RuntimeError(
                f"tensor {name!r} local shard is not on CUDA; v2 publish requires GPU residency"
            )
        # NIXL register_memory requires contiguous storage. DTensor's
        # to_local() shard is contiguous in the common cases but can be
        # strided after view ops.
        if hasattr(registered_tensor, "is_contiguous") and not registered_tensor.is_contiguous():
            registered_tensor = registered_tensor.contiguous()

        # `_registry` is a per-publisher-lifetime list and was previously
        # append-only, while `_registered_tensors` is keyed by name. On a
        # long-lived publisher reused across steps, re-publishing the same
        # tensor (under a new version) would accumulate stale descriptors
        # and receivers could plan against old metadata. Replace any
        # existing entry for this name before appending.
        self._registry = [d for d in self._registry if d.name != name]
        self._registry.append(descriptor)
        # Use a key that's unique per descriptor (including any potential
        # name collisions from layer publishing). For v2 we publish all
        # tensors at once, so the name is sufficient.
        self._registered_tensors[name] = registered_tensor

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

        registry_blob = encode_registry(
            self._registry,
            version=version,
            trainer_world_layout=self._world_layout.encode(),
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
            return ident

        # Build the v2 sidecar payload (preserves all the same data as
        # extra_parameters but in a transport the server actually echoes).
        # ``shape_registry`` is intentionally embedded as a nested JSON string
        # inside this JSON document — receivers parse the outer JSON with
        # decode_registry's matching call to handle the inner blob.
        sidecar_payload = json.dumps(
            {
                "mx_v2": "1",
                "role": ROLE_TRAINER,
                "worker_rank": int(self._worker_rank),
                "training_step": int(version),
                "world_layout": self._world_layout.encode(),
                "framework": "nemo_rl",
                "shape_registry": registry_blob,
            },
            separators=(",", ":"),
        )

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
class V2SourceCandidate:
    """A discovered source with v2 metadata parsed.

    ``compile_targets`` is the set of distinct ``TensorDescriptorV2.compile_target``
    values present across the candidate's registry. A receiver filters on this
    via :meth:`MxV2RefitReceiver.discover_v2_sources` (``compile_target_filter=``).
    The most common shapes:

    - ``{"hf_raw"}`` — clean HF state-dict bytes; any kernel-aware receiver
      can compile from it.
    - ``{"deep_gemm_fp8"}`` — already quantised + reordered for DeepGemm.
    - ``{"hf_raw", "deep_gemm_fp8"}`` — mixed: some tensors raw, some compiled
      (rare but legal; receivers must check per-tensor).
    """

    ref: SourceRef
    role: str  # "trainer" | "inference_replica"
    worker_rank: int
    registry: dict | None  # decoded registry; None for inference_replica
    owned_experts_per_layer: dict[int, set[int]]  # layer_idx → expert IDs
    updated_at: int  # ms epoch
    compile_targets: frozenset[str] = frozenset({COMPILE_TARGET_HF_RAW})


@dataclass
class TargetTPLayout:
    """What slice of the global tensor a Phase-4 receiver wants.

    Phase 4 (mixed-TP / multi-source slice discovery, see post-#2389 RFC §5).
    A receiver running at inference-time describes its local view by:

      - ``world_size``: the inference-side world that's splitting the tensor
        (e.g. inference TP=8 even if trainer was TP=4).
      - ``rank``: this receiver's rank within ``world_size``. Used to compute
        an even slice by default.
      - ``shard_axis``: which tensor axis is sharded across that world.
      - ``target_range``: optional explicit ``(start, end)`` along
        ``shard_axis`` to override the default even-split math. Necessary
        for uneven layouts (e.g. expert sharding with custom owner maps).

    The publisher's ``placement_kind`` + ``local_shard_range`` are looked up
    per tensor; the planner intersects ``target_range`` against every
    publisher slice and emits the minimal candidate set.
    """

    world_size: int
    rank: int
    shard_axis: int = 0
    target_range: tuple[int, int] | None = None


@dataclass
class SliceSource:
    """One source contribution toward filling a receiver's target slice.

    Emitted by :class:`SliceCoveragePlan`. The receiver issues one NIXL
    RDMA read per ``SliceSource``, copying ``src_range`` bytes from the
    candidate's buffer into ``dst_range`` of the local destination.
    """

    candidate: V2SourceCandidate
    tensor_name: str
    src_range: tuple[int, int]
    dst_range: tuple[int, int]
    shard_axis: int


@dataclass
class _TensorPlan:
    """Internal: result of planning one tensor's coverage."""

    contributions: list[SliceSource]
    reason: str


@dataclass
class SliceCoveragePlan:
    """Result of :meth:`MxV2RefitReceiver.discover_v2_sources_for_slice`.

    Fields:
        candidates: every v2 candidate that passed the filter.
        per_tensor_sources: per-tensor list of :class:`SliceSource`
            describing how to fill that tensor's target slice. An empty
            entry means no plan was found (see ``missing``).
        missing: list of ``"name: reason"`` for tensors that couldn't be
            fully covered. If non-empty, the receiver should treat the
            plan as failed.
        target_layout: echoed back for convenience.
        legacy_single_source: True iff the picker found candidates but
            none carried a v2 registry — in that case ``per_tensor_sources``
            is empty and the caller should fall back to
            :meth:`MxV2RefitReceiver.receive_from` with ``candidates[0]``.
    """

    candidates: list[V2SourceCandidate]
    per_tensor_sources: dict[str, list[SliceSource]]
    missing: list[str]
    target_layout: TargetTPLayout
    legacy_single_source: bool = False

    @property
    def fully_covered(self) -> bool:
        return not self.missing and (
            self.legacy_single_source or bool(self.per_tensor_sources)
        )


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

        # Metrics surface for benchmarks / dashboards. Discovery numbers
        # are at the v2 layer (catalog walk + per-instance get_metadata);
        # the per-transfer RDMA numbers live on the wrapped MxRefitReceiver
        # in `self._receiver.last_stats` and `self._receiver.history`.
        self._last_discovery_seconds: float = 0.0
        self._last_discovery_candidates: int = 0

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
        compile_target_filter: set[str] | frozenset[str] | None = None,
        required_compile_metadata: dict[str, object] | None = None,
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
            compile_target_filter: receiver-side whitelist of acceptable
                ``compile_target`` strings. A candidate is admitted only if
                *every* tensor in its registry has a compile_target in this
                set (mixed-layout candidates are rejected — see RFC §5).
                ``None`` (default) accepts everything, matching pre-Phase-3
                behaviour.
            required_compile_metadata: optional kv pairs that every tensor's
                ``compile_metadata`` must match. Use for pinning block sizes,
                scale layouts, kernel versions, etc.

        Returns:
            Candidates sorted by freshness (largest ``updated_at`` first).
            Empty list if none matched.
        """
        if not self._initialized:
            raise RuntimeError("call initialize() before discover_v2_sources()")

        # Track catalog discovery time on the underlying receiver's metrics
        # so benchmarks can see control-plane latency vs RDMA latency.
        import time as _time
        discovery_start = _time.monotonic()

        client = self._receiver._client
        assert client is not None, "_receiver._client must be set after initialize()"
        try:
            response = client.list_sources(
                status_filter=p2p_pb2.SOURCE_STATUS_READY,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("list_sources failed: %s", e)
            self._last_discovery_seconds = _time.monotonic() - discovery_start
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

            # Phase 3b: enforce compile_target_filter / required_compile_metadata.
            # We require ALL tensors in the registry to match — partial matches
            # would mean the receiver consumes some bytes correctly and silently
            # corrupts others. If the candidate has no registry (e.g. v0 trainer
            # or an inference replica that didn't republish a registry), we
            # admit it only when no filter is set, so callers explicitly opt in
            # to compile-aware behaviour.
            descriptors = [
                t
                for t in (registry["tensors"] if registry else [])
                if isinstance(t, TensorDescriptorV2)
            ]
            if compile_target_filter is not None or required_compile_metadata:
                if not descriptors:
                    logger.debug(
                        "skipping candidate worker_id=%s: compile filter set "
                        "but candidate has no v2 registry",
                        instance.worker_id,
                    )
                    continue
                allowed = (
                    frozenset(compile_target_filter)
                    if compile_target_filter is not None
                    else None
                )
                ok = all(
                    compile_target_matches(
                        d,
                        allowed_targets=allowed,
                        required_metadata=required_compile_metadata,
                    )
                    for d in descriptors
                )
                if not ok:
                    logger.debug(
                        "skipping candidate worker_id=%s: compile filter mismatch "
                        "(targets=%s, want=%s)",
                        instance.worker_id,
                        sorted({d.compile_target for d in descriptors}),
                        sorted(compile_target_filter) if compile_target_filter else "*",
                    )
                    continue

            compile_targets = (
                frozenset(d.compile_target for d in descriptors)
                if descriptors
                else frozenset({COMPILE_TARGET_HF_RAW})
            )

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
                    compile_targets=compile_targets,
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
        self._last_discovery_seconds = _time.monotonic() - discovery_start
        self._last_discovery_candidates = len(candidates)
        return candidates

    def pick_best_source(
        self,
        candidates: list[V2SourceCandidate],
        *,
        needed_experts_per_layer: dict[int, set[int]] | None = None,
    ) -> V2SourceCandidate | None:
        """Pick the best candidate. Optionally requires expert coverage.

        If ``needed_experts_per_layer`` is set, the candidate must own a
        superset of the requested experts. Trainer candidates are
        checked against their published per-rank expert ownership just
        like inference replicas — being a trainer doesn't bypass the
        ownership requirement, since the trainer's rank still only
        holds a slice of the MoE experts when EP > 1.
        """
        if not candidates:
            return None
        if needed_experts_per_layer is None:
            return candidates[0]

        for cand in candidates:
            covers_all = all(
                needed.issubset(cand.owned_experts_per_layer.get(layer, set()))
                for layer, needed in needed_experts_per_layer.items()
            )
            if covers_all:
                return cand
        # No single candidate covers all needed experts. The caller must
        # multi-source via `discover_v2_sources_for_slice` / planner.
        return None

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

    def receive_from_scratch(
        self,
        candidate: V2SourceCandidate,
        *,
        timeout_seconds: float = 300.0,
        tensor_shapes: dict[str, tuple[int, ...]] | None = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Pull the candidate's tensors via NIXL into receiver-allocated buffers.

        Wraps :meth:`MxRefitReceiver.receive_weights_scratch`. Use this
        when the caller has no pre-registered model parameters to
        receive into — e.g. cold-start in a vLLM worker before
        ``model.load_weights()``, or the benchmark harness. Yielded
        tensors are short-lived scratch buffers; copy them out or feed
        them through ``load_weights`` before the next call.
        """
        yield from self._receiver.receive_weights_scratch(
            candidate.ref,
            timeout_seconds=timeout_seconds,
            tensor_shapes=tensor_shapes,
        )

    def receive_via_plan(
        self,
        plan: "SliceCoveragePlan",
        *,
        timeout_seconds: float = 300.0,
        tensor_shapes: dict[str, tuple[int, ...]] | None = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Multi-source receive driven by a :class:`SliceCoveragePlan` (Phase 4).

        For each candidate in the plan, issue a scratch RDMA receive and
        copy the publisher's bytes into the receiver-local slice given by
        ``SliceSource.dst_range``. Yields one ``(name, tensor)`` per tensor
        in the plan, where ``tensor`` is the stitched view of the
        receiver's requested slice.

        **v0 caveats** (intentional, see post-#2389 RFC §5):

        1. We issue one full ``receive_weights_scratch`` per contributing
           candidate. If the publisher's local shard is larger than the
           receiver's slice (the common case for trainer-TP=4 → inference-TP=8),
           this transfers more bytes than strictly necessary. Phase 4.5 will
           push a byte-offset/byte-length argument into the NIXL transfer
           manager so we issue partial reads.

        2. We do an in-process ``torch.cat`` along ``shard_axis`` to stitch
           the contributions. For 2 contributions this is one extra D2D copy
           per tensor; for N=4+ it would warrant a fused kernel. v0 doesn't
           bother.

        3. Falls back to single-source :meth:`receive_from` if
           ``plan.legacy_single_source`` is True (no v2 registry on any
           candidate, e.g. talking to a v1-only deployment).
        """
        if not plan.fully_covered:
            raise RuntimeError(
                f"plan is not fully covered; missing={plan.missing}"
            )
        if plan.legacy_single_source:
            if not plan.candidates:
                raise RuntimeError("legacy_single_source plan has no candidates")
            yield from self.receive_from(
                plan.candidates[0], timeout_seconds=timeout_seconds
            )
            return

        # Walk contributing candidates; cache scratch results so we only
        # issue one RDMA pull per candidate even if several tensors share it.
        cand_to_scratch: dict[str, dict[str, torch.Tensor]] = {}
        for tensor_name, contributions in plan.per_tensor_sources.items():
            for src in contributions:
                key = src.candidate.ref.worker_id
                if key in cand_to_scratch:
                    continue
                scratch = {
                    name: tensor
                    for name, tensor in self._receiver.receive_weights_scratch(
                        src.candidate.ref,
                        timeout_seconds=timeout_seconds,
                        tensor_shapes=tensor_shapes,
                    )
                }
                cand_to_scratch[key] = scratch

        for tensor_name, contributions in plan.per_tensor_sources.items():
            # If a single contribution covers the slice, no stitching needed.
            if len(contributions) == 1:
                src = contributions[0]
                buf = cand_to_scratch[src.candidate.ref.worker_id][tensor_name]
                yield tensor_name, _slice_along_axis(
                    buf, src.shard_axis, src.src_range
                )
                continue

            slices = []
            for src in contributions:
                buf = cand_to_scratch[src.candidate.ref.worker_id][tensor_name]
                slices.append(_slice_along_axis(buf, src.shard_axis, src.src_range))
            stitched = torch.cat(slices, dim=contributions[0].shard_axis)
            yield tensor_name, stitched

    def discover_v2_sources_for_slice(
        self,
        *,
        model_name: str,
        target_layout: "TargetTPLayout",
        min_version: int = 0,
        same_rank_only: bool = False,
        include_replicas: bool = True,
        compile_target_filter: set[str] | frozenset[str] | None = None,
        required_compile_metadata: dict[str, object] | None = None,
    ) -> "SliceCoveragePlan":
        """Phase-4 multi-source picker: returns the minimal candidate set covering ``target_layout``.

        This is the entry point for **mixed-TP** receivers. The receiver
        states the slice it wants (``target_layout`` describes its own TP
        world size, this rank's TP rank, and which axis is the shard axis),
        and we walk all v2 candidates to find the smallest set whose union
        of ``local_shard_range`` covers that slice — *per tensor*.

        Unlike :meth:`discover_v2_sources`, ``same_rank_only`` defaults to
        ``False`` here: with mixed-TP the obvious case is "trainer TP=4,
        inference TP=8, so each inference rank pulls from one or two trainer
        ranks", which inherently requires reading across publisher ranks.

        Returns a :class:`SliceCoveragePlan` whose ``per_tensor_sources`` maps
        tensor name → ordered list of ``(candidate, src_range, dst_range)``
        slice descriptors. If any tensor cannot be fully covered, the plan's
        ``missing`` list is non-empty and the caller should error out.
        """
        if not self._initialized:
            raise RuntimeError(
                "call initialize() before discover_v2_sources_for_slice()"
            )

        candidates = self.discover_v2_sources(
            model_name=model_name,
            min_version=min_version,
            same_rank_only=same_rank_only,
            include_replicas=include_replicas,
            compile_target_filter=compile_target_filter,
            required_compile_metadata=required_compile_metadata,
        )

        per_tensor_sources: dict[str, list[SliceSource]] = {}
        missing: list[str] = []
        covered_tensors: set[str] = set()

        # Aggregate candidates by tensor name. A tensor is "covered" when the
        # union of admitted candidates' local_shard_ranges (on the requested
        # shard_axis) contains the target slice.
        all_tensor_names: set[str] = set()
        for cand in candidates:
            if not cand.registry:
                continue
            for td in cand.registry.get("tensors", []):
                if isinstance(td, TensorDescriptorV2):
                    all_tensor_names.add(td.name)

        for name in sorted(all_tensor_names):
            # Per-tensor slice planning. Use the candidate's per-tensor
            # global_shape + placement_kind to decide what THIS receiver needs.
            #
            # Slice math is intentionally simple in v0:
            #   - REPLICATE: any candidate provides the full bytes. Pick freshest.
            #   - SHARD on axis A == target_layout.shard_axis: receiver's slice
            #     is [target_start, target_end) over the global axis-A extent;
            #     we accumulate candidates whose local_shard_range intersects
            #     this slice, clipping each contribution to the wanted range.
            #   - SHARD on a different axis: not handled in v0 — emit a missing
            #     entry with a precise reason so the caller can fall back.
            #   - PARTIAL: not handled in v0.
            chosen = self._plan_tensor_slice(
                name=name,
                candidates=candidates,
                target_layout=target_layout,
            )
            if chosen.contributions:
                per_tensor_sources[name] = chosen.contributions
                covered_tensors.add(name)
            else:
                missing.append(f"{name}: {chosen.reason}")

        # If the registry of every candidate is empty (e.g. transport drop) we
        # should still produce a plan with one default contribution per
        # candidate so single-source receivers behave the same way they used
        # to. Detect that legacy mode.
        legacy_mode = not all_tensor_names and candidates
        if legacy_mode:
            return SliceCoveragePlan(
                candidates=candidates,
                per_tensor_sources={},
                missing=[],
                target_layout=target_layout,
                legacy_single_source=True,
            )

        return SliceCoveragePlan(
            candidates=candidates,
            per_tensor_sources=per_tensor_sources,
            missing=missing,
            target_layout=target_layout,
            legacy_single_source=False,
        )

    @staticmethod
    def _plan_tensor_slice(
        *,
        name: str,
        candidates: list[V2SourceCandidate],
        target_layout: "TargetTPLayout",
    ) -> "_TensorPlan":
        # Find every (candidate, td) pair publishing this tensor.
        published: list[tuple[V2SourceCandidate, TensorDescriptorV2]] = []
        for cand in candidates:
            if not cand.registry:
                continue
            for td in cand.registry.get("tensors", []):
                if isinstance(td, TensorDescriptorV2) and td.name == name:
                    published.append((cand, td))
                    break
        if not published:
            return _TensorPlan(contributions=[], reason="no publishers")

        # All publishers must agree on global shape + dtype + placement kind.
        first_td = published[0][1]
        for _, td in published[1:]:
            if td.global_shape != first_td.global_shape:
                return _TensorPlan(
                    contributions=[],
                    reason=f"shape disagreement {first_td.global_shape} vs {td.global_shape}",
                )

        if first_td.placement_kind == "REPLICATE":
            # Any candidate satisfies. Caller's already sorted candidates by
            # freshness in discover_v2_sources.
            cand, td = published[0]
            return _TensorPlan(
                contributions=[
                    SliceSource(
                        candidate=cand,
                        tensor_name=name,
                        src_range=(0, first_td.global_shape[0])
                        if first_td.global_shape
                        else (0, 0),
                        dst_range=(0, first_td.global_shape[0])
                        if first_td.global_shape
                        else (0, 0),
                        shard_axis=0,
                    )
                ],
                reason="ok-replicate",
            )

        if first_td.placement_kind != "SHARD":
            return _TensorPlan(
                contributions=[],
                reason=f"placement_kind={first_td.placement_kind} not supported in v0",
            )

        if first_td.shard_axis != target_layout.shard_axis:
            return _TensorPlan(
                contributions=[],
                reason=(
                    f"shard_axis mismatch: publisher={first_td.shard_axis} "
                    f"target={target_layout.shard_axis}"
                ),
            )

        axis_total = first_td.global_shape[target_layout.shard_axis]
        # Receiver's wanted slice. Even split for v0; uneven splits are
        # handled by the caller passing a custom ``target_layout`` that
        # already encodes the start/end pair.
        if target_layout.target_range is not None:
            t_start, t_end = target_layout.target_range
        else:
            # Reject silent truncation on non-divisible default slicing.
            # Without these guards `axis_total // world_size` would drop
            # the remainder and the plan could appear "covered" while the
            # last few elements of the axis are never requested.
            if target_layout.world_size <= 0:
                return _TensorPlan(
                    contributions=[],
                    reason="target world_size must be positive",
                )
            if not 0 <= target_layout.rank < target_layout.world_size:
                return _TensorPlan(
                    contributions=[],
                    reason=(
                        f"target rank {target_layout.rank} outside "
                        f"world_size {target_layout.world_size}"
                    ),
                )
            if axis_total % target_layout.world_size != 0:
                return _TensorPlan(
                    contributions=[],
                    reason=(
                        f"axis extent {axis_total} is not divisible by "
                        f"target world_size {target_layout.world_size}; "
                        "pass target_range for uneven layouts"
                    ),
                )
            chunk = axis_total // target_layout.world_size
            t_start = target_layout.rank * chunk
            t_end = t_start + chunk

        # Walk publishers in freshness order, accumulate intersections.
        contributions: list[SliceSource] = []
        covered_until = t_start
        # Sort by start of local_shard_range so we accumulate left-to-right.
        published_sorted = sorted(
            published,
            key=lambda pair: (
                pair[1].local_shard_range[0] if pair[1].local_shard_range else 0
            ),
        )
        for cand, td in published_sorted:
            if td.local_shard_range is None:
                continue
            p_start, p_end = td.local_shard_range
            inter_start = max(p_start, t_start)
            inter_end = min(p_end, t_end)
            if inter_start >= inter_end:
                continue
            if inter_start > covered_until:
                # gap — coverage incomplete.
                return _TensorPlan(
                    contributions=[],
                    reason=(
                        f"coverage gap at axis {target_layout.shard_axis} "
                        f"[{covered_until}, {inter_start})"
                    ),
                )
            contributions.append(
                SliceSource(
                    candidate=cand,
                    tensor_name=name,
                    src_range=(inter_start - p_start, inter_end - p_start),
                    dst_range=(inter_start - t_start, inter_end - t_start),
                    shard_axis=target_layout.shard_axis,
                )
            )
            covered_until = inter_end
            if covered_until >= t_end:
                break
        if covered_until < t_end:
            return _TensorPlan(
                contributions=[],
                reason=(
                    f"coverage gap at axis {target_layout.shard_axis} "
                    f"[{covered_until}, {t_end})"
                ),
            )
        return _TensorPlan(contributions=contributions, reason="ok-shard")

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

        # Build a lightweight identity declaring ourselves as a replica.
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

        # Build a tensor-descriptor list from our already-registered buffers.
        from .types import TensorDescriptor

        descriptors = nixl.tensor_descriptors  # already populated at register time
        worker_meta = p2p_pb2.WorkerMetadata(
            worker_rank=self._worker_rank,
            nixl_metadata=nixl.nixl_metadata,
            tensors=[
                p2p_pb2.TensorDescriptor(
                    name=d.name,
                    addr=d.addr,
                    size=d.size,
                    device_id=d.device_id,
                    dtype=d.dtype,
                )
                for d in descriptors
            ],
            status=p2p_pb2.SOURCE_STATUS_READY,
            agent_name=self._receiver._agent_name,
        )

        try:
            mx_source_id = client.publish_metadata(
                identity=identity,
                worker=worker_meta,
                worker_id=self._receiver._worker_id
                if hasattr(self._receiver, "_worker_id")
                else "",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "publish_self_as_source failed: %s", e, exc_info=True
            )
            return None
        logger.info(
            "Published self as inference_replica: rank=%d version=%d mx_source_id=%s",
            self._worker_rank,
            version,
            mx_source_id,
        )
        return mx_source_id

    def shutdown(self) -> None:
        self._receiver.shutdown()
        self._initialized = False


__all__ = [
    "MxV2RefitReceiver",
    "MxV2TrainingPublisher",
    "ROLE_INFERENCE",
    "ROLE_INFERENCE_REPLICA",
    "ROLE_TRAINER",
    "TrainerWorldLayout",
    "V2SourceCandidate",
]
