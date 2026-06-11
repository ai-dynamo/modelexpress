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
import random
from dataclasses import dataclass
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


def parse_v2_extras(meta) -> dict[str, str]:
    """Extract v2 extra_parameters from a ``GetMetadataResponse``.

    Tries three transports in order — `SourceIdentity.extra_parameters`
    (clean path), `TensorDescriptor` sidecar named `__mx_v2_meta__`
    (Rust-server-echo workaround), `WorkerMetadata.agent_name` legacy
    string encoding. Returns the merged extras dict, or an empty dict
    when the source is not v2.

    Shared between :meth:`MxV2RefitReceiver.discover_v2_sources` and the
    initial-load discovery in :mod:`modelexpress.load_strategy.rdma_strategy`.
    """
    identity = getattr(meta, "identity", None)
    extra: dict[str, str] = (
        dict(identity.extra_parameters)
        if identity is not None and identity.extra_parameters
        else {}
    )
    if not extra:
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
    return extra


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
        """Mark this source as READY and (re)arm the liveness heartbeat.

        Each :meth:`publish` mints a NEW ``mx_source_id`` (the training step is
        encoded into the agent_name), so the heartbeat — which refreshes a
        single snapshotted ``mx_source_id`` — must be re-armed onto the source
        we just published. Without re-arming, only the *first* published version
        stays heartbeated; every later version is marked READY once and then
        reaped to STALE by the server after ``MX_HEARTBEAT_TIMEOUT_SECS``
        (default 90s), so receivers' :meth:`discover_v2_sources` find no source
        for ``version>=N`` on the second and subsequent refits. (Mirrors the
        stop-then-restart logic already used in :meth:`publish_self_as_source`.)
        """
        ok = self._publisher.mark_ready(worker_rank=self._worker_rank)
        if ok and self._heartbeat_enabled:
            # Restart the heartbeat onto the source we just published. Stop the
            # prior one first (it tracked a now-superseded source_id) to avoid a
            # leaked thread and a lingering READY status on the old version.
            if self._heartbeat is not None:
                self._heartbeat.stop()
                self._heartbeat = None
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
    """A discovered source with v2 metadata parsed."""

    ref: SourceRef
    role: str  # "trainer" | "inference_replica"
    worker_rank: int
    registry: dict | None  # decoded registry; None for inference_replica
    owned_experts_per_layer: dict[int, set[int]]  # layer_idx → expert IDs
    updated_at: int  # ms epoch


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
        # Heartbeat for the self-publish so the server doesn't mark our
        # inference_replica source STALE between refit cycles. Started on
        # the first successful publish_self_as_source call; stopped on
        # shutdown. The worker_id is sourced from the inner receiver's
        # _worker_id (assigned in MxRefitReceiver.__init__; see PR #421).
        self._self_publish_heartbeat: HeartbeatThread | None = None

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

            # Read v2 metadata via the shared parser (handles the three
            # transports: identity.extra_parameters, sidecar tensor, and
            # agent_name legacy encoding).
            extra = parse_v2_extras(meta)
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

        Tree fan-out preference: when inference_replicas exist at the
        requested version, prefer them over the trainer and random-pick
        among them. The trainer is reserved as a fallback for the first
        wave (no replicas yet exist). This spreads load across the
        replicas' NICs so a sequential or wave-parallel dispatcher
        actually benefits from tree fan-out.

        Stale-source filter: MX server persists source registrations in
        Redis with no TTL (TTL=-1), so dead-pod sources from prior runs
        linger in ``list_sources(status_filter=READY)``. Filter to
        candidates whose ``updated_at`` (heartbeat) is within the last
        ~60s; fall back to the unfiltered set only if nothing recent
        remains (defensive).

        If ``needed_experts_per_layer`` is set, the candidate must own a
        superset of the requested experts (or be a trainer with full info).
        """
        if not candidates:
            return None

        import time as _time
        now_ms = int(_time.time() * 1000)
        FRESH_MS = 60_000  # 60s — generous for ~5s heartbeat cadence
        fresh = [c for c in candidates if (now_ms - c.updated_at) < FRESH_MS]
        if fresh:
            candidates = fresh

        replicas = [c for c in candidates if c.role == ROLE_INFERENCE_REPLICA]
        trainers = [c for c in candidates if c.role == ROLE_TRAINER]

        if needed_experts_per_layer is None:
            # Dense / EP=1: any same-rank source carries the full state.
            # Prefer a replica (random across them) so we fan-out load.
            if replicas:
                return random.choice(replicas)
            if trainers:
                return trainers[0]
            return candidates[0]

        # MoE / EP>1: candidate must cover our needed experts. Trainer is
        # always full-info; replicas may be partial. Prefer covering
        # replicas first, fall back to trainer.
        covering_replicas = [
            c
            for c in replicas
            if all(
                needed.issubset(c.owned_experts_per_layer.get(layer, set()))
                for layer, needed in needed_experts_per_layer.items()
            )
        ]
        if covering_replicas:
            return random.choice(covering_replicas)
        if trainers:
            return trainers[0]
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

        # Start (or restart) a heartbeat so the server doesn't mark this
        # source STALE after the heartbeat-detection timeout. Without
        # this, subsequent autoscaled pods would only see the trainer
        # (or other heartbeat-backed sources) and skip the
        # inference_replica fan-out path entirely.
        try:
            if self._self_publish_heartbeat is not None:
                self._self_publish_heartbeat.stop()
                self._self_publish_heartbeat = None
            self._self_publish_heartbeat = HeartbeatThread(
                mx_client=client,
                mx_source_id=mx_source_id,
                worker_id=worker_id,
                worker_rank=self._worker_rank,
                nixl_manager=nixl,
            )
            self._self_publish_heartbeat.start()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "publish_self_as_source: heartbeat start failed: %s",
                e,
                exc_info=True,
            )

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
        if self._self_publish_heartbeat is not None:
            try:
                self._self_publish_heartbeat.stop()
            except Exception:  # noqa: BLE001
                pass
            self._self_publish_heartbeat = None
        self._initialized = False


__all__ = [
    "MxV2RefitReceiver",
    "MxV2TrainingPublisher",
    "ROLE_INFERENCE",
    "ROLE_INFERENCE_REPLICA",
    "ROLE_TRAINER",
    "TrainerWorldLayout",
    "V2SourceCandidate",
    "parse_v2_extras",
]
