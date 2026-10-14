# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-version staged NIXL transfer for RL generator workers.

This module owns the state that makes a pull transfer correct: the selected
source manifests, logical reads and physical bindings, destination buffers, peer
metadata, transfer completion, and verification. It deliberately does not know
how an inference engine captures its load layout or installs received weights.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import cached_property
from types import MappingProxyType
from typing import Any, NamedTuple, cast

import torch
from modelexpress import envs, p2p_pb2
from modelexpress_rl import envs as rl_envs
from modelexpress.metadata.worker_server import (
    TensorReadLease,
    prepare_tensor_read,
)
from modelexpress.nixl_transfer import (
    NIXL_DRAM_MEM_TYPE,
    NIXL_VRAM_MEM_TYPE,
    NixlTransferManager,
)
from modelexpress.refit.reshard import throughput
from modelexpress.refit.reshard.cuda_pool import classic_cuda_alloc
from modelexpress.refit.reshard.rendezvous import (
    build_sources,
    merge_shard_tables,
    unwrap_rendezvous_blob,
)
from modelexpress.refit.reshard.transfer_plan import (
    LogicalTransferPlan,
    bind_transfer,
    plan_transfer_groups,
    plan_transfer,
)
from modelexpress.refit.reshard.transport import ReadDescriptor
from modelexpress.refit.reshard.transport.nixl import NixlReshardTransport
from modelexpress.refit.reshard.types import (
    CaptureResult,
    IncompleteRefit,
    UnsupportedReshard,
    summarize_unsupported,
)
from modelexpress.refit.reshard.verify import shard_region, tensor_digest
from modelexpress.types import ManifestMismatchError, TensorDescriptor

from modelexpress_rl.inference.plan import TrainerSourceSnapshot

from modelexpress_rl.inference._source_snapshot import (
    _freeze_sources,
)

from ..train.manifest import ShardChecksumKey
from .plan import StreamingSettings

# Named under modelexpress.* (not modelexpress_rl) so the per-update summary surfaces
# in the vLLM engine process, which only configures the modelexpress logger.
logger = logging.getLogger("modelexpress.reshard.staged_transfer")


@dataclass(frozen=True)
class _ResolvedSources:
    sources: dict
    session_to_agent: dict
    session_to_device: dict
    agent_metadata: dict[str, bytes]
    session_to_memory: dict[str, str] = field(default_factory=dict)
    manifest_agents: tuple[str, ...] = ()

    @cached_property
    def shard_bindings(self) -> dict:
        return {
            (
                self.session_to_agent[shard.session],
                name,
                tuple(shard.shard_offset),
                tuple(shard.shape),
            ): shard
            for name, source in self.sources.items()
            for shard in source.shards
        }


@dataclass(frozen=True)
class _SourceBinding:
    resolved: _ResolvedSources
    agents_by_slot: dict[str, str]
    physical_fingerprint: tuple

    def resolve(self, buffer) -> tuple[str, int]:
        agent = self.agents_by_slot[buffer.source_slot_id]
        shard = self.resolved.shard_bindings[
            (agent, buffer.src_name, buffer.shard_offset, buffer.shape)
        ]
        return shard.session, shard.addr


@dataclass(frozen=True)
class _WeightUpdatePlan:
    trainer_source_snapshot: TrainerSourceSnapshot
    layout_capture_snapshot: CaptureResult
    parameter_layout: MappingProxyType
    transfer_plan: LogicalTransferPlan | _StreamingSchedule
    source_binding: _SourceBinding
    transport: NixlReshardTransport


@dataclass(frozen=True)
class _PreparedNixlTransfer:
    """One bound logical plan over reusable registered destinations."""

    transfer_plan: LogicalTransferPlan
    capture: CaptureResult
    sources: dict
    descriptors: tuple[ReadDescriptor, ...]
    transport: NixlReshardTransport
    source_binding: _SourceBinding
    metrics: dict[str, float] = field(default_factory=dict)
    checksums: Mapping[ShardChecksumKey, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _StagedNixlWeights:
    """Verified tensors ready for engine installation."""

    tensors: dict[str, torch.Tensor]
    metrics: dict[str, Any]


_StagingLayout = dict[str, tuple[tuple[int, ...], torch.dtype]]
_CaptureLayout = Callable[
    [list[tuple[str, torch.dtype, tuple[int, ...]]]],
    tuple[CaptureResult, _StagingLayout],
]


class _StagingLayouts(NamedTuple):
    """The three typed views one batch carves out of its arena, in arena order."""

    recv: _StagingLayout
    convert: _StagingLayout
    full: _StagingLayout


@dataclass(frozen=True)
class _StreamingBatch:
    capture: CaptureResult
    transfer_plan: LogicalTransferPlan
    layouts: _StagingLayouts
    nbytes: int

    def __post_init__(self) -> None:
        # Readers reach these views both by name and by position, and
        # dataclasses.replace or a plain 3-tuple would satisfy only the second.
        # Coerce so the annotation holds however the batch was built.
        if not isinstance(self.layouts, _StagingLayouts):
            object.__setattr__(self, "layouts", _StagingLayouts(*self.layouts))


@dataclass(frozen=True)
class _PreparedBoundedTransfer:
    batches: tuple[_StreamingBatch, ...]
    sources: dict
    transport: NixlReshardTransport
    source_binding: _SourceBinding
    metrics: dict[str, float] = field(default_factory=dict)
    checksums: Mapping[ShardChecksumKey, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _StreamingSchedule:
    transfer_plan: LogicalTransferPlan
    module_batches: tuple[_StreamingBatch, ...]
    batches: tuple[_StreamingBatch, ...]


def _bounded_batches(
    capture: CaptureResult,
    parameter_layout: _StagingLayout,
    sources: dict,
    max_staging_bytes: int,
    *,
    complete_plan=None,
    metrics=None,
    total_staging_bytes=None,
    staging_buffers=1,
    source_slots=None,
):
    """Validate all owning-module batches before allocating or installing."""
    if (
        isinstance(max_staging_bytes, bool)
        or not isinstance(max_staging_bytes, int)
        or max_staging_bytes <= 0
    ):
        raise ValueError("max_staging_bytes must be a positive integer")
    if metrics is None:
        metrics = {}
    started = time.perf_counter()
    complete = complete_plan
    if complete is None:
        complete = plan_transfer(capture, sources, source_slots=source_slots,
                                 verify_full_sources=envs.MX_RESHARD_PUBLISH_DIGEST)
    metrics["bounded_whole_plan_s"] = time.perf_counter() - started
    metrics["bounded_whole_plan_builds"] = int(complete_plan is None)
    started = time.perf_counter()
    _NixlStagedTransfer._validate_complete(capture, parameter_layout, complete)
    metrics["bounded_whole_validation_s"] = time.perf_counter() - started
    if {copy.param_name for copy in capture.copies} - parameter_layout.keys():
        raise IncompleteRefit("bounded capture references unknown engine parameters")
    groups = {}
    for name in parameter_layout:
        groups.setdefault(name.rpartition(".")[0], {})[name] = parameter_layout[name]
    batches = []
    metrics["owner_plan_s"] = 0.0
    metrics["owner_validation_s"] = 0.0
    metrics["owner_plan_builds"] = 0
    subsets = tuple(
        CaptureResult(copies=[c for c in capture.copies if c.param_name in recv])
        for recv in groups.values()
    )
    plans = plan_transfer_groups(subsets, sources, complete)
    for (module, recv), subset in zip(groups.items(), subsets, strict=True):
        started = time.perf_counter()
        plan = next(plans)
        metrics["owner_plan_s"] += time.perf_counter() - started
        metrics["owner_plan_builds"] += 1
        started = time.perf_counter()
        _NixlStagedTransfer._validate_complete(subset, recv, plan)
        metrics["owner_validation_s"] += time.perf_counter() - started
        convert = {
            c.param_name: (tuple(c.dest_shape), c.src_dtype) for c in plan.converts
        }
        full = {f.src_name: (tuple(f.global_shape), f.dtype) for f in plan.full_pulls}
        layouts = _StagingLayouts(recv, convert, full)
        # Each typed view begins at a 256-byte boundary in one registered arena.
        nbytes = sum(
            ((math.prod(shape) * dtype.itemsize + 255) // 256) * 256
            for layout in layouts
            for shape, dtype in layout.values()
        )
        if nbytes > max_staging_bytes:
            if total_staging_bytes is not None and staging_buffers > 1:
                budget = (
                    f"max_staging_bytes={total_staging_bytes} split across "
                    f"staging_buffers={staging_buffers} gives {max_staging_bytes} "
                    "bytes per arena"
                )
                remedy = "lower staging_buffers or raise max_staging_bytes"
            else:
                budget = f"max_staging_bytes={max_staging_bytes}"
                remedy = "raise max_staging_bytes"
            raise IncompleteRefit(
                f"module {module!r} requires {nbytes} staging bytes, exceeds "
                f"{budget}; {remedy} (there is no CPU fallback)"
            )
        batches.append(_StreamingBatch(subset, plan, layouts, nbytes))
    if not batches:
        raise IncompleteRefit("bounded refit has no engine parameters")
    return tuple(batches)


def _pack_bounded_batches(
    batches: tuple[_StreamingBatch, ...], max_staging_bytes: int
) -> tuple[_StreamingBatch, ...]:
    """Pack complete modules without changing source READ ranges.

    Owning-module batches are the unit of correctness; this only coalesces
    neighbours that fit the same arena together, so each packed batch still
    installs whole modules and reads exactly the bytes the unpacked plan read.
    """

    def merge(group: list[_StreamingBatch]) -> _StreamingBatch:
        capture = CaptureResult(
            copies=[copy for batch in group for copy in batch.capture.copies]
        )
        plan = replace(
            group[0].transfer_plan,
            segments=tuple(read for batch in group for read in batch.transfer_plan.segments),
            converts=tuple(item for batch in group for item in batch.transfer_plan.converts),
            full_pulls=tuple(item for batch in group for item in batch.transfer_plan.full_pulls),
            fallback=tuple(dict.fromkeys(name for batch in group for name in batch.transfer_plan.fallback)),
            unbounded_sources=tuple(
                name for batch in group for name in batch.transfer_plan.unbounded_sources
            ),
            exact_descriptor_count=sum(batch.transfer_plan.exact_descriptor_count for batch in group),
            exact_bytes=sum(batch.transfer_plan.exact_bytes for batch in group),
        )
        layouts = _StagingLayouts({}, {}, {})
        for batch in group:
            for layout, incoming in zip(layouts, batch.layouts, strict=True):
                if layout.keys() & incoming.keys():
                    raise IncompleteRefit(
                        "packed batches contain overlapping staging keys"
                    )
                layout.update(incoming)
        return _StreamingBatch(
            capture, plan, layouts, sum(batch.nbytes for batch in group)
        )

    packed = []
    current = []
    current_bytes = 0
    full_sources = set()
    for batch in batches:
        if batch.nbytes > max_staging_bytes:
            raise IncompleteRefit("owning module exceeds the packed staging budget")
        incoming_full = set(batch.layouts.full)
        # Two modules pulling the same complete source would need one staging
        # slot for two distinct writes, so they must stay in separate batches.
        if current and (
            current_bytes + batch.nbytes > max_staging_bytes
            or full_sources & incoming_full
        ):
            packed.append(merge(current))
            current = []
            current_bytes = 0
            full_sources = set()
        current.append(batch)
        current_bytes += batch.nbytes
        full_sources.update(incoming_full)
    if current:
        packed.append(merge(current))
    return tuple(packed)


def _resolve_sources(manifests: list[bytes], *, metrics=None) -> _ResolvedSources:
    if not manifests:
        raise ValueError("at least one source manifest is required")
    if metrics is None:
        metrics = {}
    started = time.perf_counter()
    payloads = [unwrap_rendezvous_blob(manifest) for manifest in manifests]
    agents = [payload.agent_name for payload in payloads]
    if len(set(agents)) != len(agents):
        raise ValueError("source manifests contain duplicate NIXL agents")
    metrics["source_decode_s"] = time.perf_counter() - started
    started = time.perf_counter()
    merged = merge_shard_tables([payload.tensors for payload in payloads])
    metrics["source_merge_s"] = time.perf_counter() - started
    started = time.perf_counter()
    session_to_memory = {}
    sources, session_to_agent, session_to_device = build_sources(
        merged, session_to_memory=session_to_memory
    )
    for source in sources.values():
        for shard in source.shards:
            shard.digest = None
    metrics["source_build_s"] = time.perf_counter() - started
    return _ResolvedSources(
        sources=sources,
        session_to_agent=session_to_agent,
        session_to_device=session_to_device,
        session_to_memory=session_to_memory,
        manifest_agents=tuple(agents),
        agent_metadata={
            payload.agent_name: payload.agent_metadata for payload in payloads
        },
    )


def _required_agent_metadata(
    plan: LogicalTransferPlan,
    resolved: _ResolvedSources,
    source_binding: _SourceBinding,
) -> dict[str, bytes]:
    slots = {plan.source_buffers[segment.source_buffer].source_slot_id
             for segment in plan._all_segments()}
    missing_slots = sorted(slots - source_binding.agents_by_slot.keys())
    if missing_slots:
        raise RuntimeError(
            f"transfer plan references unknown source slots: {missing_slots[:10]}"
        )
    needed = {source_binding.agents_by_slot[slot] for slot in slots}
    missing = sorted(needed - set(resolved.agent_metadata))
    if missing:
        raise RuntimeError(
            "transfer plan references source agents without NIXL metadata: "
            f"{missing[:10]}"
        )
    return {
        agent: metadata
        for agent, metadata in resolved.agent_metadata.items()
        if agent in needed
    }


def _load_agent_metadata(
    manager: NixlTransferManager, metadata_by_agent: dict[str, bytes]
) -> None:
    """Load the exact source registrations carried by the version manifests."""
    for expected_agent, metadata in metadata_by_agent.items():
        loaded_agent = manager.add_remote_agent(metadata)
        if isinstance(loaded_agent, bytes):
            loaded_agent = loaded_agent.decode("utf-8")
        if loaded_agent != expected_agent:
            raise RuntimeError(
                "NIXL metadata agent does not match its manifest: "
                f"expected {expected_agent!r}, got {loaded_agent!r}"
            )


def _replay_ops(tensor: torch.Tensor, op_chain: tuple) -> torch.Tensor:
    value = tensor
    for op_name, args, frozen_kwargs in op_chain:
        kwargs = dict(frozen_kwargs)
        if op_name == "__getitem__":
            value = value.__getitem__(*args)
        else:
            value = getattr(value, op_name)(*args, **kwargs)
    return value


class _NixlStagedTransfer:
    """Own the complete prepare-and-stage lifecycle for one generator rank."""

    def __init__(
        self,
        *,
        device_id: int,
        device: torch.device,
        capture_layout: _CaptureLayout,
        generator_fingerprint: Callable[[], tuple],
        agent_name: str | None = None,
        listen_port: int | None = None,
        timeout_seconds: float | None = None,
        manager: NixlTransferManager | None = None,
        streaming: StreamingSettings | None = None,
    ) -> None:
        if streaming is not None and device.type != "cuda":
            raise ValueError("bounded NIXL staging requires a CUDA device")
        if not callable(capture_layout):
            raise TypeError("capture_layout must be callable")
        if not callable(generator_fingerprint):
            raise ValueError("generator_fingerprint must be callable")
        self._capture_layout = capture_layout
        self._generator_fingerprint = generator_fingerprint
        self._streaming = streaming
        self._buffer_budget = (
            None
            if streaming is None
            else streaming.max_staging_bytes // streaming.staging_buffers
        )
        self._device_id = device_id
        self._device = device
        self._timeout = float(
            envs.MX_TRANSFER_TIMEOUT if timeout_seconds is None else timeout_seconds
        )
        self._owns_manager = manager is None
        if manager is None:
            if agent_name is None:
                raise ValueError("an owned NIXL manager requires an agent name")
            manager = NixlTransferManager(
                agent_name=agent_name,
                device_id=device_id,
                listen_port=listen_port,
            )
        self._manager = manager
        if self._owns_manager:
            try:
                self._manager.initialize()
            except Exception:
                self._manager.shutdown()
                raise
        # Canonical engine-layout staging buffers. Exact slices land directly
        # here; reconstructed or converted values are copied here before these
        # buffers are verified, installed, and advertised to peer generators.
        self._recv_buffers: dict[str, torch.Tensor] = {}
        # Wire-dtype staging for sources whose dtype differs from the engine
        # parameter. RDMA writes here first, then stage() casts into recv buffers.
        self._convert_buffers: dict[str, torch.Tensor] = {}
        # Complete contiguous source tensors used when captured transforms must
        # be replayed locally, or when direct slicing exceeds the descriptor
        # budget. stage() reconstructs each source here, then copies its derived
        # views into the canonical receive buffers.
        self._full_buffers: dict[str, torch.Tensor] = {}
        self._registered_recv_params: set[str] = set()
        self._convert_registered = False
        self._full_registered = False
        self._active: _PreparedNixlTransfer | _PreparedBoundedTransfer | None = None
        self._loaded_agent_metadata: dict[str, bytes] = {}
        self._closed = False
        # Bounded staging: one or two byte arenas on CUDA or pinned host memory.
        # Host arenas are registered as NIXL DRAM and tracked here so they can be
        # deregistered before the agent shuts down.
        self._staging_arenas: list[torch.Tensor] = []
        self._staging_registrations: list[Any] = []
        self._staging_device: torch.device | None = None
        self._manager_ready = True
        self._weight_update_plan: _WeightUpdatePlan | None = None
        self._full_copy_descriptors: tuple[ReadDescriptor, ...] | None = None
        self._debug_validate_plan = rl_envs.MX_REFIT_DEBUG_VALIDATE_PLAN
        self._debug_validate_generator_layout = (
            rl_envs.MX_REFIT_DEBUG_VALIDATE_GENERATOR_LAYOUT
        )
        self._pack_modules = envs.MX_REFIT_PACK_MODULES
        self._plan_compile_lock = threading.Lock()
        self._descriptor_cache: list[tuple[ReadDescriptor, ...] | None] | None = None

    @property
    def bounded_staging(self) -> bool:
        return self._streaming is not None

    @property
    def _bounded_arena(self) -> torch.Tensor | None:
        return self._staging_arenas[0] if self._staging_arenas else None

    @_bounded_arena.setter
    def _bounded_arena(self, value: torch.Tensor | None) -> None:
        self._invalidate_descriptors()
        self._staging_arenas = [] if value is None else [value]

    def _invalidate_descriptors(self) -> None:
        self._descriptor_cache = None

    def _release_staging_registrations(self) -> None:
        self._invalidate_descriptors()
        registrations, self._staging_registrations = self._staging_registrations, []
        for registration in registrations:
            self._manager.deregister_memory(registration)

    def _allocate_arena(self, nbytes: int) -> torch.Tensor:
        assert self._staging_device is not None
        if self._staging_device.type == "cpu":
            # Pinned so the NIC can register it and the H2D commit is a DMA.
            return torch.empty(
                nbytes, dtype=torch.uint8, pin_memory=torch.cuda.is_available()
            )
        with classic_cuda_alloc():
            return torch.empty(nbytes, dtype=torch.uint8, device=self._device)

    def reset_workspace(self) -> None:
        """Discard released or failed preparation after disconnecting its agent."""
        self._invalidate_descriptors()
        if not self._owns_manager:
            # A shared agent still serves its owner; tearing it down here would
            # deregister memory we do not own. Only a transfer-owned agent can
            # be cycled to release staging storage safely.
            raise RuntimeError(
                "resetting staging storage requires a transfer-owned NIXL agent; "
                "restart the generator engine"
            )
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        self._release_staging_registrations()
        self._manager.shutdown()
        self._active = None
        self._recv_buffers.clear()
        self._convert_buffers.clear()
        self._full_buffers.clear()
        self._staging_arenas.clear()
        self._staging_device = None
        self._registered_recv_params.clear()
        self._convert_registered = False
        self._full_registered = False
        self._loaded_agent_metadata.clear()
        self._manager_ready = False
        self._weight_update_plan = None
        self._full_copy_descriptors = None

    def _ensure_manager_initialized(self) -> None:
        if not self._manager_ready and self._owns_manager:
            self._native_setup_started = True
            try:
                self._manager.initialize()
            except Exception:
                self._manager.shutdown()
                raise
            self._manager_ready = True

    def _resolve_metadata(
        self, manifests: list[bytes], metrics: dict
    ) -> _ResolvedSources:
        if any(not isinstance(blob, bytes) for blob in manifests):
            raise TypeError("cached source manifests must be immutable bytes")
        metrics.update(
            source_cache_enabled=1,
            source_cache_hits=0,
            source_cache_misses=0,
            source_cache_lookup_s=0.0,
            source_decode_s=0.0,
            source_merge_s=0.0,
            source_build_s=0.0,
            source_manifest_bytes=sum(len(blob) for blob in manifests),
        )
        started = time.perf_counter()
        previous = self._weight_update_plan
        hit = previous is not None and tuple(source.manifest for source in previous.trainer_source_snapshot.shards) == tuple(manifests)
        metrics["source_cache_lookup_s"] = time.perf_counter() - started
        metrics["source_cache_hits"] = int(hit)
        metrics["source_cache_misses"] = int(not hit)
        if hit:
            return previous.trainer_source_snapshot.resolved_metadata
        resolved = _resolve_sources(manifests, metrics=metrics)
        started = time.perf_counter()
        frozen_sources = _freeze_sources(resolved.sources)
        metrics["source_build_s"] += time.perf_counter() - started
        return replace(
            resolved,
            sources=frozen_sources if frozen_sources is not None else resolved.sources,
            session_to_agent=MappingProxyType(resolved.session_to_agent),
            session_to_device=MappingProxyType(resolved.session_to_device),
            session_to_memory=MappingProxyType(resolved.session_to_memory),
            agent_metadata=MappingProxyType(resolved.agent_metadata),
        )


    def _resolve_layout(
        self,
        manifests: list[bytes],
        metrics: dict[str, float],
        trainer_snapshot: TrainerSourceSnapshot,
    ) -> tuple[TrainerSourceSnapshot, CaptureResult, _StagingLayout]:
        started = time.perf_counter()
        resolved = self._resolve_metadata(manifests, metrics)
        trainer = replace(trainer_snapshot, resolved_metadata=resolved)
        source_schema = tuple(
            (name, source.dtype, tuple(source.global_shape))
            for name, source in sorted(resolved.sources.items())
        )
        metrics["source_metadata_s"] = time.perf_counter() - started
        started = time.perf_counter()
        capture, parameter_layout = self._capture_layout(list(source_schema))
        metrics["layout_capture_s"] = time.perf_counter() - started
        return trainer, capture, parameter_layout

    @staticmethod
    def _source_binding(trainer: TrainerSourceSnapshot) -> _SourceBinding:
        resolved = trainer.resolved_metadata
        slots = tuple(shard.source_slot_id for shard in trainer.shards)
        agents = resolved.manifest_agents
        if len(slots) != len(agents) or len(set(slots)) != len(slots):
            raise ValueError("source slots must identify each selected replica once")
        return _SourceBinding(
            resolved, dict(zip(slots, agents, strict=True)), trainer.physical_fingerprint
        )

    def cached_trainer_source(self) -> TrainerSourceSnapshot | None:
        plan = self._weight_update_plan
        if plan is None or self._closed:
            return None
        return plan.trainer_source_snapshot

    def _can_reuse_plan(self, trainer: TrainerSourceSnapshot) -> bool:
        previous = self._weight_update_plan
        return (
            previous is not None
            and (
                trainer is previous.trainer_source_snapshot
                or trainer.physical_fingerprint
                == previous.trainer_source_snapshot.physical_fingerprint
            )
        )

    def _compile_streaming(
        self,
        trainer: TrainerSourceSnapshot,
        capture: CaptureResult,
        parameter_layout: Mapping,
        metrics: dict,
        binding: _SourceBinding,
    ) -> _StreamingSchedule:
        if not self._plan_compile_lock.acquire(blocking=False):
            raise RuntimeError("bounded plan compilation is already in progress")
        try:
            metrics.update(
                initial_whole_plan_s=0.0,
                initial_whole_validation_s=0.0,
                bounded_whole_plan_s=0.0,
                bounded_whole_plan_builds=0,
                bounded_whole_validation_s=0.0,
                owner_plan_s=0.0,
                owner_validation_s=0.0,
                owner_plan_builds=0,
            )
            started = time.perf_counter()
            slots_by_agent = {agent: slot for slot, agent in binding.agents_by_slot.items()}
            plan = plan_transfer(
                capture, trainer.resolved_metadata.sources,
                source_slots={
                    session: slots_by_agent[agent]
                    for session, agent in trainer.resolved_metadata.session_to_agent.items()
                },
                verify_full_sources=envs.MX_RESHARD_PUBLISH_DIGEST,
            )
            metrics["initial_whole_plan_s"] = time.perf_counter() - started
            modules = _bounded_batches(
                capture,
                parameter_layout,
                trainer.resolved_metadata.sources,
                self._buffer_budget,
                complete_plan=plan,
                metrics=metrics,
                total_staging_bytes=self._streaming.max_staging_bytes,
                staging_buffers=self._streaming.staging_buffers,
            )
            batches = (
                _pack_bounded_batches(modules, self._buffer_budget)
                if self._pack_modules else modules
            )
            compiled = _StreamingSchedule(plan, modules, batches)
            logger.info("compiled bounded transfer plan")
            return compiled

        finally:
            self._plan_compile_lock.release()

    def _connect_sources(
        self,
        resolved: _ResolvedSources,
        required_metadata: dict[str, bytes],
        *,
        host_staging: bool = False,
    ) -> NixlReshardTransport:
        self._ensure_manager_initialized()
        changed = {
            agent: metadata
            for agent, metadata in required_metadata.items()
            if self._loaded_agent_metadata.get(agent) != metadata
        }
        conflicting = sorted(
            agent for agent in changed if agent in self._loaded_agent_metadata
        )
        if changed:
            self._native_setup_started = True
        if conflicting:
            raise RuntimeError(
                "NIXL metadata changed for an already connected source agent: "
                f"{conflicting[:10]}"
            )
        _load_agent_metadata(self._manager, changed)
        self._loaded_agent_metadata.update(changed)
        return NixlReshardTransport(
            self._manager,
            resolved.session_to_agent,
            resolved.session_to_device,
            timeout_seconds=self._timeout,
            local_mem_type=NIXL_DRAM_MEM_TYPE if host_staging else NIXL_VRAM_MEM_TYPE,
            session_to_memory=resolved.session_to_memory,
        )

    def _invalidate_cached_transfer(self) -> None:
        self._weight_update_plan = None
        self._full_copy_descriptors = None
        self._descriptor_cache = None

    @contextmanager
    def _preparing(self) -> Iterator[None]:
        if self._closed:
            raise RuntimeError("NIXL staged transfer is closed")
        self._native_setup_started = False
        try:
            yield
        except Exception:
            self._invalidate_cached_transfer()
            if self._native_setup_started:
                try:
                    self.reset_workspace()
                except Exception as cleanup_error:
                    mode = "streaming" if self._streaming is not None else "full-copy"
                    raise ValueError(
                        f"failed to reset {mode} preparation; restart the generator engine"
                    ) from cleanup_error
            raise
        finally:
            self._native_setup_started = False

    def prepare(
        self,
        *,
        manifests: list[bytes],
        trainer_snapshot: TrainerSourceSnapshot,
        checksums: Mapping[ShardChecksumKey, str] | None = None,
    ) -> _PreparedNixlTransfer | _PreparedBoundedTransfer:
        """Prepare one version-scoped transfer using this instance's staging mode."""
        streaming = self._streaming
        if checksums is None:
            checksums = MappingProxyType({})
        with self._preparing():
            previous = self._weight_update_plan
            metrics = {}
            lookup_started = time.perf_counter()
            reusable = self._can_reuse_plan(trainer_snapshot)
            metrics.update(
                plan_cache_enabled=1,
                plan_cache_hits=int(reusable),
                plan_cache_misses=int(not reusable),
                plan_cache_lookup_s=time.perf_counter() - lookup_started,
                plan_cache_validate_s=0.0,
            )
            started = time.perf_counter()
            if reusable:
                trainer = previous.trainer_source_snapshot
                if trainer_snapshot.shards != trainer.shards:
                    trainer = replace(trainer_snapshot, resolved_metadata=trainer.resolved_metadata)
                capture = previous.layout_capture_snapshot
                parameter_layout = previous.parameter_layout
                source_binding = previous.source_binding
                transfer_plan = previous.transfer_plan
                if self._debug_validate_generator_layout:
                    assert (
                        self._generator_fingerprint() == capture.generator_fingerprint
                    ), "generator layout changed since capture"
                if self._debug_validate_plan:
                    self._validate_complete(
                        capture, dict(parameter_layout), transfer_plan, metrics
                    )
            else:
                trainer, capture, parameter_layout = self._resolve_layout(
                    manifests, metrics, trainer_snapshot
                )
                source_binding = self._source_binding(trainer)
                started = time.perf_counter()
                if streaming is None:
                    transfer_plan = self._compile_full_copy(
                        trainer, capture, parameter_layout, metrics, source_binding
                    )
                else:
                    assert self._buffer_budget is not None
                    transfer_plan = self._compile_streaming(
                        trainer, capture, parameter_layout, metrics, source_binding
                    )
            metrics["transfer_planning_s"] = time.perf_counter() - started
            if reusable:
                cached = previous if trainer is previous.trainer_source_snapshot else replace(
                    previous, trainer_source_snapshot=trainer
                )
                transport = cached.transport
                if streaming is not None:
                    metrics["connection_registration_s"] = 0.0
            else:
                transport = self._prepare_transport(
                    trainer, transfer_plan, source_binding, metrics
                )
                cached = _WeightUpdatePlan(
                    trainer,
                    capture,
                    parameter_layout,
                    transfer_plan,
                    source_binding,
                    transport,
                )
            if streaming is None:
                prepared = self._prepare_full_copy(
                    cached,
                    transport,
                    metrics,
                    checksums,
                    reusable,
                )
            else:
                prepared = self._prepare_streaming(
                    cached,
                    transport,
                    metrics,
                    checksums,
                    reusable,
                )
            self._weight_update_plan = cached
            self._active = prepared
            return prepared

    def _prepare_transport(self, trainer, transfer_plan, source_binding, metrics):
        streaming = self._streaming
        compiled = transfer_plan if isinstance(transfer_plan, _StreamingSchedule) else None
        plan = compiled.transfer_plan if compiled is not None else transfer_plan
        started = time.perf_counter()
        resolved = trainer.resolved_metadata
        required_metadata = _required_agent_metadata(
            plan, resolved, source_binding
        )
        if compiled is not None:
            for batch in compiled.batches:
                required_metadata.update(
                    _required_agent_metadata(
                        batch.transfer_plan, resolved, source_binding
                    )
                )
        metrics["transfer_planning_s"] += time.perf_counter() - started
        started = time.perf_counter()
        transport = self._connect_sources(
            resolved,
            required_metadata,
            host_staging=(
                streaming is not None and streaming.staging_device == "cpu"
            ),
        )
        if streaming is not None:
            self._prepare_arenas(cast(_StreamingSchedule, transfer_plan).batches)
            metrics["connection_registration_s"] = time.perf_counter() - started
        return transport

    def _compile_full_copy(
        self,
        trainer: TrainerSourceSnapshot,
        capture: CaptureResult,
        parameter_layout: Mapping,
        metrics: dict[str, float],
        binding: _SourceBinding,
    ) -> LogicalTransferPlan:
        started = time.perf_counter()
        slots_by_agent = {agent: slot for slot, agent in binding.agents_by_slot.items()}
        plan = plan_transfer(
            capture,
            trainer.resolved_metadata.sources,
            source_slots={
                session: slots_by_agent[agent]
                for session, agent in trainer.resolved_metadata.session_to_agent.items()
            },
            verify_full_sources=envs.MX_RESHARD_PUBLISH_DIGEST,
        )
        metrics["initial_whole_plan_s"] = time.perf_counter() - started
        validation_started = time.perf_counter()
        self._validate_complete(capture, dict(parameter_layout), plan)
        metrics["initial_whole_validation_s"] = time.perf_counter() - validation_started
        metrics["transfer_planning_s"] = time.perf_counter() - started
        return plan

    def _prepare_full_copy(
        self,
        cached: _WeightUpdatePlan,
        transport: NixlReshardTransport,
        metrics: dict[str, float],
        checksums: Mapping[ShardChecksumKey, str],
        reusable: bool,
    ) -> _PreparedNixlTransfer:
        plan = cast(LogicalTransferPlan, cached.transfer_plan)
        capture = cached.layout_capture_snapshot
        parameter_layout = cached.parameter_layout
        resolved = cached.trainer_source_snapshot.resolved_metadata
        source_binding = cached.source_binding
        self._ensure_workspace(plan, parameter_layout)
        if not reusable or self._full_copy_descriptors is None:
            self._full_copy_descriptors = tuple(
                self._descriptors(plan, source_binding=source_binding)
            )
        return _PreparedNixlTransfer(
            transfer_plan=plan,
            capture=capture,
            sources={
                copy.src_name: resolved.sources[copy.src_name]
                for copy in capture.copies
                if copy.src_name in resolved.sources
            },
            descriptors=self._full_copy_descriptors,
            transport=transport,
            source_binding=source_binding,
            metrics=metrics,
            checksums=checksums,
        )

    def _prepare_streaming(
        self,
        cached: _WeightUpdatePlan,
        transport: NixlReshardTransport,
        metrics: dict[str, float],
        checksums: Mapping[ShardChecksumKey, str],
        reusable: bool,
    ) -> _PreparedBoundedTransfer:
        compiled = cast(_StreamingSchedule, cached.transfer_plan)
        resolved = cached.trainer_source_snapshot.resolved_metadata
        source_binding = cached.source_binding
        if not reusable or self._descriptor_cache is None:
            self._descriptor_cache = [None] * len(compiled.batches)
        prepared = _PreparedBoundedTransfer(
            compiled.batches,
            resolved.sources,
            transport,
            source_binding,
            metrics,
            checksums,
        )
        return prepared

    def _prepare_arenas(
        self,
        batches: tuple[_StreamingBatch, ...],
    ) -> None:
        streaming = cast(StreamingSettings, self._streaming)
        buffer_budget = cast(int, self._buffer_budget)
        staging_device = streaming.staging_device
        staging_buffers = streaming.staging_buffers
        arena_bytes = max(batch.nbytes for batch in batches)
        if not self._staging_arenas:
            self._staging_device = torch.device(staging_device)
            for index in range(staging_buffers):
                arena = self._allocate_arena(arena_bytes)
                # Retain storage before registration so failed setup can be cleaned up.
                self._staging_arenas.append(arena)
                self._native_setup_started = True
                if staging_device == "cpu":
                    self._staging_registrations.append(
                        self._manager.register_dram_buffer(arena)
                    )
                else:
                    self._manager.register_tensors(
                        {f"__bounded_arena_{index}__": arena}
                    )
        elif self._staging_arenas[0].numel() < arena_bytes:
            raise RuntimeError(
                "bounded workspace layout grew; restart the generator engine"
            )
        if self._staging_arenas[0].numel() > buffer_budget:
            raise RuntimeError("existing bounded arena exceeds the requested limit")

    def _bounded_descriptors(
        self, prepared, metrics,
        index: int,
        recv: dict[str, torch.Tensor],
        full: dict[str, torch.Tensor],
        convert: dict[str, torch.Tensor],
    ) -> tuple[ReadDescriptor, ...]:
        entry = self._descriptor_cache
        cached = entry[index] if entry is not None else None
        if cached is not None:
            metrics["descriptor_cache_hits"] += 1
            return cached
        metrics["descriptor_cache_misses"] += 1
        metrics["descriptor_builds"] += 1
        fresh = tuple(
            self._descriptors(
                prepared.batches[index].transfer_plan, recv, full, convert,
                source_binding=prepared.source_binding,
            )
        )
        if entry is None:
            return fresh
        entry[index] = fresh
        return fresh

    @staticmethod
    def _carve_bounded(
        batch: _StreamingBatch, arena: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], ...]:
        offset = 0
        buffers = []
        for layout in batch.layouts:
            tensors = {}
            for name, (shape, dtype) in layout.items():
                nbytes = math.prod(shape) * dtype.itemsize
                tensors[name] = (
                    arena[offset : offset + nbytes].view(dtype).view(shape)
                )
                offset += ((nbytes + 255) // 256) * 256
            buffers.append(tensors)
        return tuple(buffers)

    def _post_bounded(self, prepared, index: int, metrics) -> tuple:
        arenas = self._staging_arenas
        batch = prepared.batches[index]
        recv, convert, full = self._carve_bounded(batch, arenas[index % len(arenas)])
        # Captured loaders may leave padding untouched. Reused arenas must
        # reproduce the zero-filled load layout before the NIC writes it.
        for tensors in (recv, convert, full):
            for tensor in tensors.values():
                tensor.zero_()
        if arenas[index % len(arenas)].device.type == "cuda":
            torch.cuda.synchronize(self._device)
        sources = {
            c.src_name: prepared.sources[c.src_name] for c in batch.capture.copies
        }
        chunk = _PreparedNixlTransfer(
            batch.transfer_plan,
            batch.capture,
            sources,
            self._bounded_descriptors(prepared, metrics, index, recv, full, convert),
            prepared.transport,
            prepared.source_binding,
            checksums=prepared.checksums,
        )
        started = time.perf_counter()
        posted = prepared.transport.post_reads(list(chunk.descriptors))
        return chunk, (recv, convert, full), posted, started

    def iter_bounded(
        self, prepared: _PreparedBoundedTransfer, metrics: dict[str, Any]
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield staged batches; callers must commit before advancing.

        With one arena each batch is read, yielded, and committed in turn.
        Content digests are checked only when MX_RESHARD_PUBLISH_DIGEST is
        enabled. With two arenas the READ for batch ``i + 1`` is posted into the
        other arena before batch ``i`` is yielded, allowing asynchronous READs
        to overlap the caller's commit. An arena is only reposted after the
        commit that read from it has been synchronized.
        """
        if self._closed or prepared is not self._active:
            raise RuntimeError("bounded NIXL transfer is no longer active")
        arenas = self._staging_arenas
        assert arenas
        metrics["staging_peak_bytes"] = sum(a.numel() for a in arenas)
        metrics["staging_buffers"] = len(arenas)
        metrics["batches"] = len(prepared.batches)
        batches = prepared.batches
        metrics.update(
            descriptor_cache_hits=0, descriptor_cache_misses=0, descriptor_builds=0
        )

        pending = None
        unwinding = False
        completed = False
        try:
            pending = self._post_bounded(prepared, 0, metrics)
            for index in range(len(batches)):
                chunk, buffers, posted, started = pending
                pending = None
                self._recv_buffers, self._convert_buffers, self._full_buffers = buffers
                self._active = chunk
                staged = self._complete_stage(chunk, posted, started)
                if len(arenas) > 1 and index + 1 < len(batches):
                    # The other arena's previous batch was committed and
                    # synchronized one iteration ago, so it is free to refill.
                    pending = self._post_bounded(prepared, index + 1, metrics)
                for key, value in staged.metrics.items():
                    metrics[key] = metrics.get(key, 0) + value
                yield staged.tensors
                # All installation reads must complete before arena reuse.
                torch.cuda.synchronize(self._device)
                if len(arenas) == 1 and index + 1 < len(batches):
                    pending = self._post_bounded(prepared, index + 1, metrics)
            completed = True
        except GeneratorExit:
            # Deliberate abandonment, not a failure, so a drain error below has
            # nothing to mask and must still be reported.
            raise
        except BaseException:
            unwinding = True
            self._invalidate_cached_transfer()
            raise
        finally:
            if not completed:
                self._descriptor_cache = None
            if pending is not None:
                # A prefetched READ is in flight for a batch the caller will
                # never consume; drain it so the handles are released.
                try:
                    prepared.transport.await_reads(pending[2])
                except Exception as error:
                    # Not merely uncleaned: until this READ is drained the arena
                    # may still receive RDMA writes, so reusing or freeing it is
                    # unsafe. Report it rather than returning as if the transfer
                    # had ended, and only downgrade to a log when an earlier
                    # failure is already propagating and must not be masked.
                    logger.error(
                        "draining a prefetched bounded READ batch failed; the "
                        "staging arena may still be written by an in-flight read "
                        "and the generator engine must be restarted",
                        exc_info=True,
                    )
                    self._invalidate_cached_transfer()
                    if not unwinding:
                        raise RuntimeError(
                            "a prefetched bounded READ could not be drained, so the "
                            "staging arena may still be written; restart the "
                            "generator engine"
                        ) from error
            self._active = prepared

    @staticmethod
    def _validate_complete(
        capture: CaptureResult,
        parameter_layout: dict[str, tuple[tuple[int, ...], torch.dtype]],
        plan: LogicalTransferPlan | _StreamingSchedule,
        metrics: dict[str, float] | None = None,
    ) -> None:
        started = time.perf_counter()
        if isinstance(plan, _StreamingSchedule):
            whole_started = time.perf_counter()
            _NixlStagedTransfer._validate_complete(
                capture, parameter_layout, plan.transfer_plan
            )
            if metrics is not None:
                metrics["bounded_whole_validation_s"] = time.perf_counter() - whole_started
            owner_started = time.perf_counter()
            for batch in plan.module_batches:
                _NixlStagedTransfer._validate_complete(
                    batch.capture, batch.layouts[0], batch.transfer_plan
                )
            if metrics is not None:
                metrics["owner_validation_s"] = time.perf_counter() - owner_started
                metrics["plan_cache_validate_s"] = time.perf_counter() - started
            return

        written = {copy.param_name for copy in capture.copies}
        missing = sorted(set(parameter_layout) - written)
        unsupported = list(capture.unsupported)
        if missing or unsupported or capture.unattributed or plan.fallback:
            causes = summarize_unsupported(capture.unsupported_reasons)
            raise IncompleteRefit(
                "full-tensor refit must cover every engine parameter; "
                f"missing={len(missing)}, unsupported={len(unsupported)}, "
                f"unattributed={capture.unattributed}, fallback={len(plan.fallback)}, "
                f"causes={causes}, missing_names={missing[:10]}"
            )
        if metrics is not None:
            metrics["plan_cache_validate_s"] = time.perf_counter() - started

    @staticmethod
    def _layout(tensors: dict[str, torch.Tensor]) -> dict:
        return {
            name: (tuple(tensor.shape), tensor.dtype)
            for name, tensor in tensors.items()
        }

    def _ensure_buffers(
        self,
        current: dict[str, torch.Tensor],
        expected: dict[str, tuple[tuple[int, ...], torch.dtype]],
        *,
        label: str,
    ) -> None:
        if current:
            if self._layout(current) != expected:
                raise RuntimeError(
                    f"{label} layout changed; restart the generator engine"
                )
            return
        with classic_cuda_alloc():
            current.update(
                {
                    name: torch.empty(shape, dtype=dtype, device=self._device)
                    for name, (shape, dtype) in expected.items()
                }
            )

    def _ensure_workspace(
        self,
        plan: LogicalTransferPlan,
        parameter_layout: dict[str, tuple[tuple[int, ...], torch.dtype]],
    ) -> None:
        recv_expected = {
            name: (tuple(shape), dtype)
            for name, (shape, dtype) in parameter_layout.items()
        }
        self._ensure_buffers(self._recv_buffers, recv_expected, label="receive-buffer")

        convert_expected = {
            convert.param_name: (tuple(convert.dest_shape), convert.src_dtype)
            for convert in plan.converts
        }
        self._ensure_buffers(
            self._convert_buffers, convert_expected, label="conversion-buffer"
        )
        full_expected = {
            full.src_name: (tuple(full.global_shape), full.dtype)
            for full in plan.full_pulls
        }
        self._ensure_buffers(
            self._full_buffers, full_expected, label="full-pull buffer"
        )

        recv_params = set(recv_expected)
        if self._registered_recv_params and self._registered_recv_params != recv_params:
            raise RuntimeError(
                "receive parameter set changed; restart the generator engine"
            )
        if convert_expected and not self._convert_registered:
            self._native_setup_started = True
            self._manager.register_tensors(
                {
                    f"__convert__{name}": tensor
                    for name, tensor in self._convert_buffers.items()
                }
            )
            self._convert_registered = True
        if full_expected and not self._full_registered:
            self._native_setup_started = True
            self._manager.register_tensors(
                {
                    f"__full__{name}": tensor
                    for name, tensor in self._full_buffers.items()
                }
            )
            self._full_registered = True
        if not self._registered_recv_params and recv_params:
            self._native_setup_started = True
            self._manager.register_tensors(self._recv_buffers)
            self._registered_recv_params = recv_params

    def _descriptors(
        self,
        plan: LogicalTransferPlan,
        recv: dict[str, torch.Tensor] | None = None,
        full: dict[str, torch.Tensor] | None = None,
        convert: dict[str, torch.Tensor] | None = None,
        *,
        source_binding: _SourceBinding,
    ) -> list[ReadDescriptor]:
        recv_buffers = self._recv_buffers if recv is None else recv
        full_buffers = self._full_buffers if full is None else full
        convert_buffers = self._convert_buffers if convert is None else convert

        buffers = {"recv": recv_buffers, "full": full_buffers, "convert": convert_buffers}
        return bind_transfer(
            plan, source_binding.resolve,
            lambda kind, name: buffers[kind][name].data_ptr(),
        ).descriptors

    def stage(self, prepared: _PreparedNixlTransfer) -> _StagedNixlWeights:
        """Pull, reconstruct, convert, and verify without touching live weights."""
        if self._closed:
            raise RuntimeError("NIXL staged transfer is closed")
        if prepared is not self._active:
            raise RuntimeError("NIXL transfer plan is no longer active")
        started = time.perf_counter()
        try:
            posted = prepared.transport.post_reads(list(prepared.descriptors))
            return self._complete_stage(prepared, posted, started)
        except BaseException:
            self._invalidate_cached_transfer()
            raise

    @torch.no_grad()
    def _complete_stage(
        self, prepared: _PreparedNixlTransfer, posted: list, started: float
    ) -> _StagedNixlWeights:
        """Wait for posted READs, then reconstruct, convert, and verify."""
        wait_started = time.perf_counter()
        prepared.transport.await_reads(posted)
        wire_wait_seconds = time.perf_counter() - wait_started
        wire_seconds = time.perf_counter() - started

        reconstruct_started = time.perf_counter()
        for full in prepared.transfer_plan.full_pulls:
            source = self._full_buffers[full.src_name]
            for copy in full.copies:
                destination = self._recv_buffers[copy.param_name].as_strided(
                    copy.dest_shape,
                    copy.dest_stride,
                    self._recv_buffers[copy.param_name].storage_offset()
                    + copy.dest_offset,
                )
                destination.copy_(_replay_ops(source, copy.op_chain))
        converted = {convert.param_name for convert in prepared.transfer_plan.converts}
        conversion_copies = {
            copy.param_name: copy
            for copy in prepared.capture.copies
            if copy.param_name in converted
        }
        for convert in prepared.transfer_plan.converts:
            copy = conversion_copies[convert.param_name]
            target = self._recv_buffers[convert.param_name]
            destination = target.as_strided(
                copy.dest_shape,
                copy.dest_stride,
                target.storage_offset() + copy.dest_offset,
            )
            destination.copy_(self._convert_buffers[convert.param_name])
        torch.cuda.synchronize(self._device)
        reconstruct_seconds = time.perf_counter() - reconstruct_started
        # Only digest mode has complete tensors and stamped digests to check.
        if envs.MX_RESHARD_PUBLISH_DIGEST:
            self._verify(prepared)

        bytes_received = sum(d.nbytes for d in prepared.descriptors)
        # This is the path the FSDP trainer refits over, and the path the 20x
        # collapse was measured on, so it is the one the floor most needs to cover.
        throughput.warn_if_below_floor(
            wire_bytes=bytes_received,
            wire_seconds=wire_seconds,
            log=logger,
            context={"device_id": self._device_id, "phase": "stage"},
        )
        logger.info(
            "[TIMING] staged xfer: %.3f GB, %d descriptors "
            "(seg=%d full_pull=%d convert=%d), %d tensors | "
            "wire=%.3fs reconstruct=%.3fs",
            bytes_received / 1e9,
            len(prepared.descriptors),
            len(prepared.transfer_plan.segments),
            len(prepared.transfer_plan.full_pulls),
            len(prepared.transfer_plan.converts),
            len(self._recv_buffers),
            wire_seconds,
            reconstruct_seconds,
        )
        return _StagedNixlWeights(
            tensors=self._recv_buffers,
            metrics={
                "bytes_received": bytes_received,
                "segments": len(prepared.descriptors),
                "wire_s": wire_seconds,
                "wire_wait_s": wire_wait_seconds,
                "reconstruct_s": reconstruct_seconds,
                "full_pull_sources": len(prepared.transfer_plan.full_pulls),
                "converts": len(prepared.transfer_plan.converts),
            },
        )

    @staticmethod
    def _validate_peer_manifest(
        manifest: p2p_pb2.GetTensorManifestResponse,
        destination_tensors: dict[str, torch.Tensor],
    ) -> list[TensorDescriptor]:
        source_tensors = [
            TensorDescriptor(
                name=tensor.name,
                addr=tensor.addr,
                size=tensor.size,
                device_id=tensor.device_id,
                dtype=tensor.dtype,
            )
            for tensor in manifest.tensors
        ]
        if not source_tensors:
            raise RuntimeError("P2P source has no tensor descriptors")

        source_names = {tensor.name for tensor in source_tensors}
        local_names = set(destination_tensors)
        if source_names != local_names:
            local_only = sorted(local_names - source_names)
            source_only = sorted(source_names - local_names)
            raise ManifestMismatchError(
                "runtime tensor name mismatch: "
                f"{len(local_only)} local-only (first: {local_only[:5]}), "
                f"{len(source_only)} source-only (first: {source_only[:5]})"
            )
        for source in source_tensors:
            destination = destination_tensors[source.name]
            size = destination.numel() * destination.element_size()
            if source.size != size or source.dtype != str(destination.dtype):
                raise ManifestMismatchError(
                    f"runtime tensor metadata mismatch for {source.name!r}"
                )
        return source_tensors

    @staticmethod
    def _peer_endpoint(
        manifest: p2p_pb2.GetTensorManifestResponse,
    ) -> tuple[str, int, str]:
        endpoint = manifest.metadata_endpoint
        try:
            host, port_text = endpoint.rsplit(":", 1)
            port = int(port_text)
        except ValueError as error:
            raise RuntimeError(
                f"P2P source published an unusable metadata endpoint: {endpoint!r}"
            ) from error
        if not host or not 1 <= port <= 65535 or not manifest.agent_name:
            raise RuntimeError(
                "P2P source published unusable NIXL connection metadata: "
                f"endpoint={endpoint!r}, agent_name={manifest.agent_name!r}"
            )
        return host, port, manifest.agent_name

    def prepare_peer_read(
        self,
        *,
        source: p2p_pb2.WorkerMetadata,
        mx_source_id: str,
        worker_id: str,
        destination_tensors: dict[str, torch.Tensor],
    ) -> TensorReadLease:
        """Validate and retain a donor lease until apply or release."""
        if self._closed:
            raise RuntimeError("NIXL staged transfer is closed")
        if not source.worker_grpc_endpoint:
            raise RuntimeError("generator P2P source has no tensor lease endpoint")
        lease, _ = prepare_tensor_read(
            source.worker_grpc_endpoint,
            mx_source_id,
            worker_id=worker_id,
            timeout=self._timeout,
        )
        try:
            self._validate_peer_manifest(lease.manifest, destination_tensors)
            self._peer_endpoint(lease.manifest)
        except BaseException:
            lease.close()
            raise
        return lease

    def receive_peer(
        self,
        *,
        tensor_read: TensorReadLease,
        destination_tensors: dict[str, torch.Tensor],
        on_transfer_start: Callable[[], None],
    ) -> dict[str, Any]:
        """Pull a peer's runtime tensors directly into live engine storage."""
        if self._closed:
            raise RuntimeError("NIXL staged transfer is closed")

        remote_agent_name: str | None = None
        started = time.perf_counter()
        manifest = tensor_read.manifest
        source_tensors = self._validate_peer_manifest(
            manifest,
            destination_tensors,
        )
        host, port, remote_agent_name = self._peer_endpoint(manifest)
        try:
            self._manager.fetch_remote_and_wait(
                remote_agent_name=remote_agent_name,
                ip=host,
                port=port,
                timeout_seconds=self._timeout,
            )
            bytes_received, tensor_count, wire_seconds = (
                self._manager.receive_from_source(
                    source_metadata=b"",
                    source_tensors=source_tensors,
                    timeout_seconds=self._timeout,
                    remote_agent_name=remote_agent_name,
                    require_exact_match=True,
                    destination_tensors=destination_tensors,
                    on_transfer_start=on_transfer_start,
                )
            )
        finally:
            if remote_agent_name is not None:
                self._manager.remove_remote_agent(remote_agent_name)

        throughput.warn_if_below_floor(
            wire_bytes=bytes_received,
            wire_seconds=wire_seconds,
            log=logger,
            context={"device_id": self._device_id, "phase": "receive_peer"},
        )
        return {
            "bytes_received": bytes_received,
            "segments": tensor_count,
            "wire_s": round(wire_seconds, 6),
            "peer_s": round(time.perf_counter() - started, 6),
        }

    def _verification_tensor(self, prepared: _PreparedNixlTransfer, name: str):
        source = prepared.sources[name]
        if name in self._full_buffers:
            return self._full_buffers[name]
        copy = next(
            (
                copy
                for copy in prepared.capture.copies
                if copy.src_name == name
                and not copy.op_chain
                and tuple(copy.dest_shape) == tuple(source.global_shape)
            ),
            None,
        )
        if copy is None:
            raise RuntimeError(f"cannot recover complete staged source {name!r}")
        if copy.param_name in self._convert_buffers:
            return self._convert_buffers[copy.param_name]
        buffer = self._recv_buffers[copy.param_name]
        return buffer.as_strided(
            copy.dest_shape,
            copy.dest_stride,
            buffer.storage_offset() + copy.dest_offset,
        )

    def _verify(self, prepared: _PreparedNixlTransfer) -> None:
        for name, source in prepared.sources.items():
            tensor = self._verification_tensor(prepared, name)
            for shard in source.shards:
                expected = prepared.checksums.get(
                    (
                        name, shard.session, shard.addr,
                        tuple(shard.shard_offset), tuple(shard.shape),
                    )
                )
                if not expected:
                    raise RuntimeError(
                        f"source {name!r} did not publish a verification digest"
                    )
                actual = tensor_digest(
                    shard_region(
                        tensor,
                        source.global_shape,
                        shard.shard_offset,
                        shard.shape,
                    )
                )
                if actual != expected:
                    raise RuntimeError(
                        f"staged weight digest mismatch for source {name!r} "
                        f"at offset {tuple(shard.shard_offset)}"
                    )

    def close(self) -> None:
        if self._closed:
            return
        self._invalidate_descriptors()
        self._weight_update_plan = None
        self._full_copy_descriptors = None
        self._closed = True
        if self._owns_manager:
            self._release_staging_registrations()
            self._manager.shutdown()
            # The agent's registrations are gone, so staging storage can be
            # freed eagerly. A shared agent may still hold these buffers
            # registered; keep them referenced until its owner shuts down.
            self._recv_buffers.clear()
            self._convert_buffers.clear()
            self._full_buffers.clear()
            self._staging_arenas.clear()
            self._staging_device = None


__all__: list[str] = []
