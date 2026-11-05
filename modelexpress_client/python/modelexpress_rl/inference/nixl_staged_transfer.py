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
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, cast

import torch
from modelexpress import envs
from modelexpress_rl import envs as rl_envs
from modelexpress.nixl_transfer import (
    NIXL_DRAM_MEM_TYPE,
    NIXL_VRAM_MEM_TYPE,
    NixlTransferManager,
)
from modelexpress_rl.refit.reshard import throughput
from modelexpress_rl.refit.reshard.cuda_pool import classic_cuda_alloc
from modelexpress_rl.refit.reshard.rendezvous import (
    build_sources,
    merge_shard_tables,
    unwrap_rendezvous_blob,
)
from modelexpress_rl.refit.reshard.transfer_plan import (
    LogicalTransferPlan,
    TransferBindingCache,
)
from modelexpress_rl.refit.reshard.transport import ReadDescriptor
from modelexpress_rl.refit.reshard.transport.nixl import NixlReshardTransport
from modelexpress_rl.refit.reshard.types import (
    CaptureResult,
)
from modelexpress_rl.refit.reshard.verify import shard_region, tensor_digest

from modelexpress_rl.inference.plan import SourceReplayBatch, TrainerSourceSnapshot

from ..train.manifest import ShardChecksumKey
from ..version import MeshIdentity
from .plan import StreamingSettings
from .trainer_plan import TrainerTransferPlan, TransferBatch

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
    """Own decoded physical sources and their selected logical-slot bindings."""

    resolved: _ResolvedSources
    agents_by_slot: dict[str, str]
    physical_fingerprint: tuple

    @classmethod
    def from_snapshot(cls, snapshot: TrainerSourceSnapshot, metrics: dict) -> _SourceBinding:
        if not snapshot.shards:
            raise ValueError("at least one source manifest is required")
        payloads = {}
        started = time.perf_counter()
        for shard in snapshot.shards:
            if not isinstance(shard.metadata, bytes):
                raise TypeError("cached source manifests must be immutable bytes")
            if shard.source_slot_id in payloads:
                raise ValueError("source slots must identify each selected replica once")
            payloads[shard.source_slot_id] = unwrap_rendezvous_blob(shard.metadata)
        agents = {slot: payload.agent_name for slot, payload in payloads.items()}
        if len(set(agents.values())) != len(agents):
            raise ValueError("source manifests contain duplicate NIXL agents")
        metrics["source_decode_s"] = time.perf_counter() - started
        started = time.perf_counter()
        merged = merge_shard_tables([payload.tensors for payload in payloads.values()])
        metrics["source_merge_s"] = time.perf_counter() - started
        started = time.perf_counter()
        session_to_memory = {}
        sources, session_to_agent, session_to_device = build_sources(
            merged, session_to_memory=session_to_memory
        )
        metrics["source_build_s"] = time.perf_counter() - started
        resolved = _ResolvedSources(
            sources, session_to_agent, session_to_device,
            {payload.agent_name: payload.agent_metadata for payload in payloads.values()},
            session_to_memory,
        )
        return cls(resolved, agents, snapshot.physical_fingerprint)

    def resolve(self, buffer) -> tuple[str, int]:
        agent = self.agents_by_slot[buffer.source_slot_id]
        shard = self.resolved.shard_bindings[
            (agent, buffer.src_name, buffer.shard_offset, buffer.shape)
        ]
        return shard.session, shard.addr


@dataclass(frozen=True)
class _PreparedNixlTransfer:
    """One bound logical plan over reusable registered destinations."""

    transfer_plan: LogicalTransferPlan
    sources: dict
    descriptors: tuple[ReadDescriptor, ...]
    transport: NixlReshardTransport
    source_binding: _SourceBinding
    metrics: dict[str, float] = field(default_factory=dict)
    checksums: Mapping[ShardChecksumKey, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _StagedNixlWeights:
    """Verified tensors ready for engine installation."""

    tensors: SourceReplayBatch
    metrics: dict[str, Any]


_StagingLayout = dict[str, tuple[tuple[int, ...], torch.dtype]]
_CaptureLayout = Callable[
    [list[tuple[str, torch.dtype, tuple[int, ...]]]],
    tuple[CaptureResult, _StagingLayout],
]


@dataclass(frozen=True)
class _PreparedBoundedTransfer:
    batches: tuple[TransferBatch, ...]
    sources: dict
    transport: NixlReshardTransport
    source_binding: _SourceBinding
    metrics: dict[str, float] = field(default_factory=dict)
    checksums: Mapping[ShardChecksumKey, str] = field(default_factory=dict)
    descriptors: tuple[tuple[ReadDescriptor, ...], ...] = ()
    descriptors_reused: tuple[bool, ...] = ()

    @property
    def replay_id(self):
        return self.batches[0].transfer_plan.replay_id


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


class _NixlStagedTransfer:
    """Own trainer receive storage and NIXL transfer execution.

    Own the NIXL manager, registered receive buffers, source connections, physical
    bindings and read completion. Delegate geometry validation and batching to
    TrainerTransferPlan, then produce ordered SourceReplayBatch values for the
    installer. Keep receive storage alive until its registrations are torn down.
    """

    def __init__(
        self,
        *,
        device_id: int,
        device: torch.device,
        capture_layout: _CaptureLayout,
        generator_fingerprint: Callable[[], tuple],
        agent_name: str,
        listen_port: int | None = None,
        timeout_seconds: float | None = None,
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
        self._device_id = device_id
        self._device = device
        self._timeout = float(
            envs.MX_TRANSFER_TIMEOUT if timeout_seconds is None else timeout_seconds
        )
        self._manager = NixlTransferManager(
            agent_name=agent_name,
            device_id=device_id,
            listen_port=listen_port,
        )
        try:
            self._manager.initialize()
        except Exception:
            self._manager.shutdown()
            raise
        self._full_buffers: dict[str, torch.Tensor] = {}
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
        self._weight_update_plan: TrainerTransferPlan | None = None
        self._binding: _SourceBinding | None = None
        self._transport: NixlReshardTransport | None = None
        self._descriptor_caches: tuple[TransferBindingCache, ...] | None = None
        self._debug_validate_plan = rl_envs.MX_REFIT_DEBUG_VALIDATE_PLAN
        self._debug_validate_generator_layout = (
            rl_envs.MX_REFIT_DEBUG_VALIDATE_GENERATOR_LAYOUT
        )
        self._plan_compile_lock = threading.Lock()

    @property
    def bounded_staging(self) -> bool:
        return self._streaming is not None

    def _invalidate_descriptors(self) -> None:
        self._descriptor_caches = None
        self._binding = None
        self._invalidate_cached_transfer()

    def _invalidate_cached_transfer(self) -> None:
        self._transport = None

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
        self._invalidate_cached_transfer()
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        self._release_staging_registrations()
        self._manager.shutdown()
        self._active = None
        self._full_buffers.clear()
        self._staging_arenas.clear()
        self._staging_device = None
        self._full_registered = False
        self._loaded_agent_metadata.clear()
        self._manager_ready = False

    def _ensure_manager_initialized(self) -> None:
        if not self._manager_ready:
            self._native_setup_started = True
            try:
                self._manager.initialize()
            except Exception:
                self._manager.shutdown()
                raise
            self._manager_ready = True

    def _resolve_metadata(
        self, snapshot: TrainerSourceSnapshot, metrics: dict
    ) -> _SourceBinding:
        previous = self._binding
        if previous is not None and previous.physical_fingerprint == snapshot.physical_fingerprint:
            metrics.update(source_cache_enabled=1, source_cache_hits=1, source_cache_misses=0)
            return previous
        metrics.update(
            source_cache_enabled=1,
            source_cache_hits=0,
            source_cache_misses=1,
            source_cache_lookup_s=0.0,
            source_manifest_bytes=sum(len(shard.metadata) for shard in snapshot.shards),
        )
        return _SourceBinding.from_snapshot(snapshot, metrics)

    def _retire_previous_mesh(self, trainer: TrainerSourceSnapshot) -> None:
        previous = self._weight_update_plan
        if previous is not None and previous.mesh_identity != trainer.mesh_identity:
            self._weight_update_plan = None
            self._invalidate_descriptors()

    def _resolve_layout(
        self, binding: _SourceBinding, metrics: dict[str, float]
    ) -> tuple[CaptureResult, _StagingLayout]:
        started = time.perf_counter()
        source_schema = tuple(
            (name, source.dtype, tuple(source.global_shape))
            for name, source in sorted(binding.resolved.sources.items())
        )
        metrics["source_metadata_s"] = time.perf_counter() - started
        started = time.perf_counter()
        capture, parameter_layout = self._capture_layout(list(source_schema))
        metrics["layout_capture_s"] = time.perf_counter() - started
        return capture, parameter_layout

    def _compile_plan(self, trainer, capture, parameter_layout, metrics, binding):
        if not self._plan_compile_lock.acquire(blocking=False):
            raise RuntimeError("trainer plan compilation is already in progress")
        try:
            resolved = binding.resolved
            slots_by_agent = {agent: slot for slot, agent in binding.agents_by_slot.items()}
            return TrainerTransferPlan.compile(
                mesh_identity=trainer.mesh_identity,
                capture=capture,
                parameter_layout=parameter_layout,
                sources=resolved.sources,
                source_slots={
                    session: slots_by_agent[agent]
                    for session, agent in resolved.session_to_agent.items()
                },
                streaming=self._streaming,
                metrics=metrics,
                verify_full_sources=rl_envs.MX_RESHARD_PUBLISH_DIGEST,
            )
        finally:
            self._plan_compile_lock.release()

    def _connect_sources(
        self,
        resolved: _ResolvedSources,
        required_metadata: dict[str, bytes],
        *,
        host_staging: bool = False,
        recreate_workspace: bool = False,
    ) -> NixlReshardTransport:
        changed = {
            agent: metadata
            for agent, metadata in required_metadata.items()
            if self._loaded_agent_metadata.get(agent) != metadata
        }
        if recreate_workspace or any(agent in self._loaded_agent_metadata for agent in changed):
            self.reset_workspace()
            changed = dict(required_metadata)
        self._ensure_manager_initialized()
        if changed:
            self._native_setup_started = True
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
        trainer_snapshot: TrainerSourceSnapshot,
        checksums: Mapping[ShardChecksumKey, str] | None = None,
    ) -> _PreparedNixlTransfer | _PreparedBoundedTransfer:
        """Prepare one version-scoped transfer using this instance's staging mode."""
        streaming = self._streaming
        if checksums is None:
            checksums = {}
        self._retire_previous_mesh(trainer_snapshot)
        with self._preparing():
            previous = self._weight_update_plan
            metrics = {}
            lookup_started = time.perf_counter()
            reusable = (
                self._binding is not None and self._transport is not None
                and self._binding.physical_fingerprint == trainer_snapshot.physical_fingerprint
            )
            metrics.update(
                plan_cache_enabled=1,
                plan_cache_hits=int(reusable),
                plan_cache_misses=int(not reusable),
                plan_cache_lookup_s=time.perf_counter() - lookup_started,
                plan_cache_validate_s=0.0,
            )
            source_binding = self._resolve_metadata(trainer_snapshot, metrics)
            if previous is not None:
                capture = previous.layout_capture_snapshot
                parameter_layout = previous.parameter_layout
                if self._debug_validate_generator_layout:
                    assert (
                        self._generator_fingerprint() == capture.generator_fingerprint
                    ), "generator layout changed since capture"
            else:
                capture, parameter_layout = self._resolve_layout(source_binding, metrics)

            planning_started = time.perf_counter()
            cached = previous or self._compile_plan(
                trainer_snapshot, capture, parameter_layout, metrics, source_binding
            )
            if previous is not None and self._debug_validate_plan:
                started = time.perf_counter()
                cached.validate()
                metrics["plan_cache_validate_s"] = time.perf_counter() - started
            metrics["transfer_planning_s"] = time.perf_counter() - planning_started
            plan = cached.transfer_plan
            metrics.update(
                exact_descriptors=plan.exact_descriptor_count,
                planned_descriptors=plan.descriptor_count(),
                descriptor_savings=plan.descriptor_savings(),
                extra_wire_bytes=plan.extra_wire_bytes(),
            )
            if reusable:
                transport = self._transport
                if streaming is not None:
                    metrics["connection_registration_s"] = 0.0
            else:
                transport = self._prepare_transport(
                    cached, source_binding, metrics
                )

            if self._descriptor_caches is None:
                plans = (
                    (plan,) if streaming is None
                    else tuple(batch.transfer_plan for batch in cached.batches)
                )
                self._descriptor_caches = tuple(
                    TransferBindingCache(plan) for plan in plans
                )
            prepare = self._prepare_full_copy if streaming is None else self._prepare_streaming
            prepared = prepare(
                cached, trainer_snapshot, source_binding, transport, metrics, checksums, reusable
            )
            self._weight_update_plan = cached
            self._binding = source_binding
            self._transport = transport
            self._active = prepared
            return prepared

    def _prepare_transport(self, cached, source_binding, metrics):
        streaming = self._streaming
        started = time.perf_counter()
        resolved = source_binding.resolved
        required_metadata = _required_agent_metadata(
            cached.transfer_plan, resolved, source_binding
        )
        metrics["transfer_planning_s"] += time.perf_counter() - started
        started = time.perf_counter()
        transport = self._connect_sources(
            resolved,
            required_metadata,
            host_staging=(
                streaming is not None and streaming.staging_device == "cpu"
            ),
            recreate_workspace=(
                bool(cached.batches) and bool(self._staging_arenas)
                and self._staging_arenas[0].numel() < cached.arena_bytes
            ),
        )
        if streaming is not None:
            self._prepare_arenas(cached.arena_bytes)
            metrics["connection_registration_s"] = time.perf_counter() - started
        return transport

    def _prepare_full_copy(
        self,
        cached: TrainerTransferPlan,
        trainer: TrainerSourceSnapshot,
        source_binding: _SourceBinding,
        transport: NixlReshardTransport,
        metrics: dict[str, float],
        checksums: Mapping[ShardChecksumKey, str],
        reusable: bool,
    ) -> _PreparedNixlTransfer:
        plan = cached.transfer_plan
        capture = cached.layout_capture_snapshot
        resolved = source_binding.resolved
        if not reusable:
            self._ensure_workspace(plan)
        workers = {shard.source_slot_id: shard.worker_id
                   for shard in trainer.shards}
        descriptors, reused = self._descriptor_caches[0].bind(
            workers,
            source_binding.resolve,
            lambda _kind, name: self._full_buffers[name].data_ptr(),
        )
        metrics["descriptor_builds"] = int(not reused)
        return _PreparedNixlTransfer(
            transfer_plan=plan,
            sources={
                copy.src_name: resolved.sources[copy.src_name]
                for copy in capture.copies
                if copy.src_name in resolved.sources
            },
            descriptors=descriptors,
            transport=transport,
            source_binding=source_binding,
            metrics=metrics,
            checksums=checksums,
        )

    def _prepare_streaming(
        self,
        cached: TrainerTransferPlan,
        trainer: TrainerSourceSnapshot,
        source_binding: _SourceBinding,
        transport: NixlReshardTransport,
        metrics: dict[str, float],
        checksums: Mapping[ShardChecksumKey, str],
        reusable: bool,
    ) -> _PreparedBoundedTransfer:
        compiled = cached
        workers = {shard.source_slot_id: shard.worker_id
                   for shard in trainer.shards}
        descriptors = []
        reused = []
        for index, batch in enumerate(compiled.batches):
            cache = self._descriptor_caches[index]
            if reusable:
                reads, hit = cache.last_descriptors, True
            else:
                full = self._carve_bounded(
                    batch, self._staging_arenas[index % len(self._staging_arenas)]
                )
                reads, hit = cache.bind(
                    workers,
                    source_binding.resolve,
                    lambda _kind, name: full[name].data_ptr(),
                )
            descriptors.append(reads)
            reused.append(hit)
        metrics["descriptor_binding_reused"] = int(all(reused))
        metrics["descriptor_builds"] = sum(not hit for hit in reused)
        metrics["replay_scratch_peak_bytes"] = compiled.replay_scratch_peak_bytes
        return _PreparedBoundedTransfer(
            compiled.batches, source_binding.resolved.sources, transport, source_binding,
            metrics, checksums, tuple(descriptors), tuple(reused),
        )

    def _prepare_arenas(
        self,
        arena_bytes: int,
    ) -> None:
        streaming = cast(StreamingSettings, self._streaming)
        staging_device = streaming.staging_device
        staging_buffers = streaming.staging_buffers
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

    @staticmethod
    def _carve_bounded(
        batch: TransferBatch, arena: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {
            name: arena[start:end].view(dtype).view(shape)
            for name, (start, end, shape, dtype) in batch.layout.items()
        }

    def _post_bounded(self, prepared, index: int, metrics) -> tuple:
        arenas = self._staging_arenas
        batch = prepared.batches[index]
        metrics["descriptor_cache_hits" if prepared.descriptors_reused[index] else "descriptor_cache_misses"] += 1
        full = self._carve_bounded(batch, arenas[index % len(arenas)])
        # Captured loaders may leave padding untouched. Reused arenas must
        # reproduce the zero-filled load layout before the NIC writes it.
        arenas[index % len(arenas)][:batch.nbytes].zero_()
        if arenas[index % len(arenas)].device.type == "cuda":
            torch.cuda.synchronize(self._device)
        chunk = _PreparedNixlTransfer(
            batch.transfer_plan,
            prepared.sources,
            prepared.descriptors[index],
            prepared.transport,
            prepared.source_binding,
            checksums=prepared.checksums,
        )
        started = time.perf_counter()
        posted = prepared.transport.post_reads(list(chunk.descriptors))
        return chunk, full, posted, started

    def iter_bounded(
        self, prepared: _PreparedBoundedTransfer, metrics: dict[str, Any]
    ) -> Iterator[SourceReplayBatch]:
        """Yield staged batches to the stream owner, which fences before advancing.

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
        metrics["replay_scratch_peak_bytes"] = prepared.metrics.get("replay_scratch_peak_bytes", 0)
        batches = prepared.batches
        metrics.update(
            descriptor_cache_hits=0, descriptor_cache_misses=0,
            descriptor_builds=prepared.metrics.get("descriptor_builds", 0)
        )

        pending = None
        unwinding = False
        completed = False
        try:
            pending = self._post_bounded(prepared, 0, metrics)
            for index in range(len(batches)):
                chunk, buffers, posted, started = pending
                pending = None
                self._full_buffers = buffers
                self._active = chunk
                staged = self._complete_stage(chunk, posted, started)
                if len(arenas) > 1 and index + 1 < len(batches):
                    # The other arena's previous batch was committed and
                    # synchronized one iteration ago, so it is free to refill.
                    pending = self._post_bounded(prepared, index + 1, metrics)
                for key, value in staged.metrics.items():
                    metrics[key] = (
                        max(metrics.get(key, 0), value) if key.endswith("_peak_bytes")
                        else metrics.get(key, 0) + value
                    )
                yield staged.tensors
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
                self._invalidate_cached_transfer()
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

    def _ensure_workspace(self, plan: LogicalTransferPlan) -> None:
        expected = {
            pull.src_name: (tuple(pull.global_shape), pull.dtype)
            for pull in plan.full_pulls
        }
        self._ensure_buffers(self._full_buffers, expected, label="source-buffer")
        if expected and not self._full_registered:
            self._native_setup_started = True
            self._manager.register_tensors(
                {f"__full__{name}": tensor for name, tensor in self._full_buffers.items()}
            )
            self._full_registered = True

    def stage(self, prepared: _PreparedNixlTransfer) -> _StagedNixlWeights:
        """Pull and verify source views without touching live weights."""
        if self._closed:
            raise RuntimeError("NIXL staged transfer is closed")
        if prepared is not self._active:
            raise RuntimeError("NIXL transfer plan is no longer active")
        started = time.perf_counter()
        try:
            for tensor in self._full_buffers.values():
                tensor.zero_()
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
            posted = prepared.transport.post_reads(list(prepared.descriptors))
            return self._complete_stage(prepared, posted, started)
        except BaseException:
            self._invalidate_cached_transfer()
            raise

    @torch.no_grad()
    def _complete_stage(
        self, prepared: _PreparedNixlTransfer, posted: list, started: float
    ) -> _StagedNixlWeights:
        """Wait for posted READs and verify source views for ordered replay."""
        wait_started = time.perf_counter()
        prepared.transport.await_reads(posted)
        wire_wait_seconds = time.perf_counter() - wait_started
        wire_seconds = time.perf_counter() - started

        plan = prepared.transfer_plan
        throughput.warn_if_below_floor(
            wire_bytes=sum(d.nbytes for d in prepared.descriptors),
            wire_seconds=wire_seconds, log=logger,
            context={"device_id": self._device_id, "phase": "stage"},
        )
        if plan.verify_full_sources:
            for pull in plan.full_pulls:
                name = pull.copies[0].src_name
                self._verify_source(prepared, name, self._full_buffers[pull.src_name])
        writes = tuple(
            (index, copy, self._full_buffers[pull.src_name])
            for pull in plan.full_pulls
            for index, copy in zip(pull.copy_indices, pull.copies, strict=True)
        )
        return _StagedNixlWeights(
            SourceReplayBatch(
                writes, plan.replay_id,
            ),
            {
                "bytes_received": sum(d.nbytes for d in prepared.descriptors),
                "segments": len(prepared.descriptors),
                "wire_s": wire_seconds, "wire_wait_s": wire_wait_seconds,
                "reconstruct_s": 0.0, "full_pull_sources": len(plan.full_pulls),
                "converts": 0,
            },
        )


    def _verify_source(self, prepared, name, tensor) -> None:
        source = prepared.sources[name]
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
        self._closed = True
        self._release_staging_registrations()
        self._manager.shutdown()
        self._active = None
        self._full_buffers.clear()
        self._staging_arenas.clear()
        self._staging_device = None


__all__: list[str] = []
