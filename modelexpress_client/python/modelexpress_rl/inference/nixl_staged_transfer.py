# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-version staged NIXL transfer for RL generator workers.

This module owns the state that makes a pull transfer correct: the selected
source manifests, the physical plan, registered destination buffers, peer
metadata, transfer completion, and verification. It deliberately does not know
how an inference engine captures its load layout or installs received weights.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import torch

from modelexpress import envs, p2p_pb2
from modelexpress.metadata.payload import worker_tensor_descriptors
from modelexpress.metadata.worker_server import (
    TensorReadLease,
    prepare_tensor_read,
)
from modelexpress.nixl_transfer import NIXL_DRAM_MEM_TYPE, NixlTransferManager
from modelexpress.refit.reshard import throughput
from modelexpress.refit.reshard.cuda_pool import classic_cuda_alloc
from modelexpress.refit.reshard.rendezvous import (
    build_sources,
    merge_shard_tables,
    unwrap_rendezvous_blob,
)
from modelexpress.refit.reshard.slice_plan import plan_pull
from modelexpress.refit.reshard.transfer_plan import (
    FullPullSource,
    TransferPlan,
    exact_descriptors,
    plan_transfer,
)
from modelexpress.refit.reshard.transport import ReadDescriptor
from modelexpress.refit.reshard.transport.nixl import NixlReshardTransport
from modelexpress.refit.reshard.types import (
    CaptureResult,
    IncompleteRefit,
    RecordedCopy,
    UnsupportedReshard,
    summarize_unsupported,
)
from modelexpress.refit.reshard.verify import shard_region, tensor_digest
from modelexpress.types import ManifestMismatchError, TensorDescriptor

# Named under modelexpress.* (not modelexpress_rl) so the per-update summary surfaces
# in the vLLM engine process, which only configures the modelexpress logger.
logger = logging.getLogger("modelexpress.reshard.staged_transfer")


@dataclass(frozen=True)
class _ResolvedSources:
    sources: dict
    session_to_agent: dict
    session_to_device: dict
    agent_metadata: dict[str, bytes]


@dataclass(frozen=True)
class _PreparedNixlTransfer:
    """One immutable physical plan over reusable registered destinations."""

    plan: TransferPlan
    capture: CaptureResult
    sources: dict
    descriptors: tuple[ReadDescriptor, ...]
    transport: NixlReshardTransport


@dataclass(frozen=True)
class _StagedNixlWeights:
    """Verified tensors ready for engine installation."""

    tensors: dict[str, torch.Tensor]
    metrics: dict[str, Any]


_StagingLayout = dict[str, tuple[tuple[int, ...], torch.dtype]]


class _StagingLayouts(NamedTuple):
    """The three typed views one batch carves out of its arena, in arena order."""

    recv: _StagingLayout
    convert: _StagingLayout
    full: _StagingLayout


@dataclass(frozen=True)
class _BoundedBatch:
    capture: CaptureResult
    plan: TransferPlan
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
    batches: tuple[_BoundedBatch, ...]
    sources: dict
    transport: NixlReshardTransport
    metrics: dict[str, float] = field(default_factory=dict)


def _require_positive_bytes(value: object, name: str) -> int:
    """Return ``value`` as a byte count, rejecting bool and non-positive ints."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _bounded_batches(
    capture: CaptureResult,
    parameter_layout: _StagingLayout,
    sources: dict,
    max_staging_bytes: int,
    *,
    total_staging_bytes: int | None = None,
    staging_buffers: int = 1,
) -> tuple[_BoundedBatch, ...]:
    """Plan one batch per owning module within ``max_staging_bytes`` per arena.

    Every batch is validated before anything is allocated or installed.
    ``total_staging_bytes`` and ``staging_buffers`` only shape the error when a
    module does not fit, so the caller sees the split that produced the share.
    """
    _require_positive_bytes(max_staging_bytes, "max_staging_bytes")
    complete = _plan_staged_transfer(capture, sources)
    _NixlStagedTransfer._validate_complete(capture, parameter_layout, complete)
    if {copy.param_name for copy in capture.copies} - parameter_layout.keys():
        raise IncompleteRefit("bounded capture references unknown engine parameters")
    groups = {}
    for name in parameter_layout:
        groups.setdefault(name.rpartition(".")[0], {})[name] = parameter_layout[name]
    batches = []
    for module, recv in groups.items():
        subset = CaptureResult(
            copies=[c for c in capture.copies if c.param_name in recv]
        )
        plan = _plan_staged_transfer(subset, sources)
        _NixlStagedTransfer._validate_complete(subset, recv, plan)
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
        batches.append(_BoundedBatch(subset, plan, layouts, nbytes))
    if not batches:
        raise IncompleteRefit("bounded refit has no engine parameters")
    return tuple(batches)


def _pack_bounded_batches(
    batches: tuple[_BoundedBatch, ...], max_staging_bytes: int
) -> tuple[_BoundedBatch, ...]:
    """Pack consecutive complete modules without changing the READ descriptors.

    Owning-module batches are the unit of correctness; this only coalesces
    neighbours that fit the same arena together, so each packed batch still
    installs whole modules and reads exactly the bytes the unpacked plan read.
    """
    _require_positive_bytes(max_staging_bytes, "max_staging_bytes")

    def merge(group: list[_BoundedBatch]) -> _BoundedBatch:
        capture = CaptureResult(
            copies=[copy for batch in group for copy in batch.capture.copies]
        )
        plan = TransferPlan()
        layouts = _StagingLayouts({}, {}, {})
        for batch in group:
            _merge_plan(plan, batch.plan)
            for layout, incoming in zip(layouts, batch.layouts, strict=True):
                if layout.keys() & incoming.keys():
                    raise IncompleteRefit(
                        "packed batches contain overlapping staging keys"
                    )
                layout.update(incoming)
        return _BoundedBatch(
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


def _resolve_sources(manifests: list[bytes]) -> _ResolvedSources:
    if not manifests:
        raise ValueError("at least one source manifest is required")
    payloads = [unwrap_rendezvous_blob(manifest) for manifest in manifests]
    agents = [payload.agent_name for payload in payloads]
    if len(set(agents)) != len(agents):
        raise ValueError("source manifests contain duplicate NIXL agents")
    merged = merge_shard_tables([payload.tensors for payload in payloads])
    sources, session_to_agent, session_to_device = build_sources(merged)
    return _ResolvedSources(
        sources=sources,
        session_to_agent=session_to_agent,
        session_to_device=session_to_device,
        agent_metadata={
            payload.agent_name: payload.agent_metadata for payload in payloads
        },
    )


def _required_agent_metadata(
    plan: TransferPlan, resolved: _ResolvedSources
) -> dict[str, bytes]:
    sessions = plan.sessions()
    missing_sessions = sorted(sessions - set(resolved.session_to_agent))
    if missing_sessions:
        raise RuntimeError(
            f"transfer plan references unknown source sessions: {missing_sessions[:10]}"
        )
    needed = {resolved.session_to_agent[session] for session in sessions}
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


def _source_structure(source) -> tuple:
    """Reduce a source to the fields a reused transfer plan has baked in.

    Deliberately excludes the per-shard content digests, which are exactly what
    a refresh replaces, and includes the addresses, which a plan holds and must
    never be allowed to drift underneath.
    """
    return (
        source.dtype,
        tuple(source.global_shape),
        source.elsize,
        tuple(
            (
                shard.session,
                shard.addr,
                shard.elsize,
                tuple(shard.shard_offset),
                tuple(shard.shape),
            )
            for shard in source.shards
        ),
    )


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


def _row_major_strides(shape: tuple) -> tuple:
    strides = []
    stride = 1
    for extent in reversed(shape):
        strides.append(stride)
        stride *= int(extent)
    return tuple(reversed(strides))


def _merge_plan(target: TransferPlan, source: TransferPlan) -> None:
    target.segments.extend(source.segments)
    target.converts.extend(source.converts)
    target.full_pulls.extend(source.full_pulls)
    target.unbounded_sources.extend(source.unbounded_sources)
    for name in source.fallback:
        if name not in target.fallback:
            target.fallback.append(name)
    target.exact_descriptor_count += source.exact_descriptor_count
    target.exact_bytes += source.exact_bytes


def _plan_staged_transfer(capture: CaptureResult, sources: dict) -> TransferPlan:
    """Plan reads for each source.

    Default: minimal slice reads via plan_transfer (a partial read of a shard cannot
    be whole-shard digest-verified, so correctness rests on the coverage gate). Under
    MX_RESHARD_PUBLISH_DIGEST (verification mode): reconstruct every source that isn't
    a whole-tensor identity copy as a full pull, so _verify has a complete shard to
    digest-check.
    """
    if not envs.MX_RESHARD_PUBLISH_DIGEST:
        return plan_transfer(capture, sources)

    result = TransferPlan()
    copies_by_source: dict[str, list[RecordedCopy]] = {}
    for copy in capture.copies:
        copies_by_source.setdefault(copy.src_name, []).append(copy)

    for name in capture.unsupported:
        if name not in result.fallback:
            result.fallback.append(name)

    for name, source in sources.items():
        copies = copies_by_source.pop(name, [])
        if not copies:
            continue
        directly_recoverable = any(
            not copy.op_chain and tuple(copy.dest_shape) == tuple(source.global_shape)
            for copy in copies
        )
        if directly_recoverable:
            _merge_plan(
                result,
                plan_transfer(CaptureResult(copies=copies), {name: source}),
            )
            continue

        identity = RecordedCopy(
            src_name=name,
            op_chain=(),
            param_name=name,
            dest_offset=0,
            dest_shape=tuple(source.global_shape),
            dest_stride=_row_major_strides(source.global_shape),
            dest_dtype=source.dtype,
        )
        try:
            segments = plan_pull(
                identity,
                source.global_shape,
                source.dtype,
                source.elsize,
                source.shards,
            )
        except UnsupportedReshard as error:
            raise UnsupportedReshard(
                f"{name}: strict staged verification cannot reconstruct the "
                "complete published source"
            ) from error
        result.full_pulls.append(
            FullPullSource(
                src_name=name,
                global_shape=tuple(source.global_shape),
                dtype=source.dtype,
                elsize=source.elsize,
                segments=segments,
                copies=copies,
            )
        )
        result.exact_descriptor_count += len(segments)
        result.exact_bytes += sum(segment.nbytes for segment in segments)

    for name in copies_by_source:
        if name not in result.fallback:
            result.fallback.append(name)
    return result


class _NixlStagedTransfer:
    """Own the complete prepare-and-stage lifecycle for one generator rank."""

    def __init__(
        self,
        *,
        device_id: int,
        device: torch.device,
        agent_name: str | None = None,
        listen_port: int | None = None,
        timeout_seconds: float | None = None,
        manager: NixlTransferManager | None = None,
    ) -> None:
        self._device_id = device_id
        self._device = device
        self._timeout = float(
            envs.MX_TRANSFER_TIMEOUT
            if timeout_seconds is None
            else timeout_seconds
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
        self._workspace_mode: str | None = None

    @property
    def _bounded_arena(self) -> torch.Tensor | None:
        return self._staging_arenas[0] if self._staging_arenas else None

    @_bounded_arena.setter
    def _bounded_arena(self, value: torch.Tensor | None) -> None:
        self._staging_arenas = [] if value is None else [value]

    def _release_staging_registrations(self) -> None:
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
        self._workspace_mode = None

    def _select_workspace_mode(self, mode: str) -> None:
        """Replace released staging storage only after tearing down its agent."""
        if self._workspace_mode == mode:
            return
        if self._workspace_mode is not None:
            self.reset_workspace()
        if self._owns_manager:
            # The constructor initialized an owned agent and reset_workspace()
            # shuts it down, so a mode change has to bring it back. A borrowed
            # agent is initialized and torn down by its owner, and
            # reset_workspace() refuses to cycle one, so it needs neither.
            try:
                self._manager.initialize()
            except Exception:
                self._manager.shutdown()
                raise
        self._workspace_mode = mode

    def prepare(
        self,
        *,
        manifests: list[bytes],
        capture_layout: Callable[
            [list[tuple[str, torch.dtype, tuple[int, ...]]]],
            tuple[
                CaptureResult,
                dict[str, tuple[tuple[int, ...], torch.dtype]],
            ],
        ],
        max_staging_bytes: int | None = None,
        staging_device: str = "cuda",
        staging_buffers: int = 1,
    ) -> _PreparedNixlTransfer | _PreparedBoundedTransfer:
        """Compile one exact source version into a physical NIXL plan.

        ``max_staging_bytes`` bounds the bounded-mode arenas in total.
        ``staging_device`` places them on ``"cuda"`` (RDMA lands in VRAM and the
        commit is a device copy) or ``"cpu"`` (pinned host memory registered as
        NIXL DRAM; the commit is a host-to-device copy). ``staging_buffers`` of
        2 splits the budget across two arenas so the next batch's READ overlaps
        the current batch's commit.
        """
        if self._closed:
            raise RuntimeError("NIXL staged transfer is closed")
        if max_staging_bytes is not None and self._device.type != "cuda":
            raise ValueError("bounded NIXL staging requires a CUDA device")
        if staging_device not in ("cuda", "cpu"):
            raise ValueError("staging_device must be 'cuda' or 'cpu'")
        if (
            isinstance(staging_buffers, bool)
            or not isinstance(staging_buffers, int)
            or staging_buffers < 1
        ):
            raise ValueError("staging_buffers must be a positive integer")
        phase_started = time.perf_counter()
        resolved = _resolve_sources(manifests)
        manifest = [
            (name, source.dtype, tuple(source.global_shape))
            for name, source in resolved.sources.items()
        ]
        metrics = {"source_metadata_s": time.perf_counter() - phase_started}
        phase_started = time.perf_counter()
        capture, parameter_layout = capture_layout(manifest)
        metrics["layout_capture_s"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        plan = _plan_staged_transfer(capture, resolved.sources)
        self._validate_complete(capture, parameter_layout, plan)
        batches = None
        buffer_budget = None
        if max_staging_bytes is not None:
            # The caller's limit bounds total staging, so each arena gets a share.
            buffer_budget = max_staging_bytes // staging_buffers
            if buffer_budget <= 0:
                raise ValueError(
                    "max_staging_bytes must cover at least one byte per staging buffer"
                )
            batches = _bounded_batches(
                capture,
                parameter_layout,
                resolved.sources,
                buffer_budget,
                total_staging_bytes=max_staging_bytes,
                staging_buffers=staging_buffers,
            )
            if envs.MX_REFIT_PACK_MODULES:
                batches = _pack_bounded_batches(batches, buffer_budget)
        metrics["transfer_planning_s"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        self._select_workspace_mode(
            f"bounded:{staging_device}:{staging_buffers}"
            if batches is not None
            else "full"
        )
        required_metadata = _required_agent_metadata(plan, resolved)
        if batches is not None:
            for batch in batches:
                required_metadata.update(_required_agent_metadata(batch.plan, resolved))
        changed = {
            agent: metadata
            for agent, metadata in required_metadata.items()
            if self._loaded_agent_metadata.get(agent) != metadata
        }
        conflicting = sorted(
            agent for agent in changed if agent in self._loaded_agent_metadata
        )
        if conflicting:
            raise RuntimeError(
                "NIXL metadata changed for an already connected source agent: "
                f"{conflicting[:10]}"
            )
        _load_agent_metadata(self._manager, changed)
        self._loaded_agent_metadata.update(changed)
        host_staging = batches is not None and staging_device == "cpu"
        transport = NixlReshardTransport(
            self._manager,
            resolved.session_to_agent,
            resolved.session_to_device,
            timeout_seconds=self._timeout,
            local_mem_type=NIXL_DRAM_MEM_TYPE if host_staging else None,
        )
        if batches is not None:
            assert buffer_budget is not None
            arena_bytes = max(b.nbytes for b in batches)
            if not self._staging_arenas:
                self._staging_device = torch.device(staging_device)
                for index in range(staging_buffers):
                    arena = self._allocate_arena(arena_bytes)
                    # Keep the buffer referenced before registering it so a
                    # failed registration still has live storage to deregister
                    # when the workspace is reset.
                    self._staging_arenas.append(arena)
                    if host_staging:
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
            metrics["connection_registration_s"] = time.perf_counter() - phase_started
            prepared = _PreparedBoundedTransfer(
                batches, resolved.sources, transport, metrics
            )
            self._active = prepared
            return prepared
        self._ensure_workspace(plan, parameter_layout)
        descriptors = tuple(self._descriptors(plan))
        used_sources = {
            copy.src_name: resolved.sources[copy.src_name]
            for copy in capture.copies
            if copy.src_name in resolved.sources
        }
        prepared = _PreparedNixlTransfer(
            plan=plan,
            capture=capture,
            sources=used_sources,
            descriptors=descriptors,
            transport=transport,
        )
        self._active = prepared
        return prepared

    def iter_bounded(
        self, prepared: _PreparedBoundedTransfer, metrics: dict[str, Any]
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield verified staged batches; callers must commit before advancing.

        With one arena each batch is read, verified, yielded, and committed in
        turn. With two arenas the READ for batch ``i + 1`` is posted into the
        other arena before batch ``i`` is yielded, so the transfer overlaps the
        caller's commit. An arena is only reposted after the commit that read
        from it has been synchronized.
        """
        if self._closed or prepared is not self._active:
            raise RuntimeError("bounded NIXL transfer is no longer active")
        arenas = self._staging_arenas
        assert arenas
        metrics["staging_peak_bytes"] = sum(a.numel() for a in arenas)
        metrics["staging_buffers"] = len(arenas)
        metrics["batches"] = len(prepared.batches)
        batches = prepared.batches

        def carve(
            batch: _BoundedBatch, arena: torch.Tensor
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

        def post(index: int) -> tuple:
            batch = batches[index]
            recv, convert, full = carve(batch, arenas[index % len(arenas)])
            sources = {
                c.src_name: prepared.sources[c.src_name] for c in batch.capture.copies
            }
            chunk = _PreparedNixlTransfer(
                batch.plan,
                batch.capture,
                sources,
                tuple(self._descriptors(batch.plan, recv, full, convert)),
                prepared.transport,
            )
            started = time.perf_counter()
            posted = prepared.transport.post_reads(list(chunk.descriptors))
            return chunk, (recv, convert, full), posted, started

        pending = None
        unwinding = False
        try:
            pending = post(0)
            for index in range(len(batches)):
                chunk, buffers, posted, started = pending
                pending = None
                self._recv_buffers, self._convert_buffers, self._full_buffers = buffers
                self._active = chunk
                staged = self._complete_stage(chunk, posted, started)
                if len(arenas) > 1 and index + 1 < len(batches):
                    # The other arena's previous batch was committed and
                    # synchronized one iteration ago, so it is free to refill.
                    pending = post(index + 1)
                for key, value in staged.metrics.items():
                    metrics[key] = metrics.get(key, 0) + value
                yield staged.tensors
                # All installation reads must complete before arena reuse.
                torch.cuda.synchronize(self._device)
                if len(arenas) == 1 and index + 1 < len(batches):
                    pending = post(index + 1)
        except GeneratorExit:
            # Deliberate abandonment, not a failure, so a drain error below has
            # nothing to mask and must still be reported.
            raise
        except BaseException:
            unwinding = True
            raise
        finally:
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
                    if not unwinding:
                        raise RuntimeError(
                            "a prefetched bounded READ could not be drained, so the "
                            "staging arena may still be written; restart the "
                            "generator engine"
                        ) from error
            self._active = prepared

    def refresh_sources(
        self, prepared: _PreparedNixlTransfer, manifests: list[bytes]
    ) -> None:
        """Refresh version-specific source digests without rebuilding the plan."""
        if prepared is not self._active:
            raise RuntimeError("NIXL transfer plan is no longer active")
        resolved = _resolve_sources(manifests)
        used_sources = {
            copy.src_name: resolved.sources[copy.src_name]
            for copy in prepared.capture.copies
            if copy.src_name in resolved.sources
        }
        if set(used_sources) != set(prepared.sources):
            raise RuntimeError("source tensor set changed while reusing transfer plan")
        if any(
            _source_structure(source) != _source_structure(prepared.sources[name])
            for name, source in used_sources.items()
        ):
            raise RuntimeError("source geometry changed while reusing transfer plan")
        prepared.sources.clear()
        prepared.sources.update(used_sources)

    @staticmethod
    def _validate_complete(
        capture: CaptureResult,
        parameter_layout: dict[str, tuple[tuple[int, ...], torch.dtype]],
        plan: TransferPlan,
    ) -> None:
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
        plan: TransferPlan,
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
            self._manager.register_tensors(
                {
                    f"__convert__{name}": tensor
                    for name, tensor in self._convert_buffers.items()
                }
            )
            self._convert_registered = True
        if full_expected and not self._full_registered:
            self._manager.register_tensors(
                {
                    f"__full__{name}": tensor
                    for name, tensor in self._full_buffers.items()
                }
            )
            self._full_registered = True
        if not self._registered_recv_params and recv_params:
            self._manager.register_tensors(self._recv_buffers)
            self._registered_recv_params = recv_params

    def _descriptors(
        self,
        plan: TransferPlan,
        recv: dict[str, torch.Tensor] | None = None,
        full: dict[str, torch.Tensor] | None = None,
        convert: dict[str, torch.Tensor] | None = None,
    ) -> list[ReadDescriptor]:
        recv_buffers = self._recv_buffers if recv is None else recv
        full_buffers = self._full_buffers if full is None else full
        convert_buffers = self._convert_buffers if convert is None else convert
        descriptors = exact_descriptors(
            plan, lambda name: recv_buffers[name].data_ptr()
        )
        descriptors.extend(
            ReadDescriptor(
                session=segment.session,
                src_addr=segment.src_addr,
                dst_addr=full_buffers[pull.src_name].data_ptr() + segment.dst_byte,
                nbytes=segment.nbytes,
            )
            for pull in plan.full_pulls
            for segment in pull.segments
        )
        descriptors.extend(
            ReadDescriptor(
                session=segment.session,
                src_addr=segment.src_addr,
                dst_addr=convert_buffers[conv.param_name].data_ptr()
                + segment.dst_byte,
                nbytes=segment.nbytes,
            )
            for conv in plan.converts
            for segment in conv.segments
        )
        return descriptors

    def stage(self, prepared: _PreparedNixlTransfer) -> _StagedNixlWeights:
        """Pull, reconstruct, convert, and verify without touching live weights."""
        if self._closed:
            raise RuntimeError("NIXL staged transfer is closed")
        if prepared is not self._active:
            raise RuntimeError("NIXL transfer plan is no longer active")
        started = time.perf_counter()
        posted = prepared.transport.post_reads(list(prepared.descriptors))
        return self._complete_stage(prepared, posted, started)

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
        for full in prepared.plan.full_pulls:
            source = self._full_buffers[full.src_name]
            for copy in full.copies:
                destination = self._recv_buffers[copy.param_name].as_strided(
                    copy.dest_shape,
                    copy.dest_stride,
                    self._recv_buffers[copy.param_name].storage_offset()
                    + copy.dest_offset,
                )
                destination.copy_(_replay_ops(source, copy.op_chain))
        for convert in prepared.plan.converts:
            self._recv_buffers[convert.param_name].copy_(
                self._convert_buffers[convert.param_name]
            )
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
            len(prepared.plan.segments),
            len(prepared.plan.full_pulls),
            len(prepared.plan.converts),
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
                "full_pull_sources": len(prepared.plan.full_pulls),
                "converts": len(prepared.plan.converts),
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
                "P2P source published an unusable metadata endpoint: "
                f"{endpoint!r}"
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
                if not shard.digest:
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
                if actual != shard.digest:
                    raise RuntimeError(
                        f"staged weight digest mismatch for source {name!r} "
                        f"at offset {tuple(shard.shard_offset)}"
                    )

    def close(self) -> None:
        if self._closed:
            return
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
