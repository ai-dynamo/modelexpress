# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-version staged NIXL transfer for RL generator workers.

This module owns the state that makes a pull transfer correct: the selected
source manifests, the physical plan, registered destination buffers, peer
metadata, transfer completion, and verification. It deliberately does not know
how an inference engine captures its load layout or installs received weights.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from modelexpress import p2p_pb2
from modelexpress.client import MxClientBase
from modelexpress.load_strategy.base import unpublish_metadata_for_worker
from modelexpress.metadata.payload import worker_tensor_descriptors
from modelexpress.metadata.publish import publish_metadata_and_ready
from modelexpress.nixl_transfer import NixlTransferManager
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
from modelexpress.types import TensorDescriptor


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
    """Plan each source independently, reconstructing only unverifiable views."""
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
        agent_name: str,
        device_id: int,
        device: torch.device,
        listen_port: int,
        timeout_seconds: float = 1200.0,
    ) -> None:
        self._device_id = device_id
        self._device = device
        self._timeout = timeout_seconds
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
        self._active: _PreparedNixlTransfer | None = None
        self._loaded_agent_metadata: dict[str, bytes] = {}
        self._published_peer_rank: int | None = None
        self._closed = False

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
    ) -> _PreparedNixlTransfer:
        """Compile one exact source version into a physical NIXL plan."""
        if self._closed:
            raise RuntimeError("NIXL staged transfer is closed")
        resolved = _resolve_sources(manifests)
        manifest = [
            (name, source.dtype, tuple(source.global_shape))
            for name, source in resolved.sources.items()
        ]
        capture, parameter_layout = capture_layout(manifest)
        plan = _plan_staged_transfer(capture, resolved.sources)
        self._validate_complete(capture, parameter_layout, plan)
        required_metadata = _required_agent_metadata(plan, resolved)
        changed = {
            agent: metadata
            for agent, metadata in required_metadata.items()
            if self._loaded_agent_metadata.get(agent) != metadata
        }
        conflicting = sorted(
            agent
            for agent in changed
            if agent in self._loaded_agent_metadata
        )
        if conflicting:
            raise RuntimeError(
                "NIXL metadata changed for an already connected source agent: "
                f"{conflicting[:10]}"
            )
        _load_agent_metadata(self._manager, changed)
        self._loaded_agent_metadata.update(changed)
        transport = NixlReshardTransport(
            self._manager,
            resolved.session_to_agent,
            resolved.session_to_device,
            timeout_seconds=self._timeout,
        )
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
                f"causes={causes}"
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
        if (
            self._registered_recv_params
            and self._registered_recv_params != recv_params
        ):
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

    def _descriptors(self, plan: TransferPlan) -> list[ReadDescriptor]:
        descriptors = exact_descriptors(
            plan, lambda name: self._recv_buffers[name].data_ptr()
        )
        descriptors.extend(
            ReadDescriptor(
                session=segment.session,
                src_addr=segment.src_addr,
                dst_addr=self._full_buffers[full.src_name].data_ptr()
                + segment.dst_byte,
                nbytes=segment.nbytes,
            )
            for full in plan.full_pulls
            for segment in full.segments
        )
        descriptors.extend(
            ReadDescriptor(
                session=segment.session,
                src_addr=segment.src_addr,
                dst_addr=self._convert_buffers[convert.param_name].data_ptr()
                + segment.dst_byte,
                nbytes=segment.nbytes,
            )
            for convert in plan.converts
            for segment in convert.segments
        )
        return descriptors

    @torch.no_grad()
    def stage(self, prepared: _PreparedNixlTransfer) -> _StagedNixlWeights:
        """Pull, reconstruct, convert, and verify without touching live weights."""
        if self._closed:
            raise RuntimeError("NIXL staged transfer is closed")
        if prepared is not self._active:
            raise RuntimeError("NIXL transfer plan is no longer active")
        started = time.perf_counter()
        prepared.transport.read(list(prepared.descriptors))
        wire_seconds = time.perf_counter() - started

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
        self._verify(prepared)

        return _StagedNixlWeights(
            tensors=self._recv_buffers,
            metrics={
                "bytes_received": sum(d.nbytes for d in prepared.descriptors),
                "segments": len(prepared.descriptors),
                "wire_s": round(wire_seconds, 6),
                "full_pull_sources": len(prepared.plan.full_pulls),
                "converts": len(prepared.plan.converts),
            },
        )

    def stage_peer(
        self,
        *,
        source: p2p_pb2.WorkerMetadata,
        parameter_layout: dict[str, tuple[tuple[int, ...], torch.dtype]],
    ) -> _StagedNixlWeights:
        """Pull an identical-rank peer's canonical staging buffers."""
        if self._closed:
            raise RuntimeError("NIXL staged transfer is closed")
        self.unpublish_peer()
        self._ensure_buffers(
            self._recv_buffers,
            parameter_layout,
            label="receive-buffer",
        )
        recv_params = set(parameter_layout)
        if self._registered_recv_params and self._registered_recv_params != recv_params:
            raise RuntimeError(
                "receive parameter set changed; restart the generator engine"
            )
        if not self._registered_recv_params:
            self._manager.register_tensors(self._recv_buffers)
            self._registered_recv_params = recv_params

        source_tensors = [
            TensorDescriptor(
                name=tensor.name,
                addr=tensor.addr,
                size=tensor.size,
                device_id=tensor.device_id,
                dtype=tensor.dtype,
            )
            for tensor in worker_tensor_descriptors(source)
        ]
        if not source_tensors:
            raise RuntimeError("P2P source has no tensor descriptors")

        remote_agent_name: str | None = None
        started = time.perf_counter()
        try:
            if source.worker_grpc_endpoint:
                endpoint = source.metadata_endpoint
                try:
                    host, port_text = endpoint.rsplit(":", 1)
                    port = int(port_text)
                except ValueError as error:
                    raise RuntimeError(
                        f"P2P source published an unusable metadata endpoint: "
                        f"{endpoint!r}"
                    ) from error
                if not host or not 1 <= port <= 65535:
                    raise RuntimeError(
                        f"P2P source published an unusable metadata endpoint: "
                        f"{endpoint!r}"
                    )
                remote_agent_name = source.agent_name
                self._manager.fetch_remote_and_wait(
                    remote_agent_name=remote_agent_name,
                    ip=host,
                    port=port,
                    timeout_seconds=self._timeout,
                )
            else:
                remote_agent_name = self._manager.add_remote_agent(
                    source.nixl_metadata
                )
            bytes_received, tensor_count, wire_seconds = (
                self._manager.receive_from_source(
                    source_metadata=b"",
                    source_tensors=source_tensors,
                    timeout_seconds=self._timeout,
                    remote_agent_name=remote_agent_name,
                    require_exact_match=True,
                    destination_tensors=self._recv_buffers,
                )
            )
        finally:
            if remote_agent_name is not None:
                self._manager.remove_remote_agent(remote_agent_name)

        self._active = None
        return _StagedNixlWeights(
            tensors=self._recv_buffers,
            metrics={
                "bytes_received": bytes_received,
                "segments": tensor_count,
                "wire_s": round(wire_seconds, 6),
                "peer_s": round(time.perf_counter() - started, 6),
            },
        )

    def publish_peer(
        self,
        *,
        staged: _StagedNixlWeights,
        identity: p2p_pb2.SourceIdentity,
        p2p_client: MxClientBase,
        worker_rank: int,
        worker_id: str,
        accelerator: str,
    ) -> None:
        """Advertise verified canonical buffers for the applied version."""
        previous_rank = self._published_peer_rank
        self.unpublish_peer()
        # Supersede any boot-time source owned by this rank before binding the
        # shared publication slot to the exact WeightVersion identity.
        if previous_rank != worker_rank:
            unpublish_metadata_for_worker(
                worker_rank=worker_rank,
                device_id=self._device_id,
            )
        publish_metadata_and_ready(
            p2p_client,
            self._manager,
            staged.tensors,
            worker_rank,
            self._device_id,
            identity,
            worker_id,
            accelerator=accelerator,
        )
        self._published_peer_rank = worker_rank

    def unpublish_peer(self) -> None:
        """Stop advertising buffers before they are reused by another stage."""
        if self._published_peer_rank is None:
            return
        unpublish_metadata_for_worker(
            worker_rank=self._published_peer_rank,
            device_id=self._device_id,
        )
        self._published_peer_rank = None

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
        self.unpublish_peer()
        self._closed = True
        self._manager.shutdown()


__all__: list[str] = []
