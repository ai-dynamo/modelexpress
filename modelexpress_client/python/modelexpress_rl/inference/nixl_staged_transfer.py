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
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from modelexpress import envs
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


@dataclass(frozen=True)
class _ResolvedSources:
    sources: dict
    session_to_agent: dict
    session_to_device: dict
    agent_endpoints: dict


@dataclass(frozen=True)
class _PreparedNixlTransfer:
    """One immutable physical plan over reusable registered destinations."""

    plan: TransferPlan
    capture: CaptureResult
    sources: dict
    descriptors: tuple[ReadDescriptor, ...]
    transport: NixlReshardTransport
    generation: int


@dataclass(frozen=True)
class _StagedNixlWeights:
    """Verified load-layout tensors produced by one completed transfer."""

    tensors: dict[str, torch.Tensor]
    metrics: dict[str, Any]
    generation: int


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
        agent_endpoints={
            payload.agent_name: payload.metadata_endpoint for payload in payloads
        },
    )


def _required_endpoints(plan: TransferPlan, resolved: _ResolvedSources) -> dict:
    sessions = plan.sessions()
    missing_sessions = sorted(sessions - set(resolved.session_to_agent))
    if missing_sessions:
        raise RuntimeError(
            f"transfer plan references unknown source sessions: {missing_sessions[:10]}"
        )
    needed = {resolved.session_to_agent[session] for session in sessions}
    missing = sorted(needed - set(resolved.agent_endpoints))
    if missing:
        raise RuntimeError(
            "transfer plan references source agents without metadata endpoints: "
            f"{missing[:10]}"
        )
    return {
        agent: endpoint
        for agent, endpoint in resolved.agent_endpoints.items()
        if agent in needed
    }


def _handshake(manager: NixlTransferManager, endpoints: dict) -> None:
    """Load every required peer registration within one bounded retry budget."""
    pending = deque(endpoints.items())
    attempts = {agent: 0 for agent in endpoints}
    last_error: dict[str, Exception] = {}
    deadline = time.monotonic() + envs.MX_RESHARD_HANDSHAKE_TIMEOUT_S
    stalled = 0
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            outstanding = ", ".join(
                f"{agent}@{endpoint} ({attempts[agent]} attempt(s), "
                f"last={last_error.get(agent)!r})"
                for agent, endpoint in pending
            )
            raise RuntimeError(
                "NIXL peer handshake did not complete before the deadline: "
                f"{outstanding}"
            )

        agent, endpoint = pending.popleft()
        attempts[agent] += 1
        try:
            host, port = endpoint.rsplit(":", 1)
            manager.fetch_remote_and_wait(
                agent,
                host,
                int(port),
                timeout_seconds=min(envs.MX_RESHARD_HANDSHAKE_ATTEMPT_S, remaining),
            )
        except Exception as error:  # noqa: BLE001 - every dial error is retryable
            last_error[agent] = error
            pending.append((agent, endpoint))
            stalled += 1
            if stalled >= len(pending):
                time.sleep(
                    min(
                        envs.MX_RESHARD_HANDSHAKE_BACKOFF_S,
                        max(0.0, deadline - time.monotonic()),
                    )
                )
                stalled = 0
            continue
        stalled = 0


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
        timeout_seconds: float = 1200.0,
    ) -> None:
        self._device = device
        self._timeout = timeout_seconds
        self._manager = NixlTransferManager(
            agent_name=agent_name,
            device_id=device_id,
        )
        try:
            self._manager.initialize()
        except Exception:
            self._manager.shutdown()
            raise
        self._recv_buffers: dict[str, torch.Tensor] = {}
        self._convert_buffers: dict[str, torch.Tensor] = {}
        self._full_buffers: dict[str, torch.Tensor] = {}
        self._registered_recv_params: set[str] = set()
        self._convert_registered = False
        self._full_registered = False
        self._active: _PreparedNixlTransfer | None = None
        self._generation = 0
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
        _handshake(self._manager, _required_endpoints(plan, resolved))
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
        self._generation += 1
        prepared = _PreparedNixlTransfer(
            plan=plan,
            capture=capture,
            sources=used_sources,
            descriptors=descriptors,
            transport=transport,
            generation=self._generation,
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

        segment_params = {segment.param_name for segment in plan.segments}
        if (
            self._registered_recv_params
            and self._registered_recv_params != segment_params
        ):
            raise RuntimeError(
                "direct receive parameter set changed; restart the generator engine"
            )
        if not self._registered_recv_params and segment_params:
            self._manager.register_tensors(
                {f"__recv__{name}": self._recv_buffers[name] for name in segment_params}
            )
            self._registered_recv_params = segment_params
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
            generation=prepared.generation,
        )

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
        self._manager.shutdown()


__all__: list[str] = []
