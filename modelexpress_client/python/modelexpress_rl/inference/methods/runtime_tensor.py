# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-load runtime tensor transfer over NIXL."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from modelexpress import envs, p2p_pb2
from modelexpress.metadata.worker_server import TensorReadLease, prepare_tensor_read
from modelexpress.nixl_transfer import NixlTransferManager
from modelexpress.types import ManifestMismatchError, TensorDescriptor
from modelexpress_rl.refit.reshard import throughput
from modelexpress_rl.refit.timing import add_refit_bytes, add_refit_duration

from ...train import WeightPayloadFormat
from ..plan import (
    GeneratorPeerUpdateSource,
    MethodCapabilities,
    PreparedArtifact,
    PreparedRuntimeTensors,
    ResolvedSource,
    UpdateMethod,
    WeightSource,
)


logger = logging.getLogger("modelexpress.reshard.runtime_tensor")


@dataclass
class _PreparedRuntimeTensorRead:
    """Peer selection retained until the caller reaches its safe point."""

    source: GeneratorPeerUpdateSource
    tensor_read: TensorReadLease | None
    tensors: dict[str, torch.Tensor]
    metrics: dict[str, Any] = field(default_factory=dict)
    transfer_started: bool = False

    def close(self) -> None:
        try:
            if self.tensor_read is not None:
                self.tensor_read.close()
        finally:
            self.tensor_read = None

    def mark_transfer_started(self) -> None:
        self.transfer_started = True


class RuntimeTensorNixlUpdateMethod(UpdateMethod):
    """Validate generator peers and read into live runtime tensors.

    Acquire and retain a donor lease during preparation, then transfer at the caller's
    safe point. Track whether a failed attempt may have written live storage. Borrow
    the engine's existing NIXL manager; the engine retains its lifecycle ownership.
    """

    def __init__(
        self,
        *,
        manager: NixlTransferManager,
        device_id: int,
        runtime_tensors: dict[str, torch.Tensor],
        timeout_seconds: float | None = None,
    ) -> None:
        self._manager = manager
        self._device_id = device_id
        self._timeout = float(
            envs.MX_TRANSFER_TIMEOUT if timeout_seconds is None else timeout_seconds
        )
        self._closed = False
        self._runtime_tensors = runtime_tensors
        self._active_read: _PreparedRuntimeTensorRead | None = None

    @property
    def capabilities(self) -> MethodCapabilities:
        return MethodCapabilities(
            payload_formats=frozenset({WeightPayloadFormat.FULL_TENSOR}),
            sources=frozenset({WeightSource.GENERATOR}),
            artifact_type=PreparedRuntimeTensors,
        )

    def prepare(self, *, version, source: ResolvedSource) -> PreparedArtifact:
        if self._closed:
            raise RuntimeError("runtime tensor method is closed")
        if self._active_read is not None:
            raise RuntimeError("release staged weight before staging another version")
        if not isinstance(source, GeneratorPeerUpdateSource):
            raise TypeError("runtime tensor method requires a generator source")

        tensor_read = self._prepare_peer_read(source)
        self._active_read = _PreparedRuntimeTensorRead(
            source=source,
            tensor_read=tensor_read,
            tensors=self._runtime_tensors,
        )
        return PreparedRuntimeTensors(staged=self._active_read)

    @contextmanager
    def installation_context(self, prepared: PreparedArtifact):
        if not isinstance(prepared, PreparedRuntimeTensors):
            raise TypeError("runtime tensor method requires a prepared peer read")
        read = prepared.staged
        if read is not self._active_read:
            raise RuntimeError("runtime tensor read is no longer active")
        tensor_read = read.tensor_read
        if tensor_read is None:
            if read.transfer_started:
                raise RuntimeError("runtime tensor transfer cannot be retried")
            tensor_read = self._prepare_peer_read(read.source)
            read.tensor_read = tensor_read
        try:
            read.metrics.update(self._receive_peer(read))
        finally:
            read.close()
        _attribute_transfer(read.metrics)
        yield

    def mutated_during_installation_context(
        self,
        prepared: PreparedArtifact,
    ) -> bool:
        if not isinstance(prepared, PreparedRuntimeTensors):
            return False
        read = prepared.staged
        return read is self._active_read and read.transfer_started

    def release(self, prepared: PreparedArtifact) -> None:
        if not isinstance(prepared, PreparedRuntimeTensors):
            raise TypeError("runtime tensor method requires a prepared peer read")
        if prepared.staged is not self._active_read:
            raise RuntimeError("runtime tensor read is no longer active")
        try:
            self._active_read.close()
        finally:
            self._active_read = None

    def close(self) -> None:
        try:
            if self._active_read is not None:
                self._active_read.close()
        finally:
            self._active_read = None
            self._closed = True

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

    def _prepare_peer_read(self, source: GeneratorPeerUpdateSource) -> TensorReadLease:
        """Validate and retain a donor lease until apply or release."""
        if not source.worker.worker_grpc_endpoint:
            raise RuntimeError("generator P2P source has no tensor lease endpoint")
        lease, _ = prepare_tensor_read(
            source.worker.worker_grpc_endpoint,
            source.mx_source_id,
            worker_id=source.worker_id,
            timeout=self._timeout,
        )
        try:
            self._validate_peer_manifest(lease.manifest, self._runtime_tensors)
            self._peer_endpoint(lease.manifest)
        except BaseException:
            lease.close()
            raise
        return lease

    def _receive_peer(self, read: _PreparedRuntimeTensorRead) -> dict[str, Any]:
        """Pull a peer's runtime tensors directly into live engine storage."""
        started = time.perf_counter()
        manifest = read.tensor_read.manifest
        source_tensors = self._validate_peer_manifest(
            manifest,
            read.tensors,
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
                    destination_tensors=read.tensors,
                    on_transfer_start=read.mark_transfer_started,
                )
            )
        finally:
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


def _attribute_transfer(metrics: dict[str, float]) -> None:
    add_refit_bytes(metrics.get("bytes_received", 0))
    if "wire_s" in metrics:
        add_refit_duration("wire_transfer", metrics["wire_s"])
    if "reconstruct_s" in metrics:
        add_refit_duration("receive_sync", metrics["reconstruct_s"])


__all__ = ["RuntimeTensorNixlUpdateMethod"]
