# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Per-worker gRPC server for P2P manifest exchange.

When MX_P2P_METADATA=1, each source worker starts a WorkerGrpcServer
that serves its tensor descriptors directly to target workers via the
GetTensorManifest RPC. Tensor readers acquire a bounded lease before accessing
published storage. Artifact sources can serve their sealed file manifest through
GetArtifactManifestHeader/GetArtifactManifestChunks and coordinate NIXL file
chunk transfers through PrepareArtifactChunk/ReleaseArtifactChunk.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from concurrent import futures
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Condition, Lock
from typing import Any

import grpc

from .. import envs, p2p_pb2, p2p_pb2_grpc

logger = logging.getLogger("modelexpress.metadata.worker_server")

# Number of chunk metadata records per GetArtifactManifestChunks response.
# This is not the artifact byte chunk size; 1024 keeps metadata responses bounded
# while avoiding one RPC per transfer chunk.
_ARTIFACT_CHUNK_METADATA_PAGE_SIZE = 1024


def _tensor_read_lease_timeout_seconds() -> int:
    """Cover one NIXL metadata handshake and one bounded tensor receive."""
    transfer_timeout = max(1, envs.MX_TRANSFER_TIMEOUT)
    return math.ceil(2 * max(120, transfer_timeout) + 30)


@dataclass
class _ArtifactSource:
    manifests: dict[str, p2p_pb2.ArtifactManifest]
    chunk_manager: Any


class _TensorReadLeases:
    """Protect published tensor storage while peers are reading it."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._accepting = True
        self._leases: dict[str, float] = {}

    def prepare(self, timeout_seconds: int) -> str:
        if timeout_seconds <= 0:
            raise ValueError("tensor read lease timeout must be positive")
        with self._condition:
            self._drop_expired_locked()
            if not self._accepting:
                raise RuntimeError("tensor source is draining")
            lease_id = uuid.uuid4().hex
            self._leases[lease_id] = time.monotonic() + timeout_seconds
            return lease_id

    def ensure_accepting(self) -> None:
        with self._condition:
            if not self._accepting:
                raise RuntimeError("tensor source is draining")

    def release(self, lease_id: str) -> None:
        with self._condition:
            self._drop_expired_locked()
            if self._leases.pop(lease_id, None) is None:
                raise KeyError(lease_id)
            self._condition.notify_all()

    def drain(self, timeout: float) -> None:
        if timeout < 0:
            raise ValueError("tensor read drain timeout must not be negative")
        deadline = time.monotonic() + timeout
        with self._condition:
            self._accepting = False
            while self._leases:
                self._drop_expired_locked()
                if not self._leases:
                    return
                now = time.monotonic()
                remaining = deadline - now
                if remaining <= 0:
                    self._accepting = True
                    self._condition.notify_all()
                    raise TimeoutError(
                        f"timed out waiting for {len(self._leases)} tensor readers"
                    )
                next_expiry = min(self._leases.values()) - now
                self._condition.wait(timeout=min(remaining, max(next_expiry, 0.0)))

    def _drop_expired_locked(self) -> None:
        now = time.monotonic()
        expired = [
            lease_id
            for lease_id, expires_at in self._leases.items()
            if expires_at <= now
        ]
        for lease_id in expired:
            self._leases.pop(lease_id, None)
        if expired:
            self._condition.notify_all()


class WorkerServiceServicer(p2p_pb2_grpc.WorkerServiceServicer):
    """Serves manifests for a single source worker."""

    def __init__(
        self,
        tensor_protos: list[p2p_pb2.TensorDescriptor],
        mx_source_id: str | None,
        metadata_endpoint: str = "",
        agent_name: str = "",
        worker_rank: int = 0,
        accelerator: str = "",
        artifact_manifests: Mapping[str, p2p_pb2.ArtifactManifest] | None = None,
        artifact_chunk_manager: Any | None = None,
        worker_id: str = "",
    ):
        self._tensor_protos = tensor_protos
        self._mx_source_id = mx_source_id
        self._metadata_endpoint = metadata_endpoint
        self._agent_name = agent_name
        self._worker_rank = worker_rank
        self._accelerator = accelerator
        self._worker_id = worker_id
        self._tensor_read_leases = _TensorReadLeases()
        self._tensor_read_lease_timeout = _tensor_read_lease_timeout_seconds()
        self._artifact_sources: dict[str, _ArtifactSource] = {}
        self._artifact_lock = Lock()
        if artifact_manifests and mx_source_id:
            self._artifact_sources[mx_source_id] = _ArtifactSource(
                manifests=dict(artifact_manifests),
                chunk_manager=artifact_chunk_manager,
            )

    def set_mx_source_id(self, mx_source_id: str) -> None:
        self._mx_source_id = mx_source_id

    def register_artifact_source(
        self,
        mx_source_id: str,
        artifact_id: str,
        manifest: p2p_pb2.ArtifactManifest,
        artifact_chunk_manager: Any,
    ) -> None:
        with self._artifact_lock:
            source = self._artifact_sources.get(mx_source_id)
            if source is None:
                self._artifact_sources[mx_source_id] = _ArtifactSource(
                    manifests={artifact_id: manifest},
                    chunk_manager=artifact_chunk_manager,
                )
            else:
                source.manifests[artifact_id] = manifest

    def unregister_artifact_source(self, mx_source_id: str, artifact_id: str) -> None:
        with self._artifact_lock:
            source = self._artifact_sources.get(mx_source_id)
            if source is None:
                return
            source.manifests.pop(artifact_id, None)
            if not source.manifests:
                self._artifact_sources.pop(mx_source_id, None)

    def GetTensorManifest(self, request, context):
        self._validate_tensor_source(
            request.mx_source_id,
            request.worker_id if request.HasField("worker_id") else None,
            context,
        )
        try:
            self._tensor_read_leases.ensure_accepting()
        except RuntimeError as error:
            context.abort(grpc.StatusCode.UNAVAILABLE, str(error))
        response = self._tensor_manifest_response()
        logger.info(
            f"GetTensorManifest served: {len(self._tensor_protos)} tensors, "
            f"{response.ByteSize()} bytes (worker_rank={self._worker_rank})"
        )
        return response

    def PrepareTensorRead(self, request, context):
        self._validate_tensor_source(
            request.mx_source_id,
            request.worker_id if request.HasField("worker_id") else None,
            context,
        )
        lease_id = self._prepare_tensor_read_lease(context)
        response = p2p_pb2.PrepareTensorReadResponse(
            lease_id=lease_id,
            manifest=self._tensor_manifest_response(),
        )
        logger.info(
            f"PrepareTensorRead served: {len(self._tensor_protos)} tensors, "
            f"lease_id={lease_id} (worker_rank={self._worker_rank})"
        )
        return response

    def ReleaseTensorRead(self, request, context):
        self._validate_tensor_source(
            request.mx_source_id,
            request.worker_id if request.HasField("worker_id") else None,
            context,
        )
        try:
            self._tensor_read_leases.release(request.lease_id)
        except KeyError:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"tensor read lease not found or expired: {request.lease_id}",
            )
        logger.info(
            f"ReleaseTensorRead served: lease_id={request.lease_id} "
            f"(worker_rank={self._worker_rank})"
        )
        return p2p_pb2.ReleaseTensorReadResponse()

    def drain_tensor_reads(self, timeout: float) -> None:
        self._tensor_read_leases.drain(timeout)

    def _prepare_tensor_read_lease(self, context) -> str:
        try:
            return self._tensor_read_leases.prepare(
                self._tensor_read_lease_timeout
            )
        except RuntimeError as error:
            context.abort(grpc.StatusCode.UNAVAILABLE, str(error))

    def _tensor_manifest_response(self) -> p2p_pb2.GetTensorManifestResponse:
        response = p2p_pb2.GetTensorManifestResponse(
            tensors=self._tensor_protos,
            mx_source_id=self._mx_source_id or "",
            metadata_endpoint=self._metadata_endpoint,
            agent_name=self._agent_name,
            worker_rank=self._worker_rank,
            accelerator=self._accelerator,
        )
        if self._worker_id:
            response.worker_id = self._worker_id
        return response

    def PrepareArtifactChunk(self, request, context):
        source_id, artifact_id, manifest, artifact_chunk_manager = (
            self._select_artifact_source(
                request.mx_source_id,
                request.artifact_id,
                context,
            )
        )
        if artifact_chunk_manager is None:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "artifact chunk transfer is not enabled for this worker",
            )
        if request.chunk_index >= len(manifest.chunks):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"chunk_index {request.chunk_index} exceeds chunk_count "
                f"{len(manifest.chunks)}",
            )

        chunk = manifest.chunks[request.chunk_index]
        try:
            lease_id, source, source_metadata = artifact_chunk_manager.prepare(
                manifest,
                artifact_id,
                chunk,
            )
        except FileNotFoundError as exc:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"failed to prepare artifact chunk {chunk.chunk_index}: {exc}",
            )
        except OSError as exc:
            context.abort(
                grpc.StatusCode.INTERNAL,
                f"failed to prepare artifact chunk {chunk.chunk_index}: {exc}",
            )
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except RuntimeError as exc:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))

        response = p2p_pb2.PrepareArtifactChunkResponse(
            mx_source_id=source_id,
            artifact_id=artifact_id,
            lease_id=lease_id,
            chunk=chunk,
            source=source,
            source_metadata=source_metadata,
        )
        logger.info(
            f"PrepareArtifactChunk served: chunk {chunk.chunk_index} "
            f"({source.length} bytes, lease_id={lease_id})"
        )
        return response

    def ReleaseArtifactChunk(self, request, context):
        source_id, artifact_id, _, artifact_chunk_manager = (
            self._select_artifact_source(
                request.mx_source_id,
                request.artifact_id,
                context,
            )
        )
        if artifact_chunk_manager is None:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "artifact chunk transfer is not enabled for this worker",
            )
        try:
            released_artifact_id, chunk = artifact_chunk_manager.release(
                request.lease_id,
            )
        except KeyError:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"artifact chunk lease not found: {request.lease_id}",
            )
        if released_artifact_id != artifact_id:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"artifact_id mismatch for lease {request.lease_id}",
            )

        response = p2p_pb2.ReleaseArtifactChunkResponse(
            mx_source_id=source_id,
            artifact_id=artifact_id,
            chunk=chunk,
        )
        logger.info(
            f"ReleaseArtifactChunk served: chunk {chunk.chunk_index} "
            f"(lease_id={request.lease_id})"
        )
        return response

    def GetArtifactManifestHeader(self, request, context):
        source_id, artifact_id, manifest, _ = self._select_artifact_source(
            request.mx_source_id,
            request.artifact_id,
            context,
        )
        response = p2p_pb2.GetArtifactManifestHeaderResponse(
            mx_source_id=source_id,
            artifact_id=artifact_id,
            manifest_version=manifest.manifest_version,
            mx_source_type=manifest.mx_source_type,
            total_size=sum(file.size for file in manifest.files),
            file_count=len(manifest.files),
            chunk_count=len(manifest.chunks),
            chunk_size=manifest.chunk_size,
            metadata_endpoint=self._metadata_endpoint,
            agent_name=self._agent_name,
            worker_rank=self._worker_rank,
            files=manifest.files,
        )
        logger.info(
            f"GetArtifactManifestHeader served: {len(manifest.files)} files, "
            f"{response.ByteSize()} bytes"
        )
        return response

    def GetArtifactManifestChunks(self, request, context):
        source_id, artifact_id, manifest, _ = self._select_artifact_source(
            request.mx_source_id,
            request.artifact_id,
            context,
        )
        start = request.start_chunk_index
        if start > len(manifest.chunks):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"start_chunk_index {start} exceeds chunk_count {len(manifest.chunks)}",
            )
        max_chunks = min(
            request.max_chunks or _ARTIFACT_CHUNK_METADATA_PAGE_SIZE,
            _ARTIFACT_CHUNK_METADATA_PAGE_SIZE,
        )
        end = min(start + max_chunks, len(manifest.chunks))
        response = p2p_pb2.GetArtifactManifestChunksResponse(
            mx_source_id=source_id,
            artifact_id=artifact_id,
            start_chunk_index=start,
            chunks=manifest.chunks[start:end],
            next_page_token=str(end) if end < len(manifest.chunks) else "",
        )
        logger.info(
            f"GetArtifactManifestChunks served: chunks {start}:{end} of "
            f"{len(manifest.chunks)}, {response.ByteSize()} bytes"
        )
        return response

    def _select_artifact_source(
        self,
        mx_source_id: str,
        artifact_id: str,
        context,
    ) -> tuple[str, str, p2p_pb2.ArtifactManifest, Any]:
        with self._artifact_lock:
            if not self._artifact_sources:
                source = None
            elif mx_source_id:
                source = self._artifact_sources.get(mx_source_id)
                if source is None:
                    context.abort(
                        grpc.StatusCode.FAILED_PRECONDITION,
                        f"artifact mx_source_id not available: {mx_source_id}",
                    )
            elif len(self._artifact_sources) == 1:
                mx_source_id, source = next(iter(self._artifact_sources.items()))
            else:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "mx_source_id is required when multiple artifact sources are available",
                )

            if source is None:
                manifests = {}
                artifact_chunk_manager = None
            else:
                manifests = dict(source.manifests)
                artifact_chunk_manager = source.chunk_manager

        if not manifests:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                "artifact manifest is not available for this worker",
            )
        if not artifact_id:
            if len(manifests) != 1:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "artifact_id is required when multiple artifact manifests are available",
                )
            artifact_id = next(iter(manifests))

        manifest = manifests.get(artifact_id)
        if manifest is None:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"artifact_id not available: {artifact_id}",
            )
        return mx_source_id, artifact_id, manifest, artifact_chunk_manager

    def _validate_tensor_source(
        self,
        mx_source_id: str,
        worker_id: str | None,
        context,
    ) -> None:
        if not mx_source_id:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "mx_source_id is required for tensor reads",
            )
        if mx_source_id != self._mx_source_id:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"mx_source_id mismatch: expected {self._mx_source_id}, "
                f"got {mx_source_id}",
            )
        if worker_id is not None and worker_id != self._worker_id:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"worker_id mismatch: expected {self._worker_id}, got {worker_id}",
            )


class WorkerGrpcServer:
    """Manages a gRPC WorkerService on a source worker."""

    def __init__(
        self,
        tensor_protos: list[p2p_pb2.TensorDescriptor],
        mx_source_id: str | None,
        port: int = 0,
        metadata_endpoint: str = "",
        agent_name: str = "",
        worker_rank: int = 0,
        accelerator: str = "",
        artifact_manifests: Mapping[str, p2p_pb2.ArtifactManifest] | None = None,
        artifact_chunk_manager: Any | None = None,
        max_workers: int = 4,
        worker_id: str = "",
    ):
        if max_workers <= 0:
            raise ValueError("worker gRPC max_workers must be positive")
        self._tensor_protos = tensor_protos
        self._mx_source_id = mx_source_id
        self._requested_port = port
        self._metadata_endpoint = metadata_endpoint
        self._agent_name = agent_name
        self._worker_rank = worker_rank
        self._accelerator = accelerator
        self._artifact_manifests = dict(artifact_manifests or {})
        self._artifact_chunk_manager = artifact_chunk_manager
        self._max_workers = max_workers
        self._worker_id = worker_id
        self._server: grpc.Server | None = None
        self._servicer: WorkerServiceServicer | None = None
        self._port: int | None = None

    @property
    def port(self) -> int | None:
        return self._port

    def set_mx_source_id(self, mx_source_id: str) -> None:
        if self._servicer is None:
            raise RuntimeError("Server must be started before setting mx_source_id")
        self._mx_source_id = mx_source_id
        self._servicer.set_mx_source_id(mx_source_id)

    def register_artifact_source(
        self,
        mx_source_id: str,
        artifact_id: str,
        manifest: p2p_pb2.ArtifactManifest,
        artifact_chunk_manager: Any,
    ) -> None:
        if self._servicer is None:
            raise RuntimeError("Server must be started before registering artifacts")
        self._servicer.register_artifact_source(
            mx_source_id,
            artifact_id,
            manifest,
            artifact_chunk_manager,
        )

    def unregister_artifact_source(self, mx_source_id: str, artifact_id: str) -> None:
        if self._servicer is None:
            return
        self._servicer.unregister_artifact_source(mx_source_id, artifact_id)

    def start(self) -> int:
        """Start the gRPC server. Returns the actual bound port."""
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self._max_workers)
        )
        self._servicer = WorkerServiceServicer(
            tensor_protos=self._tensor_protos,
            mx_source_id=self._mx_source_id,
            metadata_endpoint=self._metadata_endpoint,
            agent_name=self._agent_name,
            worker_rank=self._worker_rank,
            accelerator=self._accelerator,
            artifact_manifests=self._artifact_manifests,
            artifact_chunk_manager=self._artifact_chunk_manager,
            worker_id=self._worker_id,
        )
        p2p_pb2_grpc.add_WorkerServiceServicer_to_server(self._servicer, self._server)

        if self._requested_port:
            self._port = self._server.add_insecure_port(f"[::]:{self._requested_port}")
        else:
            self._port = self._server.add_insecure_port("[::]:0")

        self._server.start()
        logger.info(
            f"WorkerGrpcServer started on port {self._port} "
            f"(mx_source_id={self._mx_source_id}, "
            f"{len(self._tensor_protos)} tensors)"
        )
        return self._port

    def stop(self, grace: float = 5.0) -> None:
        if self._server is not None:
            server = self._server
            self._server = None
            self._servicer = None
            self._port = None
            server.stop(grace)
            logger.info("WorkerGrpcServer stopped")

    def drain_tensor_reads(self, timeout: float) -> None:
        if self._servicer is None:
            raise RuntimeError("Server must be started before draining tensor reads")
        self._servicer.drain_tensor_reads(timeout)


class TensorReadLease:
    """A version-validated source manifest held against donor mutation."""

    def __init__(
        self,
        *,
        channel,
        stub,
        mx_source_id: str,
        worker_id: str,
        lease_id: str,
        manifest: p2p_pb2.GetTensorManifestResponse,
        timeout: float,
    ) -> None:
        self._channel = channel
        self._stub = stub
        self._mx_source_id = mx_source_id
        self._worker_id = worker_id
        self._lease_id = lease_id
        self._timeout = timeout
        self._released = False
        self.manifest = manifest

    def release(self) -> None:
        if self._released:
            return
        request = p2p_pb2.ReleaseTensorReadRequest(
            mx_source_id=self._mx_source_id,
            lease_id=self._lease_id,
        )
        if self._worker_id:
            request.worker_id = self._worker_id
        self._stub.ReleaseTensorRead(request, timeout=self._timeout)
        self._released = True

    def close(self) -> None:
        try:
            self.release()
        finally:
            self._channel.close()

    def __enter__(self) -> "TensorReadLease":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def prepare_tensor_read(
    endpoint: str,
    mx_source_id: str,
    *,
    worker_id: str = "",
    timeout: float = 5.0,
    max_retries: int = 0,
    retry_backoff_seconds: float = 0.0,
) -> tuple[TensorReadLease, int]:
    """Prepare a source read lease and its tensor manifest on one channel."""
    if max_retries < 0:
        raise ValueError("tensor read max_retries must not be negative")
    if retry_backoff_seconds < 0:
        raise ValueError("tensor read retry backoff must not be negative")
    request = p2p_pb2.PrepareTensorReadRequest(
        mx_source_id=mx_source_id,
    )
    if worker_id:
        request.worker_id = worker_id
    for attempt in range(max_retries + 1):
        channel = grpc.insecure_channel(endpoint)
        stub = p2p_pb2_grpc.WorkerServiceStub(channel)
        response = None
        try:
            response = stub.PrepareTensorRead(request, timeout=timeout)
            manifest = response.manifest
            if manifest.mx_source_id != mx_source_id:
                raise RuntimeError(
                    f"mx_source_id mismatch: expected {mx_source_id}, "
                    f"got {manifest.mx_source_id}"
                )
            if (
                worker_id
                and manifest.HasField("worker_id")
                and manifest.worker_id != worker_id
            ):
                raise RuntimeError(
                    f"worker_id mismatch: expected {worker_id}, "
                    f"got {manifest.worker_id}"
                )
            lease = TensorReadLease(
                channel=channel,
                stub=stub,
                mx_source_id=mx_source_id,
                worker_id=worker_id,
                lease_id=response.lease_id,
                manifest=manifest,
                timeout=timeout,
            )
            return lease, response.ByteSize()
        except grpc.RpcError as error:
            channel.close()
            if (
                error.code() == grpc.StatusCode.FAILED_PRECONDITION
                and attempt < max_retries
            ):
                time.sleep(retry_backoff_seconds)
                continue
            raise
        except RuntimeError:
            if response is not None and response.lease_id:
                manifest = response.manifest
                release_request = p2p_pb2.ReleaseTensorReadRequest(
                    mx_source_id=manifest.mx_source_id,
                    lease_id=response.lease_id,
                )
                if manifest.HasField("worker_id"):
                    release_request.worker_id = manifest.worker_id
                try:
                    stub.ReleaseTensorRead(release_request, timeout=timeout)
                except grpc.RpcError as error:
                    logger.warning(
                        "Failed to release rejected tensor read lease %s: %s",
                        response.lease_id,
                        error,
                    )
            channel.close()
            if attempt < max_retries:
                time.sleep(retry_backoff_seconds)
                continue
            raise
        except BaseException:
            channel.close()
            raise
    raise RuntimeError("tensor read preparation exhausted without a result")


def fetch_tensor_manifest(
    endpoint: str,
    mx_source_id: str,
    timeout: float = 5.0,
    *,
    worker_id: str = "",
) -> tuple[list[p2p_pb2.TensorDescriptor], int]:
    """Fetch tensor descriptors directly from a source worker's WorkerService.

    Returns a `(tensors, response_bytes)` tuple. `response_bytes` is the
    wire size of the protobuf response (`response.ByteSize()`); callers
    use it to instrument manifest fetch timing.
    """
    channel = grpc.insecure_channel(endpoint)
    stub = p2p_pb2_grpc.WorkerServiceStub(channel)
    request = p2p_pb2.GetTensorManifestRequest(mx_source_id=mx_source_id)
    if worker_id:
        request.worker_id = worker_id
    try:
        response = stub.GetTensorManifest(request, timeout=timeout)
    finally:
        channel.close()
    response_bytes = response.ByteSize()
    if (
        worker_id
        and response.HasField("worker_id")
        and response.worker_id != worker_id
    ):
        raise RuntimeError(
            f"worker_id mismatch: expected {worker_id}, got {response.worker_id}"
        )
    logger.info(
        f"Fetched {len(response.tensors)} tensors from worker at {endpoint} "
        f"({response_bytes} bytes)"
    )
    return list(response.tensors), response_bytes


def fetch_artifact_manifest_header(
    endpoint: str,
    mx_source_id: str,
    artifact_id: str = "",
    timeout: float = 5.0,
) -> tuple[p2p_pb2.GetArtifactManifestHeaderResponse, int]:
    """Fetch a sealed artifact manifest header directly from a source worker."""
    with grpc.insecure_channel(endpoint) as channel:
        stub = p2p_pb2_grpc.WorkerServiceStub(channel)
        request = p2p_pb2.GetArtifactManifestHeaderRequest(
            mx_source_id=mx_source_id,
            artifact_id=artifact_id,
        )
        response = stub.GetArtifactManifestHeader(request, timeout=timeout)
        response_bytes = response.ByteSize()
    logger.info(
        f"Fetched artifact manifest header {response.artifact_id} from worker at "
        f"{endpoint} ({response_bytes} bytes)"
    )
    return response, response_bytes


def fetch_artifact_manifest_chunks(
    endpoint: str,
    mx_source_id: str,
    artifact_id: str,
    start_chunk_index: int = 0,
    max_chunks: int = 0,
    timeout: float = 5.0,
) -> tuple[p2p_pb2.GetArtifactManifestChunksResponse, int]:
    """Fetch one sealed artifact manifest chunk page from a source worker."""
    with grpc.insecure_channel(endpoint) as channel:
        stub = p2p_pb2_grpc.WorkerServiceStub(channel)
        request = p2p_pb2.GetArtifactManifestChunksRequest(
            mx_source_id=mx_source_id,
            artifact_id=artifact_id,
            start_chunk_index=start_chunk_index,
            max_chunks=max_chunks,
        )
        response = stub.GetArtifactManifestChunks(request, timeout=timeout)
        response_bytes = response.ByteSize()
    logger.info(
        f"Fetched artifact manifest chunks {start_chunk_index}:"
        f"{start_chunk_index + len(response.chunks)} for {response.artifact_id} "
        f"from worker at {endpoint} ({response_bytes} bytes)"
    )
    return response, response_bytes


def prepare_artifact_chunk(
    endpoint: str,
    mx_source_id: str,
    artifact_id: str,
    chunk_index: int,
    timeout: float = 5.0,
) -> tuple[p2p_pb2.PrepareArtifactChunkResponse, int]:
    """Prepare one artifact chunk for NIXL transfer on a source worker."""
    with grpc.insecure_channel(endpoint) as channel:
        stub = p2p_pb2_grpc.WorkerServiceStub(channel)
        request = p2p_pb2.PrepareArtifactChunkRequest(
            mx_source_id=mx_source_id,
            artifact_id=artifact_id,
            chunk_index=chunk_index,
        )
        response = stub.PrepareArtifactChunk(request, timeout=timeout)
        response_bytes = response.ByteSize()
    logger.info(
        f"Prepared artifact chunk {chunk_index} for {response.artifact_id} "
        f"from worker at {endpoint} ({response_bytes} bytes)"
    )
    return response, response_bytes


def release_artifact_chunk(
    endpoint: str,
    mx_source_id: str,
    artifact_id: str,
    lease_id: str,
    timeout: float = 5.0,
) -> tuple[p2p_pb2.ReleaseArtifactChunkResponse, int]:
    """Release a prepared artifact chunk lease on a source worker."""
    with grpc.insecure_channel(endpoint) as channel:
        stub = p2p_pb2_grpc.WorkerServiceStub(channel)
        request = p2p_pb2.ReleaseArtifactChunkRequest(
            mx_source_id=mx_source_id,
            artifact_id=artifact_id,
            lease_id=lease_id,
        )
        response = stub.ReleaseArtifactChunk(request, timeout=timeout)
        response_bytes = response.ByteSize()
    logger.info(
        f"Released artifact chunk lease {lease_id} for {response.artifact_id} "
        f"from worker at {endpoint} ({response_bytes} bytes)"
    )
    return response, response_bytes
