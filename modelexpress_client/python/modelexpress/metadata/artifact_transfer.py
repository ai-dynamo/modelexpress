# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL transfer helpers for sealed file-backed artifacts."""

from __future__ import annotations

import logging
import os
import subprocess
import tarfile
import threading
import time
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import uuid4

import grpc
import torch

from .. import p2p_pb2
from ..nixl_transfer import NIXL_DRAM_MEM_TYPE, NixlTransferManager
from .artifact_manifest import (
    _crc32c_hex,
    artifact_source_metadata,
    artifact_manifest_id,
    build_artifact_manifest,
)
from .publisher import PublisherThread
from .publish import _get_worker_host, _publish_metadata_to_server
from .worker_server import (
    WorkerGrpcServer,
    fetch_artifact_manifest_chunks,
    fetch_artifact_manifest_header,
    prepare_artifact_chunk,
    release_artifact_chunk,
)

logger = logging.getLogger("modelexpress.metadata.artifact_transfer")

_DEFAULT_MAX_INFLIGHT_CHUNKS = 4
# Source-side leases are released and their buffers may be reused after this TTL.
# Keep it longer than the target transfer timeout so normal transfers complete
# and explicitly release leases before the source expires them.
_DEFAULT_LEASE_TTL_SECONDS = 300.0
_RESOURCE_EXHAUSTED_PREPARE_ATTEMPTS = 3
_RESOURCE_EXHAUSTED_PREPARE_DELAY_SECONDS = 0.05


@dataclass
class _ArtifactChunkLease:
    artifact_id: str
    chunk: p2p_pb2.ArtifactManifestChunk
    slot_index: int
    expires_at: float


@dataclass
class _ArtifactBufferSlot:
    buffer: torch.Tensor
    registered: object | None


@dataclass
class _TargetBufferSlot:
    buffer: torch.Tensor
    registered: object | None


@dataclass(frozen=True)
class ArtifactBundle:
    """Source-side tar bundle and manifest published by an artifact worker."""

    source_root: Path
    bundle_root: Path
    tar_path: Path
    manifest: p2p_pb2.ArtifactManifest
    artifact_id: str


@dataclass(frozen=True)
class ArtifactCacheRoot:
    """Named source and target directories packaged into one cache artifact."""

    name: str
    source_root: Path
    target_root: Path
    optional: bool = False


@dataclass(frozen=True)
class ArtifactSourceEndpoint:
    """MX-discovered worker endpoint for one published artifact source."""

    mx_source_id: str
    worker_id: str
    worker_rank: int
    worker_grpc_endpoint: str
    artifact_id: str


@dataclass
class PublishedArtifactSource:
    """Running source-side artifact publication."""

    endpoint: ArtifactSourceEndpoint
    grpc_server: WorkerGrpcServer
    heartbeat: PublisherThread
    artifact_chunk_manager: "NixlArtifactChunkManager"

    def stop(self) -> None:
        self.heartbeat.stop()
        self.grpc_server.unregister_artifact_source(
            self.endpoint.mx_source_id,
            self.endpoint.artifact_id,
        )
        self.artifact_chunk_manager.close()


@runtime_checkable
class P2PArtifactTransfer(Protocol):
    """Shared lifecycle for P2P cache artifact transfer.

    Source workers call ``prepare_source`` once to publish a manifest for their
    local cache. Target workers call ``transfer_from_worker`` to stage the
    bundle locally, then ``install`` to unpack it into the runtime cache path.
    """

    name: str
    mx_source_type: int
    roots: tuple[ArtifactCacheRoot, ...]
    bundle_root: Path

    def prepare_source(self) -> ArtifactBundle:
        """Seal local source files and return the publishable artifact."""

    def transfer_from_worker(
        self,
        endpoint: str,
        mx_source_id: str,
        artifact_id: str,
        nixl_manager: NixlTransferManager,
        *,
        timeout: float = 120.0,
        max_inflight_chunks: int = _DEFAULT_MAX_INFLIGHT_CHUNKS,
    ) -> p2p_pb2.GetArtifactManifestHeaderResponse:
        """Transfer from a source worker into target-visible staging."""

    def discover_and_transfer(
        self,
        mx_client,
        identity: p2p_pb2.SourceIdentity,
        nixl_manager: NixlTransferManager,
        *,
        worker_rank: int | None = None,
        node_rank: int | None = None,
        artifact_id: str = "",
        timeout: float = 120.0,
        max_inflight_chunks: int = _DEFAULT_MAX_INFLIGHT_CHUNKS,
    ) -> p2p_pb2.GetArtifactManifestHeaderResponse:
        """Discover an artifact source, then transfer from its worker.

        By default artifact discovery does not rank-match. Pass node_rank for
        node-scoped artifacts or worker_rank for worker-specific artifacts.
        """

    def install(
        self,
        header: p2p_pb2.GetArtifactManifestHeaderResponse,
    ) -> None:
        """Install a transferred artifact into the target root."""


@dataclass(frozen=True)
class TarredP2PArtifactTransfer(P2PArtifactTransfer):
    """Tar-backed transfer used by the cache artifact factories below."""

    name: str
    mx_source_type: int
    roots: tuple[ArtifactCacheRoot, ...]
    bundle_root: Path
    chunk_size: int | None = None
    tar_name: str = "artifact.tar"

    def prepare_source(self) -> ArtifactBundle:
        """Tar the source directories into bundle_root and build their manifest."""
        start = time.perf_counter()
        root_archives = self._root_archives()
        self._validate_archive_names(root_archives)
        bundle_path = self.bundle_root.resolve()
        resolved_sources: dict[str, Path | None] = {}
        for tar_name, root in root_archives:
            source_candidate = root.source_root.resolve()
            if bundle_path == source_candidate or bundle_path.is_relative_to(
                source_candidate
            ):
                raise ValueError("artifact bundle_root must not be inside source_root")
            try:
                resolved_source = root.source_root.resolve(strict=True)
            except FileNotFoundError:
                if not root.optional:
                    raise
                resolved_source = None
            if resolved_source is not None and not resolved_source.is_dir():
                raise ValueError(
                    f"artifact source root is not a directory: {resolved_source}"
                )
            resolved_sources[tar_name] = resolved_source

        source_path = resolved_sources[self.tar_name]
        if source_path is None:
            raise ValueError("primary artifact source root is required")

        bundle_path.mkdir(parents=True, exist_ok=True)
        tar_paths = {bundle_path / tar_name for tar_name, _ in root_archives}
        for child in bundle_path.iterdir():
            if child not in tar_paths:
                raise ValueError(
                    f"artifact bundle_root must be dedicated; found {child}"
                )
        for tar_path in tar_paths:
            tar_path.unlink(missing_ok=True)

        for tar_name, _ in root_archives:
            tar_path = bundle_path / tar_name
            resolved_source = resolved_sources[tar_name]
            if resolved_source is None:
                with tarfile.open(tar_path, "w"):
                    pass
                continue
            _reject_symlinked_source_entries(resolved_source)
            _run_tar(["-cf", str(tar_path), "-C", str(resolved_source), "."])

        manifest = build_artifact_manifest(
            bundle_path,
            chunk_size=self.chunk_size,
            mx_source_type=self.mx_source_type,
        )
        artifact_id = artifact_manifest_id(manifest)
        elapsed = time.perf_counter() - start
        total_size = sum(file.size for file in manifest.files)
        logger.info(
            "[TIMING] Artifact prepare complete: name=%s artifact_id=%s "
            "files=%d chunks=%d size=%.2f MiB elapsed=%.3fs",
            self.name,
            artifact_id,
            len(manifest.files),
            len(manifest.chunks),
            _mib(total_size),
            elapsed,
        )
        return ArtifactBundle(
            source_root=source_path,
            bundle_root=bundle_path,
            tar_path=bundle_path / self.tar_name,
            manifest=manifest,
            artifact_id=artifact_id,
        )

    def transfer_from_worker(
        self,
        endpoint: str,
        mx_source_id: str,
        artifact_id: str,
        nixl_manager: NixlTransferManager,
        *,
        timeout: float = 120.0,
        max_inflight_chunks: int = _DEFAULT_MAX_INFLIGHT_CHUNKS,
    ) -> p2p_pb2.GetArtifactManifestHeaderResponse:
        return transfer_artifact_from_worker(
            endpoint,
            mx_source_id,
            artifact_id,
            nixl_manager,
            timeout=timeout,
            max_inflight_chunks=max_inflight_chunks,
            target_file_paths=self._target_tar_paths(),
        )

    def discover_and_transfer(
        self,
        mx_client,
        identity: p2p_pb2.SourceIdentity,
        nixl_manager: NixlTransferManager,
        *,
        worker_rank: int | None = None,
        node_rank: int | None = None,
        artifact_id: str = "",
        timeout: float = 120.0,
        max_inflight_chunks: int = _DEFAULT_MAX_INFLIGHT_CHUNKS,
    ) -> p2p_pb2.GetArtifactManifestHeaderResponse:
        source = discover_artifact_source(
            mx_client,
            identity,
            worker_rank=worker_rank,
            node_rank=node_rank,
            artifact_id=artifact_id,
        )
        return self.transfer_from_worker(
            source.worker_grpc_endpoint,
            source.mx_source_id,
            source.artifact_id,
            nixl_manager,
            timeout=timeout,
            max_inflight_chunks=max_inflight_chunks,
        )

    def install(
        self,
        header: p2p_pb2.GetArtifactManifestHeaderResponse,
    ) -> None:
        start = time.perf_counter()
        root_archives = self._root_archives()
        self._validate_archive_names(root_archives)
        expected_names = {tar_name for tar_name, _ in root_archives}
        files_by_name = {Path(file.path).name: file for file in header.files}
        if set(files_by_name) != expected_names:
            raise ValueError(
                "artifact bundle files do not match configured cache roots: "
                f"{sorted(files_by_name)} != {sorted(expected_names)}"
            )
        installed_targets = []
        for tar_name, root in root_archives:
            tar_file = Path(files_by_name[tar_name].path).resolve(strict=True)
            output_path = root.target_root.resolve()
            output_path.mkdir(parents=True, exist_ok=True)
            _validate_tar_members(tar_file)
            _run_tar(["-xf", str(tar_file), "-C", str(output_path)])
            installed_targets.append(str(output_path))
        elapsed = time.perf_counter() - start
        logger.info(
            "[TIMING] Artifact install complete: artifact_id=%s targets=%s "
            "size=%.2f MiB elapsed=%.3fs",
            header.artifact_id,
            installed_targets,
            _mib(header.total_size),
            elapsed,
        )

    def _target_tar_paths(self) -> list[Path]:
        root_archives = self._root_archives()
        self._validate_archive_names(root_archives)
        bundle_path = self.bundle_root.resolve()
        for _, root in root_archives:
            target_path = root.target_root.resolve()
            if bundle_path == target_path or bundle_path.is_relative_to(target_path):
                raise ValueError("artifact bundle_root must not be inside target_root")
        bundle_path.mkdir(parents=True, exist_ok=True)
        tar_paths = [bundle_path / tar_name for tar_name, _ in root_archives]
        for child in bundle_path.iterdir():
            if child not in tar_paths:
                raise ValueError(
                    f"artifact bundle_root must be dedicated; found {child}"
                )
        for tar_path in tar_paths:
            if tar_path.is_symlink():
                raise ValueError(
                    f"artifact target bundle path must not be a symlink: {tar_path}"
                )
        return sorted(tar_paths)

    def _root_archives(self) -> tuple[tuple[str, ArtifactCacheRoot], ...]:
        if not self.roots:
            raise ValueError("artifact transfer requires at least one cache root")
        return ((self.tar_name, self.roots[0]),) + tuple(
            (f"{root.name}.tar", root) for root in self.roots[1:]
        )

    @staticmethod
    def _validate_archive_names(
        root_archives: tuple[tuple[str, ArtifactCacheRoot], ...],
    ) -> None:
        names = [tar_name for tar_name, _ in root_archives]
        if any(Path(name).name != name for name in names):
            raise ValueError("artifact tar names must be file names, not paths")
        if len(set(names)) != len(names):
            raise ValueError("artifact tar names must be unique")


def torch_compile_cache_artifact_transfer(
    source_root: str | Path,
    target_root: str | Path,
    bundle_root: str | Path,
    *,
    chunk_size: int | None = None,
) -> P2PArtifactTransfer:
    return _cache_artifact_transfer(
        "torch_compile_cache",
        p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
        source_root,
        target_root,
        bundle_root,
        chunk_size=chunk_size,
    )


def triton_cache_artifact_transfer(
    source_root: str | Path,
    target_root: str | Path,
    bundle_root: str | Path,
    *,
    chunk_size: int | None = None,
) -> P2PArtifactTransfer:
    return _cache_artifact_transfer(
        "triton_cache",
        p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
        source_root,
        target_root,
        bundle_root,
        chunk_size=chunk_size,
    )


def deep_gemm_cache_artifact_transfer(
    source_root: str | Path,
    target_root: str | Path,
    bundle_root: str | Path,
    *,
    chunk_size: int | None = None,
) -> P2PArtifactTransfer:
    return _cache_artifact_transfer(
        "deep_gemm_cache",
        p2p_pb2.MX_SOURCE_TYPE_DEEP_GEMM_CACHE,
        source_root,
        target_root,
        bundle_root,
        chunk_size=chunk_size,
    )


def tilelang_cache_artifact_transfer(
    source_root: str | Path,
    target_root: str | Path,
    bundle_root: str | Path,
    *,
    chunk_size: int | None = None,
) -> P2PArtifactTransfer:
    return _cache_artifact_transfer(
        "tilelang_cache",
        p2p_pb2.MX_SOURCE_TYPE_TILELANG_CACHE,
        source_root,
        target_root,
        bundle_root,
        chunk_size=chunk_size,
    )


def cute_dsl_cache_artifact_transfer(
    source_root: str | Path,
    target_root: str | Path,
    bundle_root: str | Path,
    *,
    chunk_size: int | None = None,
) -> P2PArtifactTransfer:
    return _cache_artifact_transfer(
        "cute_dsl_cache",
        p2p_pb2.MX_SOURCE_TYPE_CUTE_DSL_CACHE,
        source_root,
        target_root,
        bundle_root,
        chunk_size=chunk_size,
    )


def flashinfer_cache_artifact_transfer(
    source_root: str | Path,
    target_root: str | Path,
    bundle_root: str | Path,
    *,
    chunk_size: int | None = None,
    additional_roots: tuple[ArtifactCacheRoot, ...] = (),
) -> P2PArtifactTransfer:
    return _cache_artifact_transfer(
        "flashinfer_cache",
        p2p_pb2.MX_SOURCE_TYPE_FLASHINFER_CACHE,
        source_root,
        target_root,
        bundle_root,
        chunk_size=chunk_size,
        additional_roots=additional_roots,
    )


def publish_artifact_source(
    mx_client,
    transfer: P2PArtifactTransfer,
    bundle: ArtifactBundle,
    identity: p2p_pb2.SourceIdentity,
    nixl_manager: NixlTransferManager,
    worker_id: str,
    *,
    worker_grpc_server: WorkerGrpcServer,
    worker_rank: int = 0,
    node_rank: int = 0,
    max_inflight_chunks: int = _DEFAULT_MAX_INFLIGHT_CHUNKS,
    host: str | None = None,
) -> PublishedArtifactSource:
    """Publish a prepared artifact source to the MX server for discovery."""
    if identity.mx_source_type != transfer.mx_source_type:
        raise ValueError(
            "artifact identity mx_source_type does not match transfer "
            f"source type: {identity.mx_source_type} != {transfer.mx_source_type}"
        )
    if artifact_manifest_id(bundle.manifest) != bundle.artifact_id:
        raise ValueError("artifact bundle id does not match its manifest")
    if bundle.manifest.mx_source_type != transfer.mx_source_type:
        raise ValueError(
            "artifact manifest mx_source_type does not match transfer "
            f"source type: {bundle.manifest.mx_source_type} != {transfer.mx_source_type}"
        )

    worker_host = host or _get_worker_host()
    metadata_endpoint = _nixl_metadata_endpoint(worker_host, nixl_manager)
    if worker_grpc_server.port is None:
        raise RuntimeError("worker gRPC server must be started before artifact publish")
    worker_grpc_endpoint = f"{worker_host}:{worker_grpc_server.port}"
    artifact_chunk_manager = NixlArtifactChunkManager(
        nixl_manager,
        max_buffers=max_inflight_chunks,
    )

    worker = p2p_pb2.WorkerMetadata(
        worker_rank=worker_rank,
        metadata_endpoint=metadata_endpoint,
        agent_name=nixl_manager.agent_name,
        worker_grpc_endpoint=worker_grpc_endpoint,
        artifact_source=artifact_source_metadata(bundle.manifest, node_rank=node_rank),
    )
    try:
        mx_source_id = _publish_metadata_to_server(
            mx_client=mx_client,
            identity=identity,
            worker=worker,
            worker_id=worker_id,
            worker_rank=worker_rank,
        )
    except Exception:
        artifact_chunk_manager.close()
        raise
    try:
        worker_grpc_server.register_artifact_source(
            mx_source_id,
            bundle.artifact_id,
            bundle.manifest,
            artifact_chunk_manager,
        )
    except Exception:
        artifact_chunk_manager.close()
        raise

    heartbeat = PublisherThread(
        mx_client=mx_client,
        mx_source_id=mx_source_id,
        worker_id=worker_id,
        worker_rank=worker_rank,
        nixl_manager=nixl_manager,
    )
    try:
        nixl_manager.refresh_agent_metadata()
        heartbeat.start()
    except Exception:
        worker_grpc_server.unregister_artifact_source(mx_source_id, bundle.artifact_id)
        artifact_chunk_manager.close()
        raise
    logger.debug(
        "Published artifact source to MX server "
        "(mx_source_id=%s, artifact_id=%s, worker_grpc=%s)",
        mx_source_id,
        bundle.artifact_id,
        worker_grpc_endpoint,
    )
    return PublishedArtifactSource(
        endpoint=ArtifactSourceEndpoint(
            mx_source_id=mx_source_id,
            worker_id=worker_id,
            worker_rank=worker_rank,
            worker_grpc_endpoint=worker_grpc_endpoint,
            artifact_id=bundle.artifact_id,
        ),
        grpc_server=worker_grpc_server,
        heartbeat=heartbeat,
        artifact_chunk_manager=artifact_chunk_manager,
    )


def discover_artifact_source(
    mx_client,
    identity: p2p_pb2.SourceIdentity,
    *,
    worker_rank: int | None = None,
    node_rank: int | None = None,
    artifact_id: str = "",
) -> ArtifactSourceEndpoint:
    """Find a ready artifact source through the MX server.

    Artifact sources are not rank-matched by default. Pass node_rank for
    node-scoped artifacts or worker_rank for worker-specific artifacts.
    """
    sources = mx_client.list_sources(
        identity=identity,
        status_filter=p2p_pb2.SOURCE_STATUS_READY,
    )
    for source in sources.instances:
        if worker_rank is not None and source.worker_rank != worker_rank:
            continue
        metadata = mx_client.get_metadata(source.mx_source_id, source.worker_id)
        if not metadata.found:
            continue
        worker = metadata.worker
        if worker.WhichOneof("source_payload") != "artifact_source":
            continue
        if node_rank is not None and worker.artifact_source.node_rank != node_rank:
            continue
        if not worker.worker_grpc_endpoint:
            continue
        published_artifact_id = worker.artifact_source.artifact_id
        if artifact_id and published_artifact_id != artifact_id:
            continue
        return ArtifactSourceEndpoint(
            mx_source_id=metadata.mx_source_id,
            worker_id=metadata.worker_id,
            worker_rank=worker.worker_rank,
            worker_grpc_endpoint=worker.worker_grpc_endpoint,
            artifact_id=published_artifact_id,
        )
    raise LookupError("no ready artifact source found")


def _cache_artifact_transfer(
    name: str,
    mx_source_type: int,
    source_root: str | Path,
    target_root: str | Path,
    bundle_root: str | Path,
    *,
    chunk_size: int | None,
    additional_roots: tuple[ArtifactCacheRoot, ...] = (),
) -> P2PArtifactTransfer:
    return TarredP2PArtifactTransfer(
        name=name,
        mx_source_type=mx_source_type,
        roots=(
            ArtifactCacheRoot(
                name="primary",
                source_root=Path(source_root),
                target_root=Path(target_root),
            ),
            *additional_roots,
        ),
        bundle_root=Path(bundle_root),
        chunk_size=chunk_size,
    )


class NixlArtifactChunkManager:
    """Source-side lease manager for artifact chunks staged in NIXL DRAM."""

    def __init__(
        self,
        nixl_manager: NixlTransferManager,
        max_buffers: int = _DEFAULT_MAX_INFLIGHT_CHUNKS,
        lease_ttl_seconds: float = _DEFAULT_LEASE_TTL_SECONDS,
    ):
        if max_buffers <= 0:
            raise ValueError("artifact chunk manager max_buffers must be positive")
        if lease_ttl_seconds <= 0:
            raise ValueError("artifact chunk manager lease_ttl_seconds must be positive")
        self._nixl_manager = nixl_manager
        self._max_buffers = max_buffers
        self._lease_ttl_seconds = lease_ttl_seconds
        self._slots: list[_ArtifactBufferSlot] = []
        self._free_slots: list[int] = []
        self._leases: dict[str, _ArtifactChunkLease] = {}
        self._lock = threading.Lock()

    def prepare(
        self,
        manifest: p2p_pb2.ArtifactManifest,
        artifact_id: str,
        chunk: p2p_pb2.ArtifactManifestChunk,
    ) -> tuple[str, p2p_pb2.ArtifactChunkTransferDescriptor, bytes]:
        if chunk.file_index >= len(manifest.files):
            raise ValueError(
                f"chunk {chunk.chunk_index} references missing file_index "
                f"{chunk.file_index}"
            )
        if chunk.length == 0:
            raise ValueError(f"chunk {chunk.chunk_index} has zero length")
        if chunk.length > manifest.chunk_size:
            raise ValueError(
                f"chunk {chunk.chunk_index} length {chunk.length} exceeds "
                f"manifest chunk_size {manifest.chunk_size}"
            )

        slot_index = self._reserve_slot(int(manifest.chunk_size))
        slot = self._slots[slot_index]
        view = slot.buffer.narrow(0, 0, int(chunk.length))
        try:
            file = manifest.files[chunk.file_index]
            _read_file_range_into_buffer(
                Path(file.path),
                chunk.file_offset,
                chunk.length,
                view,
            )
        except Exception:
            self._release_slot(slot_index)
            raise

        lease_id = str(uuid4())
        with self._lock:
            self._leases[lease_id] = _ArtifactChunkLease(
                artifact_id=artifact_id,
                chunk=chunk,
                slot_index=slot_index,
                expires_at=time.monotonic() + self._lease_ttl_seconds,
            )
        descriptor = p2p_pb2.ArtifactChunkTransferDescriptor(
            addr=slot.buffer.data_ptr(),
            length=chunk.length,
            device_id=0,
        )
        return lease_id, descriptor, self._nixl_manager.nixl_metadata

    def release(self, lease_id: str) -> tuple[str, p2p_pb2.ArtifactManifestChunk]:
        with self._lock:
            self._expire_leases_locked()
            lease = self._leases.pop(lease_id)
        self._release_slot(lease.slot_index)
        return lease.artifact_id, lease.chunk

    def close(self) -> None:
        with self._lock:
            slots = self._slots
            self._slots = []
            self._free_slots = []
            self._leases = {}
        for slot in slots:
            if slot.registered is None:
                continue
            try:
                self._nixl_manager.deregister_memory(slot.registered)
            except Exception:
                logger.warning(
                    "Failed to deregister source artifact buffer",
                    exc_info=True,
                )

    def _reserve_slot(self, buffer_size: int) -> int:
        with self._lock:
            self._expire_leases_locked()
            if self._slots and buffer_size > self._slots[0].buffer.numel():
                raise ValueError(
                    f"artifact chunk size {buffer_size} exceeds registered buffer "
                    f"size {self._slots[0].buffer.numel()}"
                )
            if not self._slots:
                for slot_index in range(self._max_buffers):
                    buffer = torch.empty(buffer_size, dtype=torch.uint8, device="cpu")
                    registered = self._nixl_manager.register_dram_buffer(buffer)
                    self._slots.append(
                        _ArtifactBufferSlot(buffer=buffer, registered=registered)
                    )
                    self._free_slots.append(slot_index)
            if not self._free_slots:
                raise RuntimeError("all artifact transfer buffers are currently leased")
            return self._free_slots.pop()

    def _release_slot(self, slot_index: int) -> None:
        with self._lock:
            self._free_slots.append(slot_index)

    def _expire_leases_locked(self) -> None:
        now = time.monotonic()
        expired = [
            lease_id
            for lease_id, lease in self._leases.items()
            if lease.expires_at <= now
        ]
        for lease_id in expired:
            lease = self._leases.pop(lease_id)
            self._free_slots.append(lease.slot_index)
            logger.warning(
                "Expired artifact chunk lease %s for chunk %d; released buffer slot %d",
                lease_id,
                lease.chunk.chunk_index,
                lease.slot_index,
            )


def transfer_artifact_from_worker(
    endpoint: str,
    mx_source_id: str,
    artifact_id: str,
    nixl_manager: NixlTransferManager,
    timeout: float = 120.0,
    max_inflight_chunks: int = _DEFAULT_MAX_INFLIGHT_CHUNKS,
    target_file_paths: list[str | Path] | None = None,
) -> p2p_pb2.GetArtifactManifestHeaderResponse:
    """Transfer all files in an artifact from one source worker via NIXL."""
    start = time.perf_counter()
    if max_inflight_chunks <= 0:
        raise ValueError("max_inflight_chunks must be positive")
    header, _ = fetch_artifact_manifest_header(
        endpoint,
        mx_source_id=mx_source_id,
        artifact_id=artifact_id,
    )
    chunks = _fetch_all_chunks(
        endpoint,
        mx_source_id,
        header.artifact_id,
        expected_chunk_count=header.chunk_count,
    )
    _validate_fetched_artifact_manifest(header, chunks, artifact_id)
    target_header = _header_with_target_file_paths(header, target_file_paths)
    target_files = target_header.files
    _prepare_target_files(target_files)
    if not chunks:
        elapsed = time.perf_counter() - start
        logger.info(
            "[TIMING] Artifact transfer complete: artifact_id=%s source=%s "
            "files=%d chunks=0 size=%.2f MiB elapsed=%.3fs throughput=%.2f Gbps",
            header.artifact_id,
            endpoint,
            header.file_count,
            _mib(header.total_size),
            elapsed,
            _gbps(header.total_size, elapsed),
        )
        return target_header

    target_slots = _register_target_buffers(
        nixl_manager,
        buffer_size=max(int(chunk.length) for chunk in chunks),
        buffer_count=min(max_inflight_chunks, len(chunks)),
    )
    remote_agent_name = ""
    try:
        next_chunk = 0
        while next_chunk < len(chunks):
            batch = chunks[next_chunk : next_chunk + max_inflight_chunks]
            prepared = _prepare_chunk_batch(
                endpoint,
                mx_source_id,
                header.artifact_id,
                batch,
            )
            try:
                if not remote_agent_name:
                    remote_agent_name = nixl_manager.add_remote_agent(
                        prepared[-1].source_metadata
                    )
                _transfer_prepared_batch(
                    nixl_manager,
                    remote_agent_name,
                    target_files,
                    prepared,
                    target_slots,
                    timeout,
                )
            finally:
                _release_prepared_chunks(
                    endpoint,
                    mx_source_id,
                    header.artifact_id,
                    prepared,
                )
            next_chunk += len(prepared)
    except Exception:
        _cleanup_target_files(target_files)
        raise
    finally:
        _close_target_buffers(nixl_manager, target_slots)

    elapsed = time.perf_counter() - start
    logger.info(
        "[TIMING] Artifact transfer complete: artifact_id=%s source=%s "
        "files=%d chunks=%d size=%.2f MiB elapsed=%.3fs throughput=%.2f Gbps",
        header.artifact_id,
        endpoint,
        header.file_count,
        header.chunk_count,
        _mib(header.total_size),
        elapsed,
        _gbps(header.total_size, elapsed),
    )
    return target_header


def _transfer_prepared_batch(
    nixl_manager: NixlTransferManager,
    remote_agent_name: str,
    files,
    prepared: list[p2p_pb2.PrepareArtifactChunkResponse],
    target_slots: list[_TargetBufferSlot],
    timeout: float,
) -> None:
    first_error: Exception | None = None
    with futures.ThreadPoolExecutor(max_workers=len(prepared)) as executor:
        pending = [
            executor.submit(
                _transfer_prepared_chunk,
                nixl_manager,
                remote_agent_name,
                files,
                response,
                target_slots[index],
                timeout,
            )
            for index, response in enumerate(prepared)
        ]
        for future in futures.as_completed(pending):
            try:
                future.result()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
    if first_error is not None:
        raise first_error


def _transfer_prepared_chunk(
    nixl_manager: NixlTransferManager,
    remote_agent_name: str,
    files,
    response: p2p_pb2.PrepareArtifactChunkResponse,
    target_slot: _TargetBufferSlot,
    timeout: float,
) -> None:
    nixl_manager.receive_dram_into_buffer(
        remote_agent_name=remote_agent_name,
        remote_addr=response.source.addr,
        local_buffer=target_slot.buffer,
        size=response.source.length,
        remote_device_id=response.source.device_id,
        remote_mem_type=NIXL_DRAM_MEM_TYPE,
        timeout_seconds=timeout,
    )
    tensor = target_slot.buffer.narrow(0, 0, int(response.source.length))
    _verify_and_write_chunk(files, response.chunk, tensor)


def _register_target_buffers(
    nixl_manager: NixlTransferManager,
    buffer_size: int,
    buffer_count: int,
) -> list[_TargetBufferSlot]:
    slots: list[_TargetBufferSlot] = []
    try:
        for _ in range(buffer_count):
            buffer = torch.empty(buffer_size, dtype=torch.uint8, device="cpu")
            registered = nixl_manager.register_dram_buffer(buffer)
            slots.append(_TargetBufferSlot(buffer=buffer, registered=registered))
    except Exception:
        _close_target_buffers(nixl_manager, slots)
        raise
    return slots


def _close_target_buffers(
    nixl_manager: NixlTransferManager,
    slots: list[_TargetBufferSlot],
) -> None:
    for slot in slots:
        if slot.registered is None:
            continue
        try:
            nixl_manager.deregister_memory(slot.registered)
        except Exception:
            logger.warning("Failed to deregister target artifact buffer", exc_info=True)


def _reject_symlinked_source_entries(source_root: Path) -> None:
    for path in source_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"tarred artifact source contains symlink: {path}")


def _validate_tar_members(tar_path: Path) -> None:
    with tarfile.open(tar_path, mode="r:") as archive:
        for member in archive:
            name = member.name
            while name.startswith("./"):
                name = name[2:]
            parts = [part for part in name.split("/") if part]
            if name.startswith("/") or ".." in parts:
                raise ValueError(f"unsafe tar member path: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"unsupported tar member type: {member.name}")


def _run_tar(args: list[str]) -> None:
    command = ["tar", *args]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("tar executable not found") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"tar command failed: {stderr}") from exc


def _nixl_metadata_endpoint(host: str, nixl_manager: NixlTransferManager) -> str:
    listen_port = getattr(nixl_manager, "_listen_port", None)
    if listen_port is None:
        return ""
    return f"{host}:{listen_port}"


def _cleanup_target_files(files) -> None:
    for file in files:
        try:
            Path(file.path).unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to remove partial artifact file %s", file.path)


def _header_with_target_file_paths(
    header: p2p_pb2.GetArtifactManifestHeaderResponse,
    target_file_paths: list[str | Path] | None,
) -> p2p_pb2.GetArtifactManifestHeaderResponse:
    if target_file_paths is None:
        return header
    if len(target_file_paths) != len(header.files):
        raise ValueError(
            "target_file_paths length must match artifact file count: "
            f"{len(target_file_paths)} != {len(header.files)}"
        )

    target_header = p2p_pb2.GetArtifactManifestHeaderResponse()
    target_header.CopyFrom(header)
    del target_header.files[:]
    for source_file, target_path in zip(header.files, target_file_paths, strict=True):
        target_file = p2p_pb2.ArtifactManifestFile()
        target_file.CopyFrom(source_file)
        target_file.path = Path(target_path).absolute().as_posix()
        target_header.files.append(target_file)
    return target_header


def _prepare_chunk_batch(
    endpoint: str,
    mx_source_id: str,
    artifact_id: str,
    chunks: list[p2p_pb2.ArtifactManifestChunk],
) -> list[p2p_pb2.PrepareArtifactChunkResponse]:
    prepared: list[p2p_pb2.PrepareArtifactChunkResponse] = []
    resource_exhausted_attempts = 0
    while len(prepared) < len(chunks):
        chunk = chunks[len(prepared)]
        try:
            response, _ = prepare_artifact_chunk(
                endpoint,
                mx_source_id=mx_source_id,
                artifact_id=artifact_id,
                chunk_index=chunk.chunk_index,
            )
            prepared.append(response)
            resource_exhausted_attempts = 0
        except Exception as exc:
            if _is_resource_exhausted(exc):
                if prepared:
                    return prepared
                resource_exhausted_attempts += 1
                if resource_exhausted_attempts >= _RESOURCE_EXHAUSTED_PREPARE_ATTEMPTS:
                    raise
                time.sleep(_RESOURCE_EXHAUSTED_PREPARE_DELAY_SECONDS)
                continue
            _release_prepared_chunks(endpoint, mx_source_id, artifact_id, prepared)
            raise
    return prepared


def _is_resource_exhausted(exc: Exception) -> bool:
    return (
        isinstance(exc, grpc.RpcError)
        and exc.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
    )


def _release_prepared_chunks(
    endpoint: str,
    mx_source_id: str,
    artifact_id: str,
    prepared: list[p2p_pb2.PrepareArtifactChunkResponse],
) -> None:
    for response in prepared:
        try:
            release_artifact_chunk(
                endpoint,
                mx_source_id=mx_source_id,
                artifact_id=artifact_id,
                lease_id=response.lease_id,
            )
        except Exception:
            logger.warning(
                "Failed to release prepared artifact chunk lease %s",
                response.lease_id,
                exc_info=True,
            )


def _prepare_target_files(files) -> None:
    for file in files:
        path = Path(file.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise ValueError(f"artifact target file must not be a symlink: {path}")
        with path.open("wb") as output:
            output.truncate(file.size)


def _fetch_all_chunks(
    endpoint: str,
    mx_source_id: str,
    artifact_id: str,
    *,
    expected_chunk_count: int,
) -> list[p2p_pb2.ArtifactManifestChunk]:
    if expected_chunk_count == 0:
        return []
    chunks: list[p2p_pb2.ArtifactManifestChunk] = []
    start = 0
    while True:
        response, _ = fetch_artifact_manifest_chunks(
            endpoint,
            mx_source_id=mx_source_id,
            artifact_id=artifact_id,
            start_chunk_index=start,
        )
        chunks.extend(response.chunks)
        if len(chunks) > expected_chunk_count:
            raise RuntimeError("artifact chunk pages exceeded chunk_count")
        if not response.next_page_token:
            return chunks
        next_start = int(response.next_page_token)
        if next_start <= start:
            raise RuntimeError("artifact chunk page token did not advance")
        start = next_start


def _validate_fetched_artifact_manifest(
    header: p2p_pb2.GetArtifactManifestHeaderResponse,
    chunks: list[p2p_pb2.ArtifactManifestChunk],
    expected_artifact_id: str,
) -> None:
    if expected_artifact_id and header.artifact_id != expected_artifact_id:
        raise RuntimeError("artifact header id mismatch")
    if header.file_count != len(header.files) or header.chunk_count != len(chunks):
        raise RuntimeError("artifact manifest count mismatch")

    files = list(header.files)
    coverage = [0 for _ in files]
    for chunk_index, chunk in enumerate(chunks):
        if (
            chunk.chunk_index != chunk_index
            or chunk.file_index >= len(files)
            or chunk.length == 0
            or chunk.length > header.chunk_size
        ):
            raise RuntimeError("invalid artifact chunk table")
        file = files[chunk.file_index]
        if chunk.file_offset != coverage[chunk.file_index]:
            raise RuntimeError("artifact chunk coverage gap or overlap")
        coverage[chunk.file_index] += chunk.length
        if coverage[chunk.file_index] > file.size:
            raise RuntimeError("artifact chunk exceeds file size")

    if any(file.file_index != index for index, file in enumerate(files)):
        raise RuntimeError("invalid artifact file table")
    if any(covered != file.size for covered, file in zip(coverage, files, strict=True)):
        raise RuntimeError("artifact file coverage mismatch")

    manifest = p2p_pb2.ArtifactManifest(
        manifest_version=header.manifest_version,
        mx_source_type=header.mx_source_type,
        chunk_size=header.chunk_size,
        files=header.files,
        chunks=chunks,
    )
    computed_artifact_id = artifact_manifest_id(manifest)
    if computed_artifact_id != header.artifact_id:
        raise RuntimeError("artifact manifest id mismatch")


def _mib(size_bytes: int) -> float:
    return size_bytes / (1024 * 1024)


def _gbps(size_bytes: int, elapsed_secs: float) -> float:
    if elapsed_secs <= 0:
        return 0.0
    return size_bytes * 8 / elapsed_secs / 1_000_000_000


def _read_file_range_into_buffer(
    path: Path,
    offset: int,
    length: int,
    buffer: torch.Tensor,
) -> None:
    with path.open("rb") as file:
        file.seek(offset)
        view = memoryview(buffer.numpy()).cast("B")
        read = file.readinto(view)
    if read != length:
        raise OSError(f"short read from {path}: expected {length} bytes, got {read}")


def _verify_and_write_chunk(
    files,
    chunk: p2p_pb2.ArtifactManifestChunk,
    tensor: torch.Tensor,
) -> None:
    if chunk.file_index >= len(files):
        raise ValueError(
            f"chunk {chunk.chunk_index} references missing file_index "
            f"{chunk.file_index}"
        )
    array = tensor.numpy()
    checksum = _crc32c_hex(array)
    if checksum != chunk.checksum:
        raise RuntimeError(
            f"artifact chunk {chunk.chunk_index} crc32c mismatch: "
            f"expected {chunk.checksum}, got {checksum}"
        )

    data = memoryview(array).cast("B")
    path = Path(files[chunk.file_index].path)
    # Transfer threads open independent file descriptors and write non-overlapping
    # manifest chunks, so they do not share a mutable file offset.
    with path.open("r+b") as output:
        output.seek(chunk.file_offset)
        output.write(data)
