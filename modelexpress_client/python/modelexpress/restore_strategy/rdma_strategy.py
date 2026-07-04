# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-source GMS snapshot restore through NIXL RDMA."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from importlib.metadata import version as pkg_version

from .. import envs, p2p_pb2
from ..client import MxClientBase
from ..metadata.client_factory import create_metadata_client
from ..metadata.payload import accelerators_compatible, worker_tensor_descriptors
from ..nixl_transfer import NixlTransferManager, is_nixl_available
from ..types import TensorDescriptor
from .base import RestoreStrategy, RestoreStrategyFailed
from .context import GmsRestoreContext

logger = logging.getLogger("modelexpress.restore_strategy_rdma")

_TRANSFER_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class _DiscoveredSource:
    mx_source_id: str
    worker_id: str
    worker: p2p_pb2.WorkerMetadata
    regions: list[TensorDescriptor]


class RdmaRestoreStrategy(RestoreStrategy):
    """Restore a GMS snapshot from a compatible READY source over RDMA."""

    name = "rdma"

    def __init__(self) -> None:
        self._manager: NixlTransferManager | None = None

    def is_available(self, ctx: GmsRestoreContext) -> bool:
        if _metadata_endpoint(ctx) is None:
            return False
        try:
            if not is_nixl_available():
                return False
            return bool(ctx.accelerator_backend.supports_rdma_p2p())
        except Exception:
            logger.warning("RDMA capability check failed", exc_info=True)
            return False

    def restore(self, ctx: GmsRestoreContext) -> dict[str, object]:
        endpoint = _metadata_endpoint(ctx)
        if endpoint is None:
            raise RestoreStrategyFailed(
                "no ModelExpress metadata endpoint configured",
                mutated=False,
            )

        try:
            client = create_metadata_client(
                worker_rank=ctx.device,
                server_url=endpoint,
            )
        except Exception as exc:
            logger.warning("Failed to create ModelExpress metadata client: %s", exc)
            raise RestoreStrategyFailed(
                "no compatible live RDMA source",
                mutated=False,
            ) from exc

        try:
            source = self._find_compatible_source(ctx, client)
        finally:
            try:
                client.close()
            except Exception:
                logger.warning("Failed to close metadata client", exc_info=True)

        if source is None:
            raise RestoreStrategyFailed(
                "no compatible live RDMA source",
                mutated=False,
            )

        try:
            return self._receive_from_source(ctx, source)
        except Exception as exc:
            raise RestoreStrategyFailed(str(exc), mutated=True) from exc

    def rollback(self, ctx: GmsRestoreContext) -> None:
        if self._manager is None:
            return
        try:
            self._shutdown_manager(self._manager)
        except Exception:
            logger.warning("Failed to shut down NIXL restore manager", exc_info=True)

    def _shutdown_manager(self, manager: NixlTransferManager) -> None:
        manager.shutdown()
        self._manager = None

    def _find_compatible_source(
        self,
        ctx: GmsRestoreContext,
        client: MxClientBase,
    ) -> _DiscoveredSource | None:
        identity = _build_snapshot_identity(ctx)
        try:
            response = client.list_sources(
                identity=identity,
                status_filter=p2p_pb2.SOURCE_STATUS_READY,
            )
        except Exception as exc:
            logger.warning("Failed to list live GMS snapshot sources: %s", exc)
            return None

        target_accelerator = ctx.accelerator_backend.name
        candidates = [
            instance
            for instance in response.instances
            if int(instance.worker_rank) == ctx.device
            and accelerators_compatible(target_accelerator, instance.accelerator)
        ]
        for instance in candidates:
            try:
                metadata = client.get_metadata(
                    mx_source_id=instance.mx_source_id,
                    worker_id=instance.worker_id,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to fetch GMS source metadata for worker %s: %s",
                    instance.worker_id,
                    exc,
                )
                continue

            if (
                not metadata.found
                or metadata.mx_source_id != instance.mx_source_id
                or metadata.worker_id != instance.worker_id
                or int(metadata.worker.worker_rank) != ctx.device
                or metadata.worker.status != p2p_pb2.SOURCE_STATUS_READY
            ):
                continue
            worker = metadata.worker
            if not accelerators_compatible(
                target_accelerator,
                worker.accelerator,
            ):
                logger.info(
                    "Skipping incompatible GMS source worker %s: "
                    "accelerator source=%r target=%r",
                    instance.worker_id,
                    worker.accelerator,
                    target_accelerator,
                )
                continue
            try:
                regions = _source_regions(
                    worker,
                    instance.mx_source_id,
                    instance.worker_id,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to fetch GMS source regions for worker %s: %s",
                    instance.worker_id,
                    exc,
                )
                continue

            incompatibility = _source_incompatibility(ctx, worker, regions)
            if incompatibility is not None:
                logger.info(
                    "Skipping incompatible GMS source worker %s: %s",
                    instance.worker_id,
                    incompatibility,
                )
                continue

            return _DiscoveredSource(
                mx_source_id=instance.mx_source_id,
                worker_id=instance.worker_id,
                worker=worker,
                regions=regions,
            )

        return None

    def _receive_from_source(
        self,
        ctx: GmsRestoreContext,
        source: _DiscoveredSource,
    ) -> dict[str, object]:
        started = time.perf_counter()
        manager = NixlTransferManager(
            agent_name=f"mx-gms-restore-{ctx.device}-{uuid.uuid4().hex[:8]}",
            device_id=ctx.device,
            accelerator_backend=ctx.accelerator_backend,
        )
        self._manager = manager
        try:
            manager.initialize()
            target_regions = {
                allocation_id: TensorDescriptor(
                    name=allocation_id,
                    addr=int(target.va),
                    size=int(target.byte_count),
                    device_id=int(target.device),
                    dtype="uint8",
                )
                for allocation_id, target in ctx.targets.items()
            }
            manager.register_device_regions(target_regions)

            remote_agent_name = None
            source_metadata = source.worker.nixl_metadata
            if source.worker.worker_grpc_endpoint:
                host, port = _split_metadata_endpoint(
                    source.worker.metadata_endpoint
                )
                manager.fetch_remote_and_wait(
                    remote_agent_name=source.worker.agent_name,
                    ip=host,
                    port=port,
                )
                remote_agent_name = source.worker.agent_name

            bytes_transferred, region_count, _ = manager.receive_from_source(
                source_metadata=source_metadata,
                source_tensors=source.regions,
                timeout_seconds=_TRANSFER_TIMEOUT_SECONDS,
                remote_agent_name=remote_agent_name,
            )

            expected_bytes = sum(
                int(target.byte_count) for target in ctx.targets.values()
            )
            if bytes_transferred != expected_bytes or region_count != len(ctx.targets):
                raise RuntimeError(
                    "incomplete GMS RDMA restore: "
                    f"bytes={bytes_transferred}/{expected_bytes} "
                    f"regions={region_count}/{len(ctx.targets)}"
                )

            return {
                "total_bytes": bytes_transferred,
                "elapsed_s": time.perf_counter() - started,
                "selected_strategy": self.name,
                "source_count": len(ctx.sources),
                "file_count": len({source.file_path for source in ctx.sources}),
                "max_inflight_batches": 1,
            }
        finally:
            self._shutdown_manager(manager)


def _metadata_endpoint(ctx: GmsRestoreContext) -> str | None:
    endpoint = (
        envs.MODEL_EXPRESS_URL
        or envs.MX_SERVER_ADDRESS
        or ctx.backend_config.get("mx_p2p_metadata_endpoint")
    )
    if endpoint is None:
        return None
    value = str(endpoint).strip()
    return value or None


def _build_snapshot_identity(ctx: GmsRestoreContext) -> p2p_pb2.SourceIdentity:
    try:
        mx_version = pkg_version("modelexpress")
    except Exception:
        mx_version = "0.0.0"

    return p2p_pb2.SourceIdentity(
        mx_version=mx_version,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_GMS_WEIGHT_SNAPSHOT,
        model_name=_checkpoint_name(ctx),
        extra_parameters={
            "gms_snapshot_extents": _checkpoint_extent_digest(ctx),
        },
    )


def _checkpoint_name(ctx: GmsRestoreContext) -> str:
    parents = [
        os.path.dirname(os.path.normpath(str(source.file_path))) or "."
        for source in ctx.sources
    ]
    if not parents:
        return "."
    try:
        return os.path.commonpath(parents)
    except ValueError:
        return parents[0]


def _checkpoint_extent_digest(ctx: GmsRestoreContext) -> str:
    extents = [
        {
            "allocation_id": source.allocation_id,
            "file_path": os.path.normpath(str(source.file_path)),
            "file_offset": int(source.file_offset),
            "byte_count": int(source.byte_count),
        }
        for source in sorted(
            ctx.sources,
            key=lambda item: (
                item.allocation_id,
                str(item.file_path),
                int(item.file_offset),
            ),
        )
    ]
    payload = json.dumps(extents, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _source_regions(
    worker: p2p_pb2.WorkerMetadata,
    mx_source_id: str,
    worker_id: str,
) -> list[TensorDescriptor]:
    protos = worker_tensor_descriptors(worker)
    if not protos and worker.worker_grpc_endpoint:
        from ..metadata.worker_server import fetch_tensor_manifest

        protos, _ = fetch_tensor_manifest(
            endpoint=worker.worker_grpc_endpoint,
            mx_source_id=mx_source_id,
            worker_id=worker_id,
        )

    return [
        TensorDescriptor(
            name=region.name,
            addr=int(region.addr),
            size=int(region.size),
            device_id=int(region.device_id),
            dtype=region.dtype,
        )
        for region in protos
    ]


def _source_incompatibility(
    ctx: GmsRestoreContext,
    worker: p2p_pb2.WorkerMetadata,
    regions: list[TensorDescriptor],
) -> str | None:
    if not accelerators_compatible(
        ctx.accelerator_backend.name,
        worker.accelerator,
    ):
        return (
            "accelerator mismatch: "
            f"source={worker.accelerator!r} "
            f"target={ctx.accelerator_backend.name!r}"
        )

    if worker.worker_grpc_endpoint:
        if not worker.metadata_endpoint or not worker.agent_name:
            return "RDMA worker is missing NIXL endpoint metadata"
    elif not worker.nixl_metadata:
        return "centralized worker is missing NIXL metadata"

    by_name: dict[str, TensorDescriptor] = {}
    for region in regions:
        if region.name in by_name:
            return f"duplicate allocation_id {region.name!r}"
        by_name[region.name] = region

    expected = set(ctx.targets)
    actual = set(by_name)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        return f"allocation set mismatch: missing={missing} extra={extra}"

    for allocation_id, target in ctx.targets.items():
        region = by_name[allocation_id]
        if region.addr <= 0:
            return f"source allocation {allocation_id!r} has an invalid address"
        if region.size != int(target.byte_count):
            return (
                f"allocation {allocation_id!r} size mismatch: "
                f"source={region.size} target={target.byte_count}"
            )
        if region.dtype != "uint8":
            return (
                f"allocation {allocation_id!r} dtype mismatch: "
                f"source={region.dtype!r} target='uint8'"
            )

    return None


def _split_metadata_endpoint(endpoint: str) -> tuple[str, int]:
    try:
        host, port = endpoint.rsplit(":", 1)
        return host, int(port)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid NIXL metadata endpoint {endpoint!r}") from exc
