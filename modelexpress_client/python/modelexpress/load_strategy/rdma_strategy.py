# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RDMA P2P loading strategy: receive weights from an existing source via NIXL."""

from __future__ import annotations

import logging
import os
import random
import socket
import time

import torch

from ..adapter import EngineAdapter, StrategyFailed
from .base import (
    LoadContext,
    LoadStrategy,
    SourceTransferError,
    _as_load_result,
    register_tensors,
)
from .context import LoadResult
from ..nixl_transfer import is_nixl_available
from ..transfer_safety import check_transfer_allowed
from ..types import ManifestMismatchError, TensorDescriptor
from .. import p2p_pb2

logger = logging.getLogger("modelexpress.strategy_rdma")

MAX_SOURCE_RETRIES = 3


class RdmaStrategy(LoadStrategy):
    """Load weights via RDMA P2P transfer from an existing source.

    Overrides load() entirely since RDMA has a fundamentally different flow:
    prepare target storage -> RDMA receive -> register + publish.
    """

    name = "rdma"
    requires = (EngineAdapter.discover_tensors,)

    def rollback(self, ctx: LoadContext) -> None:
        """Clean up NIXL state from a failed RDMA target attempt."""
        if ctx.nixl_manager is not None:
            try:
                ctx.nixl_manager.shutdown()
            except Exception as e:
                logger.warning(
                    f"[Worker {ctx.global_rank}] Failed to shut down NIXL manager: {e}"
                )
        ctx.tensors = {}
        ctx.nixl_manager = None

    def is_available(self, ctx: LoadContext) -> bool:
        if not super().is_available(ctx):
            return False
        if not is_nixl_available():
            return False

        # Decentralized backends (k8s-service) serve their own
        # metadata; skip the central-server precondition for them.
        # Strict `is True` check so MagicMock's auto-attribute doesn't
        # masquerade as the flag in tests.
        server_addr = os.environ.get("MODEL_EXPRESS_URL") or os.environ.get("MX_SERVER_ADDRESS")
        requires_p2p = getattr(ctx.mx_client, "REQUIRES_P2P_METADATA", False) is True
        if not server_addr and not requires_p2p:
            logger.info(f"[Worker {ctx.global_rank}] No MX server configured, skipping RDMA")
            return False

        allowed, reason = check_transfer_allowed(ctx.model_config)
        if not allowed:
            logger.info(
                f"[Worker {ctx.global_rank}] RDMA transfer disabled: {reason}"
            )
            return False

        return True

    def load(self, result: LoadResult, ctx: LoadContext) -> LoadResult:
        """Load from a READY source or raise StrategyFailed for fallback.

        Source discovery and metadata misses do not mutate the target model.
        Source transfer and manifest failures after target preparation are
        cleaned up by reinitializing the engine model before trying the next
        candidate. If no source works, the chain falls through with a clean
        fresh model so the next strategy can load locally.
        """
        result = _as_load_result(result)
        candidates = self._find_source_instances(ctx)
        if not candidates:
            logger.info(f"[Worker {ctx.global_rank}] No RDMA source available, skipping")
            raise StrategyFailed("No RDMA source available", mutated=False)

        target_prepared = False
        for instance in candidates[:MAX_SOURCE_RETRIES]:
            mx_source_id = instance.mx_source_id
            worker_id = instance.worker_id

            try:
                source_worker = self._fetch_worker_metadata(
                    ctx, mx_source_id, worker_id,
                )
            except Exception as e:
                logger.warning(
                    f"[Worker {ctx.global_rank}] Failed to fetch metadata for worker {worker_id}: {e}. "
                    f"Trying next candidate."
                )
                continue

            if source_worker is None:
                continue
            if self._is_local_worker_metadata(ctx, source_worker):
                logger.info(
                    f"[Worker {ctx.global_rank}] Skipping local stale RDMA "
                    f"source worker {worker_id} ({source_worker.worker_grpc_endpoint or source_worker.metadata_endpoint})"
                )
                continue

            logger.info(
                f"[Worker {ctx.global_rank}] Trying source worker {worker_id} "
                f"({len(source_worker.tensors)} tensors)"
            )

            try:
                return self._load_as_target(
                    result, ctx, source_worker, mx_source_id, worker_id,
                )
            except StrategyFailed as e:
                if self._has_cause(e, ManifestMismatchError):
                    target_prepared = True
                    logger.warning(
                        f"[Worker {ctx.global_rank}] Rejecting RDMA source worker "
                        f"{worker_id}: {e}. Trying next source without rebuilding "
                        "target because no RDMA transfer was started."
                    )
                    continue
                if self._has_cause(e, SourceTransferError):
                    logger.warning(
                        f"[Worker {ctx.global_rank}] Rejecting RDMA source worker "
                        f"{worker_id}: {e}. Aborting RDMA because target may have "
                        "partial RDMA writes."
                    )
                    self.rollback(ctx)
                    raise StrategyFailed(str(e), mutated=True) from e
                if not e.mutated or not self._is_retryable_source_failure(e):
                    raise
                logger.warning(
                    f"[Worker {ctx.global_rank}] Rejecting RDMA source worker "
                    f"{worker_id}: {e}. Cleaning target and trying next source."
                )
                self.rollback(ctx)
                result = ctx.adapter.reinit_for_retry(result)
                continue

        tried = min(len(candidates), MAX_SOURCE_RETRIES)
        logger.warning(
            f"[Worker {ctx.global_rank}] Tried {tried} of {len(candidates)} source workers "
            f"(max retries={MAX_SOURCE_RETRIES}), falling through"
        )
        if target_prepared:
            raise StrategyFailed("No RDMA source had a compatible manifest", mutated=True)
        raise StrategyFailed("No RDMA source succeeded", mutated=False)

    @staticmethod
    def _is_retryable_source_failure(exc: BaseException) -> bool:
        return (
            RdmaStrategy._has_cause(exc, SourceTransferError)
            or RdmaStrategy._has_cause(exc, ManifestMismatchError)
        )

    @staticmethod
    def _has_cause(exc: BaseException, typ: type[BaseException]) -> bool:
        current: BaseException | None = exc
        while current is not None:
            if isinstance(current, typ):
                return True
            current = current.__cause__
        return False

    def _find_source_instances(
        self, ctx: LoadContext,
    ) -> list[p2p_pb2.SourceInstanceRef]:
        """Return all READY source instances (shuffled for load balancing)."""
        try:
            list_resp = ctx.mx_client.list_sources(
                identity=ctx.identity,
                status_filter=p2p_pb2.SOURCE_STATUS_READY,
            )
            if not list_resp.instances:
                logger.debug(f"[Worker {ctx.global_rank}] No ready source instances found")
                return []

            candidates = [
                inst for inst in list_resp.instances
                if inst.worker_rank == ctx.worker_rank
            ]
            random.shuffle(candidates)
            logger.info(
                f"[Worker {ctx.global_rank}] Found {len(candidates)} ready source worker(s)"
            )
            return candidates

        except Exception as e:
            logger.warning(
                f"[Worker {ctx.global_rank}] Error listing sources, falling through: {e}"
            )
            return []

    def _fetch_worker_metadata(
        self,
        ctx: LoadContext,
        mx_source_id: str,
        worker_id: str,
    ) -> p2p_pb2.WorkerMetadata | None:
        """Fetch tensor metadata for one worker."""
        fetch_start = time.perf_counter()
        metadata_resp = ctx.mx_client.get_metadata(
            mx_source_id=mx_source_id,
            worker_id=worker_id,
        )
        if not metadata_resp.found:
            logger.debug(
                f"[Worker {ctx.global_rank}] Metadata not found for worker {worker_id}, skipping"
            )
            return None
        worker = metadata_resp.worker
        if not worker.tensors and not worker.worker_grpc_endpoint:
            logger.debug(
                f"[Worker {ctx.global_rank}] Worker {worker_id} has no tensors "
                f"and no P2P endpoint, skipping"
            )
            return None
        fetch_time = time.perf_counter() - fetch_start
        mode = "P2P (lightweight)" if worker.worker_grpc_endpoint else "centralized"
        tensor_count = len(worker.tensors)
        logger.info(
            f"[Worker {ctx.global_rank}] [TIMING] GetMetadata ({mode}): "
            f"{fetch_time:.3f}s, {tensor_count} tensors"
        )
        return worker

    @staticmethod
    def _is_local_worker_metadata(ctx: LoadContext, worker) -> bool:
        """Return true for metadata that points at this worker's fixed ports.

        During source pod restart the central server can briefly retain READY
        metadata from the previous process. The replacement process has the
        same worker rank and fixed P2P ports, so without this guard it treats
        its own stale endpoint as an RDMA source and falls into target retry.
        """
        grpc_base = int(os.environ.get("MX_WORKER_GRPC_PORT", "6555"))
        metadata_base = int(os.environ.get("MX_METADATA_PORT", "5555"))
        local_ports = {
            grpc_base + ctx.device_id,
            metadata_base + ctx.device_id,
        }
        local_hosts = _local_endpoint_hosts()
        for endpoint in (
            getattr(worker, "worker_grpc_endpoint", ""),
            getattr(worker, "metadata_endpoint", ""),
        ):
            if _endpoint_matches(endpoint, local_hosts, local_ports):
                return True
        return False

    def _load_as_target(
        self,
        result: LoadResult,
        ctx: LoadContext,
        source_worker,
        mx_source_id: str,
        source_worker_id: str,
    ) -> LoadResult:
        """Receive fully-processed weights via RDMA from an existing source."""
        try:
            result = ctx.adapter.prepare_rdma_target(result)
            result = ctx.adapter.before_rdma_receive(result)
            self._receive_from_peer(result, ctx, source_worker, mx_source_id)
            return ctx.adapter.after_rdma_receive(result)
        except StrategyFailed:
            raise
        except ManifestMismatchError as e:
            raise StrategyFailed(
                f"RDMA source manifest incompatible: {e}", mutated=True
            ) from e
        except Exception as e:
            raise StrategyFailed(str(e), mutated=True) from e

    def _receive_from_peer(
        self,
        result: LoadResult,
        ctx: LoadContext,
        source_worker,
        mx_source_id: str,
    ) -> None:
        """Receive fully-processed tensors via RDMA from the detected source."""
        receive_start = time.perf_counter()

        is_p2p = bool(source_worker.worker_grpc_endpoint)
        remote_agent_name_override = None

        if is_p2p:
            from ..metadata.worker_server import fetch_tensor_manifest

            manifest_start = time.perf_counter()
            logger.info(
                f"[Worker {ctx.global_rank}] P2P mode: fetching tensor manifest from "
                f"{source_worker.worker_grpc_endpoint}"
            )
            tensor_protos, manifest_bytes = fetch_tensor_manifest(
                endpoint=source_worker.worker_grpc_endpoint,
                mx_source_id=mx_source_id,
            )
            manifest_time = time.perf_counter() - manifest_start
            source_tensors = [
                _tensor_descriptor_from_proto(t)
                for t in tensor_protos
            ]
            logger.info(
                f"[Worker {ctx.global_rank}] [TIMING] P2P tensor manifest: "
                f"{manifest_time:.3f}s ({len(source_tensors)} tensors, "
                f"{manifest_bytes} bytes)"
            )

        else:
            source_tensors = [
                _tensor_descriptor_from_proto(t)
                for t in source_worker.tensors
            ]

        result = ctx.adapter.prepare_rdma_target_from_manifest(result, source_tensors)
        register_tensors(result, ctx)

        if is_p2p:
            nixl_fetch_start = time.perf_counter()
            ep = source_worker.metadata_endpoint
            host, port_str = ep.rsplit(":", 1)
            ctx.nixl_manager.fetch_remote_and_wait(
                remote_agent_name=source_worker.agent_name,
                ip=host,
                port=int(port_str),
            )
            nixl_fetch_time = time.perf_counter() - nixl_fetch_start
            logger.info(
                f"[Worker {ctx.global_rank}] [TIMING] P2P NIXL metadata fetch: "
                f"{nixl_fetch_time:.3f}s"
            )
            remote_agent_name_override = source_worker.agent_name

        logger.info(
            f"[Worker {ctx.global_rank}] Receiving {len(source_tensors)} tensors from source"
            f"{' (P2P)' if is_p2p else ''}"
        )

        transfer_start = time.perf_counter()
        try:
            bytes_transferred, tensor_count, _ = ctx.nixl_manager.receive_from_source(
                source_metadata=source_worker.nixl_metadata,
                source_tensors=source_tensors,
                timeout_seconds=300.0,
                remote_agent_name=remote_agent_name_override,
            )
        except ManifestMismatchError:
            raise
        except Exception as e:
            raise SourceTransferError(f"RDMA receive failed: {e}") from e
        transfer_time = time.perf_counter() - transfer_start

        bandwidth_gbps = (bytes_transferred * 8) / (transfer_time * 1e9) if transfer_time > 0 else 0
        logger.info(
            f"[Worker {ctx.global_rank}] [TIMING] RDMA transfer complete: "
            f"{tensor_count} tensors, {bytes_transferred / 1e9:.2f} GB, "
            f"{transfer_time:.3f}s, {bandwidth_gbps:.1f} Gbps"
        )

        torch.cuda.synchronize()

        total_time = time.perf_counter() - receive_start
        logger.info(f"[Worker {ctx.global_rank}] [TIMING] Total receive time: {total_time:.2f}s")


def _local_endpoint_hosts() -> set[str]:
    hosts = {
        value
        for value in (
            os.environ.get("MX_WORKER_HOST", ""),
            os.environ.get("POD_IP", ""),
            os.environ.get("NODE_IP", ""),
        )
        if value
    }
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        host = sock.getsockname()[0]
        sock.close()
        if host and not host.startswith("127."):
            hosts.add(host)
    except Exception:
        pass
    return hosts


def _endpoint_matches(endpoint: str, hosts: set[str], ports: set[int]) -> bool:
    if not endpoint:
        return False
    try:
        host, port_str = endpoint.rsplit(":", 1)
        return host in hosts and int(port_str) in ports
    except (TypeError, ValueError):
        return False


def _tensor_descriptor_from_proto(t) -> TensorDescriptor:
    return TensorDescriptor(
        name=t.name,
        addr=t.addr,
        size=t.size,
        device_id=t.device_id,
        dtype=t.dtype,
        shape=list(getattr(t, "shape", [])),
        stride=list(getattr(t, "stride", [])),
        storage_offset=getattr(t, "storage_offset", 0),
        storage_nbytes=getattr(t, "storage_nbytes", 0),
        layout_kind=getattr(t, "layout_kind", ""),
        original_shape=list(getattr(t, "original_shape", [])),
        original_dtype=getattr(t, "original_dtype", ""),
        original_nbytes=getattr(t, "original_nbytes", 0),
        tensor_kind=getattr(t, "tensor_kind", ""),
        owner_module=getattr(t, "owner_module", ""),
        owner_class=getattr(t, "owner_class", ""),
        quant_method=getattr(t, "quant_method", ""),
        runtime_role=getattr(t, "runtime_role", ""),
        replace_policy=getattr(t, "replace_policy", ""),
    )
