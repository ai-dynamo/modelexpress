# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RDMA P2P loading strategy: receive weights from an existing source via NIXL."""

from __future__ import annotations

import copy
import logging
import os
import random
import time

import torch
import torch.nn as nn

from .base import LoadContext, LoadStrategy, SourceTransferError, register_tensors, publish_metadata
from ..nixl_transfer import RegionLayoutMismatchError, is_nixl_available
from ..tensor_utils import capture_tensor_attrs
from ..transfer_safety import check_transfer_allowed
from ..types import ManifestMismatchError, TensorDescriptor
from .. import p2p_pb2

logger = logging.getLogger("modelexpress.strategy_rdma")

MAX_SOURCE_RETRIES = 3


class RdmaStrategy(LoadStrategy):
    """Load weights via RDMA P2P transfer from an existing source.

    Overrides load() entirely since RDMA has a fundamentally different flow:
    dummy weights -> process -> RDMA receive -> register + publish.
    """

    name = "rdma"

    def rollback(self, ctx: LoadContext) -> bool:
        """Clean up NIXL state from a failed RDMA target attempt.

        Returns True if _load_as_target() ran (and thus
        process_weights_after_loading mutated the model).  Detected by
        checking whether register_tensors() populated ctx during the
        attempt.
        """
        if ctx.tensors or ctx.nixl_manager is not None:
            ctx.tensors = {}
            ctx.nixl_manager = None
            return True
        return False

    def is_available(self, ctx: LoadContext) -> bool:
        if not is_nixl_available():
            return False

        server_addr = os.environ.get("MODEL_EXPRESS_URL") or os.environ.get("MX_SERVER_ADDRESS")
        if not server_addr:
            logger.info(f"[Worker {ctx.global_rank}] No MX server configured, skipping RDMA")
            return False

        allowed, reason = check_transfer_allowed(ctx.model_config)
        if not allowed:
            logger.info(
                f"[Worker {ctx.global_rank}] RDMA transfer disabled: {reason}"
            )
            return False

        return True

    def load(self, model: nn.Module, ctx: LoadContext) -> bool:
        candidates = self._find_source_instances(ctx)
        if not candidates:
            logger.info(f"[Worker {ctx.global_rank}] No RDMA source available, skipping")
            return False

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

            logger.info(
                f"[Worker {ctx.global_rank}] Trying source worker {worker_id} "
                f"({len(source_worker.tensors)} tensors)"
            )

            try:
                self._load_as_target(model, ctx, source_worker, mx_source_id, worker_id)
                return True
            except SourceTransferError as e:
                logger.warning(
                    f"[Worker {ctx.global_rank}] Source transfer failed for worker {worker_id}: {e}. "
                    f"Trying next candidate."
                )
            except ManifestMismatchError as e:
                logger.warning(
                    f"[Worker {ctx.global_rank}] Manifest mismatch with worker {worker_id}: {e}. "
                    f"Trying next candidate."
                )

        tried = min(len(candidates), MAX_SOURCE_RETRIES)
        logger.warning(
            f"[Worker {ctx.global_rank}] Tried {tried} of {len(candidates)} source workers "
            f"(max retries={MAX_SOURCE_RETRIES}), falling through"
        )
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
                if inst.worker_rank == ctx.global_rank
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

    def _load_as_target(
        self,
        model: nn.Module,
        ctx: LoadContext,
        source_worker,
        mx_source_id: str,
        source_worker_id: str,
    ) -> None:
        """Receive fully-processed weights via RDMA from an existing source."""
        from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        dummy_config = copy.copy(ctx.load_config)
        try:
            dummy_config.load_format = "dummy"
        except AttributeError:
            object.__setattr__(dummy_config, "load_format", "dummy")
        dummy_loader = DummyModelLoader(dummy_config)
        dummy_loader.load_weights(model, ctx.model_config)

        with capture_tensor_attrs():
            process_weights_after_loading(model, ctx.model_config, ctx.target_device)

        self._receive_from_peer(model, ctx, source_worker, mx_source_id)

        publish_metadata(ctx)

    def _receive_from_peer(
        self,
        model: nn.Module,
        ctx: LoadContext,
        source_worker,
        mx_source_id: str,
    ) -> None:
        """Receive fully-processed tensors via RDMA from the detected source."""
        receive_start = time.perf_counter()
        register_tensors(model, ctx)

        is_p2p = bool(source_worker.worker_grpc_endpoint)
        remote_agent_name_override = None

        if is_p2p:
            from ..worker_server import fetch_tensor_manifest

            manifest_start = time.perf_counter()
            logger.info(
                f"[Worker {ctx.global_rank}] P2P mode: fetching tensor manifest from "
                f"{source_worker.worker_grpc_endpoint}"
            )
            tensor_protos = fetch_tensor_manifest(
                endpoint=source_worker.worker_grpc_endpoint,
                mx_source_id=mx_source_id,
            )
            manifest_time = time.perf_counter() - manifest_start
            source_tensors = [
                TensorDescriptor(
                    name=t.name, addr=t.addr, size=t.size,
                    device_id=t.device_id, dtype=t.dtype,
                )
                for t in tensor_protos
            ]
            logger.info(
                f"[Worker {ctx.global_rank}] [TIMING] P2P tensor manifest: "
                f"{manifest_time:.3f}s ({len(source_tensors)} tensors)"
            )

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
        else:
            source_tensors = [
                TensorDescriptor(
                    name=t.name, addr=t.addr, size=t.size,
                    device_id=t.device_id, dtype=t.dtype,
                )
                for t in source_worker.tensors
            ]

        logger.info(
            f"[Worker {ctx.global_rank}] Receiving {len(source_tensors)} tensors from source"
            f"{' (P2P)' if is_p2p else ''}"
        )

        coalesce = os.environ.get("MX_CONTIGUOUS_REG", "0") == "1"

        transfer_start = time.perf_counter()
        try:
            bytes_transferred, tensor_count, _ = ctx.nixl_manager.receive_from_source(
                source_metadata=source_worker.nixl_metadata,
                source_tensors=source_tensors,
                timeout_seconds=300.0,
                coalesce_transfers=coalesce,
                remote_agent_name=remote_agent_name_override,
            )
        except RegionLayoutMismatchError as e:
            # Contiguous-region layouts differ between source and this worker
            # (PyTorch CUDA allocator non-determinism). The source itself is
            # healthy; raise ManifestMismatchError so the caller just tries
            # the next candidate without marking the source STALE.
            raise ManifestMismatchError(
                f"region layout mismatch with source: {e}"
            ) from e
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
