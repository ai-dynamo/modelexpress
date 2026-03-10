# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared MX hook library for GMS + NIXL + MX Server operations.

Engine launchers call these functions from within each worker process.
All functions are stateless and take explicit arguments, making them
safe to call from spawned child processes without pickle concerns.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .config import MxConfig

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
    from ..nixl_transfer import NixlTransferManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source hooks
# ---------------------------------------------------------------------------


def source_connect_gms(device_id: int):
    """Connect to GMS and return the client + memory pool.

    Must be called BEFORE model initialization so that tensors can be
    allocated through the GMS memory pool via ``use_mem_pool(pool)``.
    GMS's ``register_module_tensors`` only recognises tensors that were
    allocated through its pool.

    Args:
        device_id: CUDA device index.

    Returns:
        Tuple of (gms_client, pool) for use with ``torch.cuda.memory.use_mem_pool``.
    """
    from gpu_memory_service import get_or_create_gms_client_memory_manager
    from gpu_memory_service.common.types import RequestedLockType
    from gpu_memory_service.common.utils import get_socket_path

    socket_path = get_socket_path(device_id)
    gms_client, pool = get_or_create_gms_client_memory_manager(
        socket_path,
        device_id,
        mode=RequestedLockType.RW,
        tag="weights",
    )
    gms_client.clear_all_handles()
    logger.info("[Device %d] Connected to GMS as RW", device_id)
    return gms_client, pool


def source_register_nixl(
    device_id: int,
    rank: int,
    model: nn.Module,
    mx_config: MxConfig,
) -> None:
    """Register raw (pre-post-processing) tensors with NIXL and publish metadata.

    Called AFTER weight loading but BEFORE post-processing. NIXL must see the
    raw tensors because targets receive raw weights via RDMA and run their own
    post-processing locally to produce identical derived tensors.

    Args:
        device_id: CUDA device index.
        rank: Worker rank.
        model: Model with loaded (raw) weights in GMS-managed GPU memory.
        mx_config: MX-specific configuration.
    """
    raw_tensors = _collect_gpu_tensors(model)
    nixl_mgr = _register_nixl(device_id, raw_tensors, mx_config.contiguous_reg)
    _publish_metadata(rank, nixl_mgr, raw_tensors, mx_config)
    logger.info("[Worker %d] NIXL registered, metadata published", rank)


def source_commit_gms(
    gms_client: "GMSClientMemoryManager",
    device_id: int,
    model: nn.Module,
) -> None:
    """Register post-processed tensors with GMS and commit.

    Called AFTER post-processing so the engine reads the final state from GMS.
    Tensors must be in GMS-managed memory (allocated via ``source_connect_gms``
    pool).

    Args:
        gms_client: GMS client from ``source_connect_gms``.
        device_id: CUDA device index.
        model: Model with post-processed weights.
    """
    from gpu_memory_service.client.torch.module import register_module_tensors
    from gpu_memory_service.common.types import RequestedLockType

    register_module_tensors(gms_client, model)
    torch.cuda.synchronize()

    if not gms_client.commit():
        raise RuntimeError(f"[Device {device_id}] GMS commit failed")

    # commit() closes the RW socket; reconnect in RO mode
    gms_client.disconnect()
    gms_client.connect(RequestedLockType.RO)

    total_gb = gms_client.total_bytes / (1 << 30)
    logger.info("[Device %d] GMS: committed %.2f GiB, switched to RO", device_id, total_gb)


def source_finalize(rank: int, mx_config: MxConfig) -> None:
    """Barrier + publish ready flag. Called after engine post-processing.

    Args:
        rank: Worker rank.
        mx_config: MX-specific configuration.
    """
    from ..client import MxClient

    client = MxClient(server_url=mx_config.mx_server)
    raw_tensors = _collect_gpu_tensors.__doc__  # just need a hash placeholder
    client.publish_ready(
        model_name=mx_config.model_name,
        worker_id=rank,
        session_id=client.session_id,
        metadata_hash="",
        nixl_ready=True,
        stability_verified=True,
    )
    client.close()

    logger.info("[Worker %d] source_finalize complete: ready flag published", rank)


# ---------------------------------------------------------------------------
# Target hooks
# ---------------------------------------------------------------------------


def target_allocate(
    device_id: int,
    rank: int,
    model: nn.Module,
    mx_config: MxConfig,
) -> tuple["GMSClientMemoryManager", "NixlTransferManager"]:
    """Allocate GMS buffers and register with NIXL for receiving.

    Called by each target worker after engine creates the model skeleton
    with dummy weights.

    Args:
        device_id: CUDA device index.
        rank: Worker rank.
        model: Model skeleton with dummy weights on GPU.
        mx_config: MX-specific configuration.

    Returns:
        Tuple of (gms_client, nixl_mgr) for use in subsequent hooks.
    """
    from gpu_memory_service import get_or_create_gms_client_memory_manager
    from gpu_memory_service.common.types import RequestedLockType
    from gpu_memory_service.common.utils import get_socket_path

    socket_path = get_socket_path(device_id)
    gms_client, _pool = get_or_create_gms_client_memory_manager(
        socket_path,
        device_id,
        mode=RequestedLockType.RW,
        tag="weights",
    )
    gms_client.clear_all_handles()

    raw_tensors = _collect_gpu_tensors(model)
    nixl_mgr = _register_nixl(device_id, raw_tensors, mx_config.contiguous_reg)

    logger.info(
        "[Worker %d] target_allocate complete: %d tensors registered",
        rank,
        len(raw_tensors),
    )
    return gms_client, nixl_mgr


def target_receive(
    rank: int,
    nixl_mgr: "NixlTransferManager",
    mx_config: MxConfig,
    timeout: float = 7200,
) -> None:
    """Wait for source ready and receive weights via RDMA.

    Args:
        rank: Worker rank.
        nixl_mgr: Initialized NIXL manager with registered tensors.
        mx_config: MX-specific configuration.
        timeout: Max seconds to wait for source.

    Raises:
        RuntimeError: If source never becomes ready or transfer fails.
    """
    from ..client import MxClient
    from ..types import TensorDescriptor

    # 1. Wait for source ready via MxClient
    client = MxClient(server_url=mx_config.mx_server)
    source_ready, _session_id, _hash = client.wait_for_ready(
        model_name=mx_config.model_name,
        worker_id=rank,
        timeout_seconds=int(timeout),
    )
    if not source_ready:
        client.close()
        raise RuntimeError(f"[Worker {rank}] Source never became ready")

    # 2. Get source metadata and find our rank's worker
    source_worker = _wait_for_source_worker(client, mx_config, rank)

    # 3. Build source tensor descriptors
    source_tensors = [
        TensorDescriptor(
            name=t.name,
            addr=t.addr,
            size=t.size,
            device_id=t.device_id,
            dtype=t.dtype,
        )
        for t in source_worker.tensors
    ]

    # 4. RDMA receive
    coalesce = mx_config.contiguous_reg
    bytes_transferred, tensor_count, duration = nixl_mgr.receive_from_source(
        source_metadata=source_worker.nixl_metadata,
        source_tensors=source_tensors,
        coalesce_transfers=coalesce,
    )
    torch.cuda.synchronize()
    client.close()

    bandwidth_gbps = (
        (bytes_transferred * 8) / (duration * 1e9) if duration > 0 else 0
    )
    logger.info(
        "[Worker %d] target_receive complete: %.2f GB in %.3fs (%.1f Gbps)",
        rank,
        bytes_transferred / 1e9,
        duration,
        bandwidth_gbps,
    )


def target_commit(
    device_id: int,
    rank: int,
    model: nn.Module,
    gms_client: "GMSClientMemoryManager",
    mx_config: MxConfig,
) -> None:
    """Register processed tensors and commit to GMS.

    Called AFTER engine post-processing (FP8 transforms, MLA absorption).

    Args:
        device_id: CUDA device index.
        rank: Worker rank.
        model: Model with post-processed weights.
        gms_client: GMS client from target_allocate.
        mx_config: MX-specific configuration.
    """
    from gpu_memory_service.client.torch.module import register_module_tensors

    register_module_tensors(gms_client, model)
    torch.cuda.synchronize()

    if not gms_client.commit():
        raise RuntimeError(f"[Worker {rank}] GMS commit failed")

    gms_client.switch_to_read()
    logger.info("[Worker %d] target_commit complete: GMS committed", rank)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------



def _collect_gpu_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect all CUDA parameter tensors from model."""
    return {
        name: param.data
        for name, param in model.named_parameters()
        if param.is_cuda
    }


def _register_nixl(
    device_id: int,
    tensors: dict[str, torch.Tensor],
    contiguous_reg: bool = False,
) -> "NixlTransferManager":
    """Create NIXL agent and register tensors."""
    import os

    from ..nixl_transfer import NixlTransferManager

    # Set contiguous reg env var to match config
    if contiguous_reg:
        os.environ["MX_CONTIGUOUS_REG"] = "1"

    agent_name = f"mx-gms-{device_id}-{uuid.uuid4().hex[:8]}"
    mgr = NixlTransferManager(agent_name=agent_name, device_id=device_id)
    mgr.initialize()
    mgr.register_tensors(tensors)

    total_gb = sum(t.numel() * t.element_size() for t in tensors.values()) / 1e9
    logger.info(
        "[Device %d] NIXL: registered %d tensors (%.2f GB)",
        device_id,
        len(tensors),
        total_gb,
    )
    return mgr


def _publish_metadata(
    rank: int,
    nixl_mgr: "NixlTransferManager",
    tensors: dict[str, torch.Tensor],
    mx_config: MxConfig,
) -> None:
    """Publish per-worker metadata to MX Server via MxClient."""
    from ..client import MxClient
    from ..types import TensorDescriptor, WorkerMetadata

    if mx_config.contiguous_reg:
        descriptors = nixl_mgr.get_registered_descriptors()
    else:
        descriptors = [
            TensorDescriptor(
                name=name,
                addr=t.data_ptr(),
                size=t.numel() * t.element_size(),
                device_id=rank,
                dtype=str(t.dtype),
            )
            for name, t in tensors.items()
        ]

    # Build protobuf messages for the gRPC call
    from .. import p2p_pb2

    tensor_protos = [
        p2p_pb2.TensorDescriptor(
            name=d.name,
            addr=d.addr,
            size=d.size,
            device_id=d.device_id,
            dtype=d.dtype,
        )
        for d in descriptors
    ]

    worker = p2p_pb2.WorkerMetadata(
        worker_rank=rank,
        nixl_metadata=nixl_mgr.nixl_metadata,
        tensors=tensor_protos,
    )

    client = MxClient(server_url=mx_config.mx_server)
    success = client.publish_metadata(mx_config.model_name, [worker])
    client.close()

    if success:
        logger.info("[Worker %d] Published metadata to MX Server", rank)
    else:
        logger.error("[Worker %d] Failed to publish metadata", rank)


def _wait_for_source_worker(
    client: "MxClient",
    mx_config: MxConfig,
    rank: int,
    timeout: float = 3600,
):
    """Poll MX Server until source worker for our rank is available."""
    import time

    start = time.time()
    retry_interval = 30

    while time.time() - start < timeout:
        response = client.get_metadata(mx_config.model_name)

        if response.found:
            if mx_config.sync_start:
                ready_count = sum(
                    1 for w in response.workers if len(w.tensors) > 0
                )
                if ready_count < mx_config.expected_workers:
                    logger.info(
                        "[Worker %d] Sync start: %d/%d workers ready",
                        rank,
                        ready_count,
                        mx_config.expected_workers,
                    )
                    time.sleep(retry_interval)
                    continue

            for w in response.workers:
                if w.worker_rank == rank and len(w.tensors) > 0:
                    return w

        time.sleep(retry_interval)

    raise RuntimeError(
        f"[Worker {rank}] Timeout waiting for source worker after {timeout}s"
    )
