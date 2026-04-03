# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank detection and NIXL manager initialization utilities."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .nixl_transfer import NixlTransferManager

logger = logging.getLogger("modelexpress.rank_utils")


def get_global_rank(device: torch.device) -> int:
    """Get the global rank of this worker across all TP and PP groups."""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
            logger.debug(f"Got global rank from torch.distributed: {rank}")
            return rank
    except (ImportError, RuntimeError) as e:
        logger.debug(f"Could not get global rank from torch.distributed: {e}")

    if hasattr(device, "index") and device.index is not None:
        logger.debug(f"Using device.index as rank: {device.index}")
        return device.index

    return 0


def get_worker_rank(device: torch.device) -> int:
    """Get the local CUDA device ordinal (TP rank) of this worker."""
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        rank = get_tensor_model_parallel_rank()
        logger.debug(f"Got TP rank from vllm.distributed: {rank}")
        return rank
    except (ImportError, RuntimeError) as e:
        logger.debug(f"Could not get TP rank from vllm.distributed: {e}")

    if hasattr(device, "index") and device.index is not None:
        logger.debug(f"Using device.index as rank: {device.index}")
        return device.index

    return 0


def init_nixl_manager(
    global_rank: int, device_id: int, role: str, listen_port: int = 0,
) -> "NixlTransferManager":
    """Create and initialize a NIXL transfer manager."""
    from .nixl_transfer import NixlTransferManager

    agent_name = f"mx-{role}-worker{global_rank}-{uuid.uuid4().hex[:8]}"
    logger.debug(f"[Worker {global_rank}] Initializing NIXL manager with agent_name={agent_name}")
    manager = NixlTransferManager(
        agent_name=agent_name,
        device_id=device_id,
        listen_port=listen_port,
    )
    manager.initialize()
    logger.debug(f"[Worker {global_rank}] NIXL manager initialized")
    return manager
