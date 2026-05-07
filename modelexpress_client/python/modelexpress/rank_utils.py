# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank detection utilities."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("modelexpress.rank_utils")


def get_global_rank(device: torch.device) -> int:
    """Get the global distributed rank for this worker."""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
            logger.debug(f"Got global rank from torch.distributed: {rank}")
            return rank
    except (ImportError, RuntimeError) as e:
        logger.debug(f"Could not get global rank from torch.distributed: {e}")

    if hasattr(device, "index") and device.index is not None:
        logger.debug(f"Using device.index as global rank fallback: {device.index}")
        return device.index

    return 0
