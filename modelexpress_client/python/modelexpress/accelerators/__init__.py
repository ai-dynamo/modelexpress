# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accelerator backend abstraction for device-specific operations."""

from __future__ import annotations

import torch

from .base import NIXL_ACCELERATOR_MEM_TYPE, AcceleratorBackend
from .cuda import CudaAcceleratorBackend

__all__ = [
    "NIXL_ACCELERATOR_MEM_TYPE",
    "AcceleratorBackend",
    "CudaAcceleratorBackend",
    "accelerator_backend_for",
]


def accelerator_backend_for(device: torch.device | str) -> AcceleratorBackend:
    """Return the backend implementation for ``device``.

    Only CUDA devices are currently supported by this factory.
    """
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        return CudaAcceleratorBackend()
    raise ValueError(f"Unsupported accelerator backend for torch device {torch_device!s}")
