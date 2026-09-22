# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accelerator backend abstraction for device-specific operations."""

from __future__ import annotations

import torch

from .base import NIXL_ACCELERATOR_MEM_TYPE, AcceleratorBackend
from .cuda import CudaAcceleratorBackend
from .xpu import XpuAcceleratorBackend

__all__ = [
    "NIXL_ACCELERATOR_MEM_TYPE",
    "AcceleratorBackend",
    "CudaAcceleratorBackend",
    "XpuAcceleratorBackend",
    "accelerator_backend_for",
]


def _is_torch_xpu_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    is_available = getattr(xpu, "is_available", None)
    if is_available is None:
        return False
    try:
        return bool(is_available())
    except Exception:
        return False


def _supported_device_types() -> str:
    supported = ["cuda"]
    if _is_torch_xpu_available():
        supported.append("xpu")
    return ", ".join(supported)


def accelerator_backend_for(device: torch.device | str) -> AcceleratorBackend:
    """Return the backend implementation for ``device``.

    CUDA is always selectable. XPU is selectable when torch exposes an
    available ``torch.xpu`` runtime.
    """
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        return CudaAcceleratorBackend()
    if torch_device.type == "xpu":
        if _is_torch_xpu_available():
            return XpuAcceleratorBackend()
        raise ValueError(
            "Unsupported accelerator backend for torch device "
            f"{torch_device!s}: torch.xpu is not available; "
            f"supported device types: {_supported_device_types()}"
        )
    raise ValueError(
        "Unsupported accelerator backend for torch device "
        f"{torch_device!s}; supported device types: {_supported_device_types()}"
    )
