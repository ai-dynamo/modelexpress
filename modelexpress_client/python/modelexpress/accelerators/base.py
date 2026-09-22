# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accelerator backend boundary for device-specific operations."""

from __future__ import annotations

from typing import Protocol

import torch


NIXL_ACCELERATOR_MEM_TYPE = "VRAM"


class AcceleratorBackend(Protocol):
    """Boundary for torch device control and accelerator capabilities."""

    @property
    def name(self) -> str:
        """Backend family name for logs and capability policy, for example ``cuda``."""
        ...

    @property
    def torch_device_type(self) -> str:
        """Torch device type used to construct tensors, which may differ from ``name``."""
        ...

    @property
    def nixl_mem_type(self) -> str:
        """NIXL memory segment for accelerator memory."""
        ...

    def set_device(self, device_id: int) -> None:
        """Make ``device_id`` current for this backend."""
        ...

    def current_device(self) -> int:
        """Return the current local device ordinal."""
        ...

    def synchronize(self, device_id: int | None = None) -> None:
        """Synchronize backend work on ``device_id`` or the current device."""
        ...

    def empty_cache(self) -> None:
        """Release backend allocator cache where supported."""
        ...

    def torch_device(self, device_id: int) -> torch.device:
        """Return a torch device object for ``device_id``."""
        ...

    def is_accel_tensor(self, tensor: torch.Tensor) -> bool:
        """Return whether ``tensor`` lives on this backend's accelerator memory."""
        ...

    def supports_rdma_p2p(self) -> bool:
        """Return whether this backend supports NIXL RDMA P2P transfers."""
        ...

    def supports_pool_reg(self) -> bool:
        """Return whether allocation-level NIXL pool registration is supported."""
        ...

    def supports_vmm(self) -> bool:
        """Return whether the CUDA VMM arena fast path is supported."""
        ...

    def supports_gds(self) -> bool:
        """Return whether GPUDirect Storage loading is supported."""
        ...
