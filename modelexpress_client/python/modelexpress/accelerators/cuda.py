# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA implementation of the accelerator backend boundary."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import NIXL_ACCELERATOR_MEM_TYPE


@dataclass(frozen=True)
class CudaAcceleratorBackend:
    """CUDA implementation of the accelerator backend boundary."""

    @property
    def name(self) -> str:
        return "cuda"

    @property
    def torch_device_type(self) -> str:
        return "cuda"

    @property
    def nixl_mem_type(self) -> str:
        return NIXL_ACCELERATOR_MEM_TYPE

    def set_device(self, device_id: int) -> None:
        torch.cuda.set_device(device_id)

    def current_device(self) -> int:
        return int(torch.cuda.current_device())

    def synchronize(self, device_id: int | None = None) -> None:
        if device_id is None:
            torch.cuda.synchronize()
        else:
            torch.cuda.synchronize(device_id)

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def torch_device(self, device_id: int) -> torch.device:
        return torch.device(self.torch_device_type, device_id)

    def is_accel_tensor(self, tensor: torch.Tensor) -> bool:
        return bool(tensor.is_cuda)

    def supports_rdma_p2p(self) -> bool:
        return True

    def supports_pool_reg(self) -> bool:
        return True

    def supports_vmm(self) -> bool:
        return True

    def supports_gds(self) -> bool:
        return True
