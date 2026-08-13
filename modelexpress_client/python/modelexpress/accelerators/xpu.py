# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XPU implementation of the accelerator backend boundary."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import NIXL_ACCELERATOR_MEM_TYPE


@dataclass(frozen=True)
class XpuAcceleratorBackend:
    """XPU implementation of the accelerator backend boundary."""

    @property
    def name(self) -> str:
        return "xpu"

    @property
    def torch_device_type(self) -> str:
        return "xpu"

    @property
    def nixl_mem_type(self) -> str:
        return NIXL_ACCELERATOR_MEM_TYPE

    def _xpu(self):
        xpu = getattr(torch, "xpu", None)
        if xpu is None:
            raise RuntimeError("torch.xpu is not available")
        return xpu

    def set_device(self, device_id: int) -> None:
        self._xpu().set_device(device_id)

    def current_device(self) -> int:
        return int(self._xpu().current_device())

    def synchronize(self, device_id: int | None = None) -> None:
        xpu = self._xpu()
        if device_id is None:
            xpu.synchronize()
            return

        try:
            xpu.synchronize(device_id)
        except TypeError:
            current_device = int(xpu.current_device())
            xpu.set_device(device_id)
            try:
                xpu.synchronize()
            finally:
                xpu.set_device(current_device)

    def empty_cache(self) -> None:
        empty_cache = getattr(self._xpu(), "empty_cache", None)
        if callable(empty_cache):
            empty_cache()

    def torch_device(self, device_id: int) -> torch.device:
        return torch.device(self.torch_device_type, device_id)

    def is_accel_tensor(self, tensor: torch.Tensor) -> bool:
        return tensor.device.type == self.torch_device_type

    def supports_rdma_p2p(self) -> bool:
        return True

    def supports_pool_reg(self) -> bool:
        return False

    def supports_vmm(self) -> bool:
        return False

    def supports_gds(self) -> bool:
        return False

    def requires_classic_alloc_pool(self) -> bool:
        # The classic pool works around a CUDA-specific interaction between
        # expandable-segment allocations and nvidia_peermem pinning. No equivalent
        # hazard is known or has been observed on XPU, so registered buffers use
        # the default allocator.
        #
        # Not a proof of absence. The CUDA failure mode passes ibv_reg_mr and only
        # fails at WRITE time, so a successful registration does not rule it out;
        # ruling it out would take a real RDMA write into an XPU buffer allocated
        # under whatever expandable-segment equivalent XPU has, if it has one.
        # torch.xpu does expose MemPool and XPUPluggableAllocator, so an
        # alternate XPU pool could be implemented here if that test ever says one
        # is needed.
        return False
